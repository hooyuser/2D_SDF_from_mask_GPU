import cv2
import taichi as ti
import pathlib

ti.init(arch=ti.gpu, kernel_profiler=True, debug=True, print_ir=False)

MAX_DIST = 2147483647
null = ti.Vector([-1, -1, MAX_DIST])
vec3 = lambda scalar: ti.Vector([scalar, scalar, scalar])


@ti.data_oriented
class SDF2D:
    def __init__(self, filename):
        self.filename = filename
        self.num = 0  # index of bit_pic

        self.im = cv2.imread(filename)
        self.width, self.height = self.im.shape[1], self.im.shape[0]
        self.pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.bit_pic_white = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.bit_pic_black = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.output_linear = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        self.max_reduction = ti.field(dtype=ti.i32, shape=self.width * self.height)

    def output_filename(self, ins):
        path = pathlib.Path(self.filename)
        out_dir = path.parent / 'output'
        if not (out_dir.exists() and out_dir.is_dir()):
            out_dir.mkdir()
        return str(out_dir / (path.stem + ins + path.suffix))

    @ti.kernel
    def pre_process(self, bit_pic: ti.template(), keep_white: ti.i32):  # keep_white, 1 == True, -1 == False
        for i, j in self.pic:
            if (self.pic[i, j][0] - 127) * keep_white > 0:
                bit_pic[0, i, j] = ti.Vector([i, j, 0])
                bit_pic[1, i, j] = ti.Vector([i, j, 0])
            else:
                bit_pic[0, i, j] = null
                bit_pic[1, i, j] = null

    @ti.func
    def cal_dist_sqr(self, p1_x, p1_y, p2_x, p2_y):
        return (p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2

    @ti.kernel
    def jump_flooding(self, bit_pic: ti.template(), stride: ti.i32, n: ti.i32):
        # print('n =', n, '\n')
        for i, j in ti.ndrange(self.width, self.height):
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                i_off = i + stride * di
                j_off = j + stride * dj
                if 0 <= i_off < self.width and 0 <= j_off < self.height:
                    dist_sqr = self.cal_dist_sqr(i, j, bit_pic[n, i_off, j_off][0],
                                                 bit_pic[n, i_off, j_off][1])
                    # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr,', ', i_off, j_off)
                    if not bit_pic[n, i_off, j_off][0] < 0 and dist_sqr < bit_pic[1 - n, i, j][2]:
                        bit_pic[1 - n, i, j][0] = bit_pic[n, i_off, j_off][0]
                        bit_pic[1 - n, i, j][1] = bit_pic[n, i_off, j_off][1]
                        bit_pic[1 - n, i, j][2] = dist_sqr
                        # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr, ', ', i_off, j_off)

    @ti.kernel
    def copy(self, bit_pic: ti.template()):
        for i, j in ti.ndrange(self.width, self.height):
            self.max_reduction[i * self.width + j] = bit_pic[self.num, i, j][2]

    @ti.kernel
    def max_reduction_kernel(self, r_stride: ti.i32):
        for i in range(r_stride):
            self.max_reduction[i] = max(self.max_reduction[i], self.max_reduction[i + r_stride])

    @ti.kernel
    def post_process_udf(self, bit_pic: ti.template(), n: ti.i32, coff: ti.f32, offset: ti.f32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3(ti.cast(ti.sqrt(bit_pic[n, i, j][2]) * coff + offset, ti.u32))

    @ti.kernel
    def post_process_sdf(self, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32, coff: ti.f32,
                         offset: ti.f32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3(
                ti.cast((ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])) * coff + offset, ti.u32))

    @ti.kernel
    def post_process_sdf_linear_1channel(self, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32):
        for i, j in self.output_pic:
            self.output_linear[i, j][0] = ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])

    # @ti.kernel
    # def print_p(self, n: ti.i32):
    #     print(n, '\n')
    #     for i, j in ti.ndrange(self.width, self.height):
    #         print('i:', i, 'j:', j, 'store:', self.bit_pic[n, i, j][0], self.bit_pic[n, i, j][1],
    #               self.bit_pic[n, i, j][2])
    #     print('\n')

    def gen_udf(self, dist_buffer, keep_white=True):
        self.pic.from_numpy(self.im)
        keep_white_para = 1 if keep_white else -1
        self.pre_process(dist_buffer, keep_white_para)
        self.num = 0
        stride = self.width >> 1
        while stride > 0:
            self.jump_flooding(dist_buffer, stride, self.num)
            stride >>= 1
            self.num = 1 - self.num

        # self.jump_flooding(self.bit_pic_white, 2, self.num)
        # self.num = 1 - self.num

        self.jump_flooding(dist_buffer, 1, self.num)
        self.num = 1 - self.num

    def find_max(self, dist_buffer):
        self.copy(dist_buffer)

        r_stride = self.width * self.height >> 1
        while r_stride > 0:
            self.max_reduction_kernel(r_stride)
            r_stride >>= 1

        return self.max_reduction[0]

    def mask2udf(self, normalized=(0, 1), to_rgb=True, output=True):  # unsigned distance
        self.gen_udf(self.bit_pic_white)

        max_dist = ti.sqrt(self.find_max(self.bit_pic_white))

        if to_rgb:  # scale sdf proportionally to [0, 1]
            coefficient = 255.0 / max_dist
            offset = 0.0
        else:
            coefficient = (normalized[1] - normalized[0]) / max_dist
            offset = normalized[0]

        self.post_process_udf(self.bit_pic_white, self.num, coefficient, offset)
        if output:
            if to_rgb:
                cv2.imwrite(self.output_filename('_udf'), self.output_pic.to_numpy())

    def mask2sdf(self, to_rgb=True, output=True):
        self.gen_udf(self.bit_pic_white, keep_white=True)
        self.gen_udf(self.bit_pic_black, keep_white=False)

        if to_rgb:  # grey value == 0.5 means sdf == 0, scale sdf proportionally
            max_positive_dist = ti.sqrt(self.find_max(self.bit_pic_white))
            min_negative_dist = ti.sqrt(self.find_max(self.bit_pic_black))  # this value is positive
            coefficient = 127.5 / max(max_positive_dist, min_negative_dist)
            offset = 127.5
            self.post_process_sdf(self.bit_pic_white, self.bit_pic_black, self.num, coefficient, offset)
            if output:
                cv2.imwrite(self.output_filename('_sdf'), self.output_pic.to_numpy())
        else:  # no normalization
            if output:
                pass
            else:
                self.post_process_sdf_linear_1channel(self.bit_pic_white, self.bit_pic_black, self.num)


@ti.data_oriented
class MultiSDF2D:
    def __init__(self, file_name, file_num, sample_num=256):
        self.file_name = file_name
        self.file_path = pathlib.Path(file_name)
        self.file_num = file_num
        self.sample_num = sample_num
        self.name_base = self.file_path.stem[:-2]
        self.sdf_list = self.gen_sdf_list()
        self.width, self.height = self.sdf_list[0].width, self.sdf_list[0].height
        self.output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))

    def output_filename(self, ins='output'):
        out_dir = self.file_path.parent / 'output'
        if not (out_dir.exists() and out_dir.is_dir()):
            out_dir.mkdir()
        return str(out_dir / (self.name_base + ins + self.file_path.suffix))

    def gen_sdf_list(self):
        lst = []
        for i in range(self.file_num):
            name = str(self.file_path.parent / f'{self.name_base}_{i + 1}{self.file_path.suffix}')
            lst.append(SDF2D(name))
        return lst

    def blur_mix_sdf(self):
        for sdf in self.sdf_list:
            sdf.mask2sdf(to_rgb=False, output=False)
        self.blur_mix(self.sdf_list[0].output_linear, self.sdf_list[1].output_linear)
        cv2.imwrite(self.output_filename('_blur_mix'), self.output_pic.to_numpy())

    @ti.kernel
    def blur_mix(self, sdf1: ti.template(), sdf2: ti.template()):
        for i, j in self.output_pic:
            dis1 = sdf1[i, j][0]
            dis2 = sdf2[i, j][0]
            if dis1 < 0.4999 and dis2 < 0.4999:
                self.output_pic[i, j] = vec3(255)
            elif dis1 > 0.5 and dis2 > 0.5:
                self.output_pic[i, j] = vec3(0)
            else:
                res = -1
                for n in range(self.sample_num):
                    mix = n / self.sample_num
                    if (1 - mix) * dis1 + mix * dis2 < 0.4999:
                        res += 256
                self.output_pic[i, j] = vec3(res // self.sample_num)


img_name = r"test_file/test_1024_1.png"

# mySDF2D = SDF2D(img_name)
# mySDF2D.mask2sdf()

myMultiSDF2D = MultiSDF2D(img_name, 2)
myMultiSDF2D.blur_mix_sdf()

ti.kernel_profiler_print()
