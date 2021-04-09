import cv2
import taichi as ti
import pathlib

MAX_DIST = 2147483647
null = ti.Vector([-1, -1, MAX_DIST])
vec3 = lambda scalar: ti.Vector([scalar, scalar, scalar])


@ti.data_oriented
class SDF2D:
    def __init__(self, filename, multiple_files=False):
        ti.init(arch=ti.gpu, kernel_profiler=True, debug=False, print_ir=False)
        self.filename = filename
        self.out_filename = self.output_filename()
        self.multiple_files = multiple_files
        if not multiple_files:
            self.num = 0  # index of bit_pic

            self.im = cv2.imread(filename)
            self.width, self.height = self.im.shape[1], self.im.shape[0]
            self.pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
            self.bit_pic = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
            self.output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
            self.max_reduction = ti.field(dtype=ti.i32, shape=self.width * self.height)

            self.pic.from_numpy(self.im)
            self.pre_process()

    def output_filename(self):
        path = pathlib.Path(self.filename)
        out_dir = path.parent / 'output'
        if not (out_dir.exists() and out_dir.is_dir()):
            out_dir.mkdir()
        return str(out_dir / (path.stem + '_sdf' + path.suffix))

    @ti.kernel
    def pre_process(self):
        for i, j in self.pic:
            if self.pic[i, j][0] > 128:
                self.bit_pic[0, i, j] = ti.Vector([i, j, 0])
                self.bit_pic[1, i, j] = ti.Vector([i, j, 0])
            else:
                self.bit_pic[0, i, j] = null
                self.bit_pic[1, i, j] = null

    @ti.func
    def cal_dist_sqr(self, p1_x, p1_y, p2_x, p2_y):
        return (p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2

    @ti.kernel
    def jump_flooding(self, stride: ti.i32, n: ti.i32):
        # print('n =', n, '\n')
        for i, j in ti.ndrange(self.width, self.height):
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                i_off = i + stride * di
                j_off = j + stride * dj
                if 0 <= i_off < self.width and 0 <= j_off < self.height:
                    dist_sqr = self.cal_dist_sqr(i, j, self.bit_pic[n, i_off, j_off][0],
                                                 self.bit_pic[n, i_off, j_off][1])
                    # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr,', ', i_off, j_off)
                    if not self.bit_pic[n, i_off, j_off][0] < 0 and dist_sqr < self.bit_pic[1 - n, i, j][2]:
                        self.bit_pic[1 - n, i, j][0] = self.bit_pic[n, i_off, j_off][0]
                        self.bit_pic[1 - n, i, j][1] = self.bit_pic[n, i_off, j_off][1]
                        self.bit_pic[1 - n, i, j][2] = dist_sqr
                        # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr, ', ', i_off, j_off)

    @ti.kernel
    def post_process(self, n: ti.i32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3(ti.cast(ti.sqrt(self.bit_pic[n, i, j][2]) / self.max_dist, ti.u32))

    @ti.kernel
    def copy(self):
        for i, j in ti.ndrange(self.width, self.height):
            self.max_reduction[i * self.width + j] = self.bit_pic[self.num, i, j][2]

    @ti.kernel
    def max_reduction_kernel(self, r_stride: ti.i32):
        for i in range(r_stride):
            self.max_reduction[i] = max(self.max_reduction[i], self.max_reduction[i + r_stride])

    # @ti.kernel
    # def print_p(self, n: ti.i32):
    #     print(n, '\n')
    #     for i, j in ti.ndrange(self.width, self.height):
    #         print('i:', i, 'j:', j, 'store:', self.bit_pic[n, i, j][0], self.bit_pic[n, i, j][1],
    #               self.bit_pic[n, i, j][2])
    #     print('\n')

    def mask2udf(self, normalized=True, output=False):  # unsigned distance
        stride = self.width >> 1
        while stride > 0:
            self.jump_flooding(stride, self.num)
            stride >>= 1
            self.num = 1 - self.num

        # self.jump_flooding(2, self.num)
        # self.num = 1 - self.num

        self.jump_flooding(1, self.num)
        self.num = 1 - self.num

        self.copy()

        r_stride = self.width * self.height >> 1
        while r_stride > 0:
            self.max_reduction_kernel(r_stride)
            r_stride >>= 1

        self.max_dist = ti.sqrt(self.max_reduction[0]) / 255.0

        self.post_process(self.num)

        if output:
            cv2.imwrite(self.out_filename, self.output_pic.to_numpy())


filename = r"test_file/test_2048.png"

mySDF2D = SDF2D(filename)
mySDF2D.mask2udf(output=True)

ti.kernel_profiler_print()
