import cv2
import taichi as ti

MAX_DIST = 2147483647

ti.init(arch=ti.gpu, kernel_profiler=True, debug=False)

im = cv2.imread(r"test_file/test_2048.png")
width, height = im.shape[1], im.shape[0]

pic = ti.Vector.field(3, dtype=ti.i32, shape=(width, height))
bit_pic = ti.Vector.field(3, dtype=ti.i32, shape=(2, width, height))
output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(width, height))
max_reduction = ti.field(dtype=ti.i32, shape=width * height)
maximum = ti.field(dtype=ti.i32, shape=())

pic.from_numpy(im)
null = ti.Vector([-1, -1, MAX_DIST])

vec3 = lambda scalar: ti.Vector([scalar, scalar, scalar])


@ti.kernel
def pre_process():
    for i, j in pic:
        if pic[i, j][0] > 128:
            bit_pic[0, i, j] = ti.Vector([i, j, 0])
            bit_pic[1, i, j] = ti.Vector([i, j, 0])
        else:
            bit_pic[0, i, j] = null
            bit_pic[1, i, j] = null


@ti.func
def cal_dist_sqr(p1_x, p1_y, p2_x, p2_y):
    return (p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2


@ti.kernel
def jump_flooding(stride: ti.i32, n: ti.i32):
    # print('n =', n, '\n')
    for i, j in ti.ndrange(width, height):
        for di, dj in ti.ndrange((-1, 2), (-1, 2)):
            i_off = i + stride * di
            j_off = j + stride * dj
            if 0 <= i_off < width and 0 <= j_off < height:
                dist_sqr = cal_dist_sqr(i, j, bit_pic[n, i_off, j_off][0], bit_pic[n, i_off, j_off][1])
                # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr,', ', i_off, j_off)
                if not bit_pic[n, i_off, j_off][0] < 0 and dist_sqr < bit_pic[1 - n, i, j][2]:
                    bit_pic[1 - n, i, j][0] = bit_pic[n, i_off, j_off][0]
                    bit_pic[1 - n, i, j][1] = bit_pic[n, i_off, j_off][1]
                    bit_pic[1 - n, i, j][2] = dist_sqr
                    # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr, ', ', i_off, j_off)


@ti.kernel
def post_process(n: ti.i32):
    for i, j in output_pic:
        output_pic[i, j] = vec3(ti.cast(ti.sqrt(bit_pic[n, i, j][2]) / max_dist, ti.u32))


@ti.kernel
def copy():
    for i, j in ti.ndrange(width, height):
        max_reduction[i * width + j] = bit_pic[num, i, j][2]


@ti.kernel
def max_reduction_kernel(r_stride: ti.i32):
    for i in range(r_stride):
        max_reduction[i] = max(max_reduction[i], max_reduction[i + r_stride])

# @ti.kernel
# def print_p(n: ti.i32):
#     print(n, '\n')
#     for i, j in ti.ndrange(width, height):
#         print('i:', i, 'j:', j, 'store:', bit_pic[n, i, j][0], bit_pic[n, i, j][1], bit_pic[n, i, j][2])
#     print('\n')


pre_process()

stride = width >> 1
num = 0
while stride > 0:
    jump_flooding(stride, num)
    stride >>= 1
    num = 1 - num

# jump_flooding(2, num)
# num = 1 - num

jump_flooding(1, num)
num = 1 - num

copy()

r_stride = width * height >> 1
while r_stride > 0:
    max_reduction_kernel(r_stride)
    r_stride >>= 1

max_dist = ti.sqrt(max_reduction[0]) / 255.0

post_process(num)

ti.kernel_profiler_print()

cv2.imwrite(r"output/test_{}_output.png".format(width), output_pic.to_numpy())
