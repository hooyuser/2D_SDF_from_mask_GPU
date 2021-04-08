import numpy as np
import cv2
import taichi as ti
import math

MAX_DIST = 2147483647

ti.init(arch=ti.gpu, kernel_profiler=True, debug=True)

im = cv2.imread(r"test_file/test_2048.png")
width, height = im.shape[1], im.shape[0]

pic = ti.Vector.field(3, dtype=ti.i32, shape=(width, height))
bit_pic = ti.Vector.field(3, dtype=ti.i32, shape=(width, height))
maximum = ti.field(dtype=ti.i32, shape=())

pic.from_numpy(im)
null = ti.Vector([-1, -1, MAX_DIST])

vec3 = lambda scalar: ti.Vector([scalar, scalar, scalar])


@ti.kernel
def pre_process():
    for i, j in pic:
        if pic[i, j][0] > 128:
            bit_pic[i, j] = ti.Vector([i, j, 0])
        else:
            bit_pic[i, j] = null


@ti.func
def cal_dist_sqr(p1_x, p1_y, p2_x, p2_y):
    return (p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2


@ti.kernel
def jump_flooding(stride: ti.i32):
    for i, j in bit_pic:
        for di, dj in ti.ndrange((-1, 2), (-1, 2)):
            i_off = i + stride * di
            j_off = j + stride * dj
            if 0 <= i_off < width and 0 <= j_off < height:
                if not bit_pic[i_off, j_off][0] < 0 and bit_pic[i_off, j_off][2] < MAX_DIST:
                    dist_sqr = cal_dist_sqr(i, j, bit_pic[i_off, j_off][0], bit_pic[i_off, j_off][1])
                    if dist_sqr < bit_pic[i, j][2]:
                        bit_pic[i, j][0] = bit_pic[i_off, j_off][0]
                        bit_pic[i, j][1] = bit_pic[i_off, j_off][1]
                        bit_pic[i, j][2] = dist_sqr


@ti.kernel
def post_process():
    for _ in range(1):
        for idx_i in range(width):
            for idx_j in range(height):
                maximum[None] = max(bit_pic[idx_i, idx_j][2], maximum[None])
    max_dist = ti.sqrt(maximum[None]) / 255.0
    for i, j in bit_pic:
        bit_pic[i, j] = vec3(ti.cast(ti.sqrt(bit_pic[i, j][2]) / max_dist, ti.u32))


pre_process()

stride = width >> 1
while stride > 0:
    jump_flooding(stride)
    stride >>= 1
jump_flooding(1)

# jump_flooding(stride)

post_process()
ti.kernel_profiler_print()

cv2.imwrite(r"output/test_{}_output.png".format(width), bit_pic.to_numpy())
