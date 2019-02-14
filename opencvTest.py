# -*- coding:utf-8 -*- 

# URL: https://zhuanlan.zhihu.com/p/24425116

# 根据功能和需求的不同，OpenCV中的函数接口大体可以分为如下部分：

# - core：核心模块，主要包含了OpenCV中最基本的结构（矩阵，点线和形状等），以及相关的基础运算/操作。

# - imgproc：图像处理模块，包含和图像相关的基础功能（滤波，梯度，改变大小等），以及一些衍生的高级功能（图像分割，直方图，形态分析和边缘/直线提取等）。

# - highgui：提供了用户界面和文件读取的基本函数，比如图像显示窗口的生成和控制，图像/视频文件的IO等。

# 如果不考虑视频应用，以上三个就是最核心和常用的模块了。

# 矩阵就用numpy的array表示。

# OpenCV的这个特殊之处还是需要注意的，比如在Python中，图像都是用numpy的array表示，但是同样的array在OpenCV中的显示效果和matplotlib中的显示效果就会不一样。下面的简单代码就可以生成两种表示方式


# # Part1: Expression of Picture 

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# # 图6-1中的矩阵
# img = np.array([
#     [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
#     [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
#     [[255, 255, 255], [128, 128, 128], [0, 0, 0]],
# ], dtype=np.uint8)

# # 用matplotlib存储
# plt.imsave('img_pyplot.jpg', img)

# # 用OpenCV存储
# cv2.imwrite('img_cv2.jpg', img)

# # 不管是RGB还是BGR，都是高度×宽度×通道数，H×W×C的表达方式，而在深度学习中，因为要对不同通道应用卷积，所以用的是另一种方式：C×H×W，就是把每个通道都单独表达成一个二维矩阵

# # 读图像用cv2.imread()，可以按照不同模式读取，一般最常用到的是读取单通道灰度图，或者直接默认读取多通道。存图像用cv2.imwrite()，注意存的时候是没有单通道这一说的，根据保存文件名的后缀和当前的array维度，OpenCV自动判断存的通道，另外压缩格式还可以指定存储质量

# # Part2: Saving of Picture 

# import cv2

# # 读取一张400x600分辨率的图像
# color_img = cv2.imread('exp.jpg')
# print(color_img.shape)

# # 直接读取单通道
# gray_img = cv2.imread('exp.jpg', cv2.IMREAD_GRAYSCALE)
# print(gray_img.shape)

# # 把单通道图片保存后，再读取，仍然是3通道，相当于把单通道值复制到3个通道保存
# cv2.imwrite('exp_grayscale.jpg', gray_img)
# reload_grayscale = cv2.imread('exp_grayscale.jpg')
# print(reload_grayscale.shape)

# # cv2.IMWRITE_JPEG_QUALITY指定jpg质量，范围0到100，默认95，越高画质越好，文件越大
# cv2.imwrite('test_imwrite.jpg', color_img, (cv2.IMWRITE_JPEG_QUALITY, 80))

# # cv2.IMWRITE_PNG_COMPRESSION指定png质量，范围0到9，默认3，越高文件越小，画质越差
# cv2.imwrite('test_imwrite.png', color_img, (cv2.IMWRITE_PNG_COMPRESSION, 5))


# # Part3: 缩放通过cv2.resize()实现，裁剪则是利用array自身的下标截取实现，此外OpenCV还可以给图像补边，这样能对一幅图像的形状和感兴趣区域实现各种操作。

# import cv2

# img = cv2.imread('exp.jpg')

# # 缩放成200x200的方形图像
# img_200x200 = cv2.resize(img, (200, 200))

# # 不直接指定缩放后大小，通过fx和fy指定缩放比例，0.5则长宽都为原来一半
# # 等效于img_200x300 = cv2.resize(img, (300, 200))，注意指定大小的格式是(宽度,高度)
# # 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
# img_200x300 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, 
#                               interpolation=cv2.INTER_NEAREST)

# # 在上张图片的基础上，上下各贴50像素的黑边，生成300x300的图像
# img_300x300 = cv2.copyMakeBorder(img, 50, 50, 0, 0, 
#                                        cv2.BORDER_CONSTANT, 
#                                        value=(0, 0, 0))

# # 对照片中树的部分进行剪裁
# patch_people_hair = img[20:150, -180:-50]

# cv2.imwrite('cropped_hair.jpg', patch_people_hair)
# cv2.imwrite('resized_200x200.jpg', img_200x200)
# cv2.imwrite('resized_200x300.jpg', img_200x300)
# cv2.imwrite('bordered_300x300.jpg', img_300x300)

# Part 4: 色调，明暗，直方图和Gamma曲线

# 比如可以通过HSV空间对色调和明暗进行调节。HSV空间是由美国的图形学专家A. R. Smith提出的一种颜色空间，HSV分别是色调（Hue），饱和度（Saturation）和明度（Value）。在HSV空间中进行调节就避免了直接在RGB空间中调节是还需要考虑三个通道的相关性。OpenCV中H的取值是[0, 180)，其他两个通道的取值都是[0, 256)



