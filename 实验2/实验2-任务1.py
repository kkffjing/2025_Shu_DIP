
import cv2 as cv
import numpy as np
import time
"""最近邻插值放大图片"""
def nearest_neighbor(img,scale):
    h_origin,w_origin = img.shape[:2]
    h_new=h_origin*scale
    w_new=w_origin*scale#得到放缩后图片尺寸，创建新图像
    new_im=np.zeros((h_new,w_new,3),np.uint8)
    for y in range(h_new):
        for x in range(w_new):
            x1=int(np.floor(x/scale))#反向映射
            y1=int(np.floor(y/scale))#新图像像素点在原图中对应的像素点最近邻
            x1=min(x1,w_origin-1)
            y1=min(y1,h_origin-1)#边界保护
            new_im[y,x]=img[y1,x1]#将最近邻像素值赋给新图像
    return new_im
def bilinear(img,scale):
    h_origin,w_origin = img.shape[:2]
    h_new = h_origin*scale
    w_new = w_origin*scale
    channels=img.shape[2]#像素通道数，方便后续用循环实现
    new_im=np.zeros((h_new,w_new,3),np.uint8)
    for y in range(h_new):
        for x in range(w_new):
            x1=int(np.floor(x/scale))#向下取整，防止超过原像素范围限制
            y1=int(np.floor(y/scale))
            x2=x1+1
            y2=y1+1#取四个最近的点

            x1=min(x1,w_origin-1)
            y1=min(y1,h_origin-1)
            x2=min(x2,w_origin-1)
            y2=min(y2,h_origin-1)#边界检测，防止超过原像素范围限制
            c11=img[y1,x1]
            c12=img[y1,x2]
            c21=img[y2,x1]
            c22=img[y2,x2]#取四个最近点的像素值
            dx = x / scale - x1
            dy = y / scale - y1
            for c in range(channels):
                new_im[y, x, c] =(1 - dx) * (1 - dy) * c11[c] + (1 - dx) * dy * c12[c] + dx * (1 - dy) * c21[c] + dx * dy * c22[c]
                #在x,y两个方向上做双线性插值计算（公式进行简化）计算每个像素通道上对应的像素值，赋值给放缩后的图像
    return new_im

"""导入所给图片"""
im1 = cv.imread('test1.jpg', 1)
im1_origin=cv.imread('test1_origin.jpg', 1)
im2 = cv.imread('test2.jpg', 1)
im2_origin=cv.imread('test2_origin.jpg', 1)
height1_origin, width1_origin = im1.shape[:2]
height2_origin, width2_origin = im2.shape[:2]
"""设置放大倍数"""
scale=2
"""test1"""
start_time0 = time.time()#放置计时器，比较代码的运行速度
resize_nn1=nearest_neighbor(im1,scale)
time_nn1 = time.time()-start_time0
cv.imwrite('resize_nn1.jpg',resize_nn1)

start_time1 = time.time()
resize_b1=bilinear(im1,scale)
time_b1= time.time()-start_time1
cv.imwrite('resize_b1.jpg',resize_b1)

start_time2 = time.time()
resize_cv_nn1=cv.resize(im1,(2*width1_origin,2*height1_origin),interpolation=cv.INTER_NEAREST)
time_cv_nn1 = time.time()-start_time2
cv.imwrite('resize_cv_nn1.jpg',resize_cv_nn1)

start_time3 = time.time()
resize_cv_b1=cv.resize(im1,(2*width1_origin,2*height1_origin),interpolation=cv.INTER_LINEAR)
time_cv_b1 = time.time()-start_time3
cv.imwrite('resize_cv_b1.jpg',resize_cv_b1)

print(f"自写最近邻插值法运行时间 (image1): {time_nn1:.4f} seconds")
print(f"OpenCV 最近邻插值法运行时间 (image1): {time_cv_nn1:.4f} seconds")
print(f"自写双线性插值法运行时间 (image1): {time_b1:.4f} seconds")
print(f"OpenCV 双线性插值法运行时间 (image1): {time_cv_b1:.4f} seconds")

cv.imshow('im1',im1)
cv.imshow('im1_origin',im1_origin)
cv.imshow('resize_nn1',resize_nn1)
cv.imshow('resize_cv_nn1',resize_cv_nn1)
cv.imshow('resize_b1',resize_b1)
cv.imshow('resize_cv_b1',resize_cv_b1)
cv.waitKey(0)
cv.destroyAllWindows()

"""test2"""
start_time4 = time.time()
resize_nn2=nearest_neighbor(im2,scale)
time_nn2 = time.time()-start_time4
cv.imwrite('resize_nn2.jpg',resize_nn2)

start_time5 = time.time()
resize_b2=bilinear(im2,scale)
time_b2= time.time()-start_time5
cv.imwrite('resize_b2.jpg',resize_b2)

start_time6 = time.time()
resize_cv_nn2=cv.resize(im2,(2*width2_origin,2*height2_origin),interpolation=cv.INTER_NEAREST)
time_cv_nn2 = time.time()-start_time6
cv.imwrite('resize_cv_nn2.jpg',resize_cv_nn2)

start_time7 = time.time()
resize_cv_b2=cv.resize(im2,(2*width2_origin,2*height2_origin),interpolation=cv.INTER_LINEAR)
time_cv_b2 = time.time()-start_time7
cv.imwrite('resize_cv_b2.jpg',resize_cv_b2)

print(f"自写最近邻插值法运行时间 (image2): {time_nn2:.4f} seconds")
print(f"OpenCV 最近邻插值法运行时间 (image2): {time_cv_nn2:.4f} seconds")
print(f"自写双线性插值法运行时间 (image2): {time_b2:.4f} seconds")
print(f"OpenCV 双线性插值法运行时间 (image2): {time_cv_b2:.4f} seconds")

cv.imshow('im2',im2)
cv.imshow('im2_origin',im2_origin)
cv.imshow('resize_nn2',resize_nn2)
cv.imshow('resize_cv_nn2',resize_cv_nn2)
cv.imshow('resize_b2',resize_b2)
cv.imshow('resize_cv_b2',resize_cv_b2)
cv.waitKey(0)
cv.destroyAllWindows()

