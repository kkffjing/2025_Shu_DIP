from turtledemo.penrose import start

import cv2
import time
import numpy as np
#用cv2自带函数拆分
def split_cv(img):
    b, g, r = cv2.split(img)#拆分bgr通道
    cv2.imwrite('b_gray.jpg', b)
    cv2.imwrite('g_gray.jpg', g)
    cv2.imwrite('r_gray.jpg', r)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)


def split_self(img):
    channels=['blue','green','red']#为通道命名
    for i,channel in enumerate (channels):
        im_in_channel=img[:,:,i]  # 获取每个通道的图像
        cv2.imwrite(f'{channel}.jpg',im_in_channel)# 使用通道名字作为文件名
        cv2.imshow(channel,im_in_channel)


if __name__ == '__main__':
    img = cv2.imread('araras.jpg',1)
    start_time_cv = time.time()
    split_cv(img)
    end_time_cv = time.time()
    time_cv = end_time_cv - start_time_cv
    start_time_self = time.time()
    split_self(img)
    end_time_self = time.time()
    time_self = end_time_cv - start_time_cv
    print(f'cv2实现所需时间：{time_cv}')
    print(f'自行实现所需时间：{time_self}')
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
