import numpy as np
import cv2 as cv
import math
def rotate(img,angle):
    h,w = img.shape[:2]
    channels=img.shape[2]
    rad=math.radians(angle)#转化为弧度
    cos0=math.cos(rad)
    sin0=math.sin(rad)
    x_center=w/2
    y_center=h/2
    #先将四个角的点进行旋转，确定旋转后图片大小
    corners_org = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
    corners_rotated = []
    for (x, y) in corners_org:
        tx=x-x_center
        ty=y-y_center
        x_rot=tx*cos0-ty*sin0
        y_rot=tx*sin0+ty*cos0#绕原点进行旋转
        corners_rotated.append([x_rot, y_rot])
    corners_rotated = np.array(corners_rotated, dtype=np.float64)
    x_min,y_min=np.min(corners_rotated, axis=0)
    x_max,y_max=np.max(corners_rotated, axis=0)
    w_new=int(np.ceil(x_max - x_min))
    h_new=int(np.ceil(y_max - y_min))#确定旋转后图片的长宽
    x_center_new=w_new/2
    y_center_new=h_new/2#以中点为旋转后的中心点，确保可以显示所有像素点
    #创建新图像
    im_rotated = np.zeros((h_new, w_new, channels), dtype=np.uint8)
    for y in range(h_new):
        for x in range(w_new):
            tx=x-x_center_new
            ty=y-y_center_new
            x_rot=tx*cos0+ty*sin0
            y_rot=-tx*sin0+ty*cos0
            x_old=x_rot+x_center
            y_old=y_rot+y_center#反向映射到原图像的像素点
            if 0<=x_old<=w-1 and 0<=y_old<=h-1:#边界检测，确保在有效范围内
                #用双线性插值法完成灰度内插
                x1 = int(np.floor(x_old))
                y1 = int(np.floor(y_old))
                x2 = x1 + 1
                y2 = y1 + 1

                x1 = min(x1, w - 1)
                y1 = min(y1, h - 1)
                x2 = min(x2, w - 1)
                y2 = min(y2, h - 1)

                dx = x_old - x1
                dy = y_old - y1

                c11 = img[y1, x1]
                c12 = img[y1, x2]
                c21 = img[y2, x1]
                c22 = img[y2, x2]

                for c in range(channels):
                    val= (1 - dx) * (1 - dy) * c11[c] + (1 - dx) * dy * c12[c] + dx * (1 - dy) * c21[c] + dx * dy * c22[c]
                    im_rotated[y,x,c]=np.clip(val,0,255)
    return im_rotated

im1=cv.imread('test1.jpg',1)
im1_rotated1=rotate(im1,45)
cv.imwrite('im1_rotate_45.jpg',im1_rotated1)
cv.imshow('im1_rotated_45',im1_rotated1)
im1_rotated2=rotate(im1,90)
cv.imwrite('im1_rotated_90.jpg',im1_rotated2)
cv.imshow('im1_rotated_90',im1_rotated2)
im1_rotated3=rotate(im1,135)
cv.imwrite('im1_rotated_135.jpg',im1_rotated3)
cv.imshow('im1_rotated_135',im1_rotated3)
im1_rotated4=rotate(im1,180)
cv.imwrite('im1_rotated_180.jpg',im1_rotated4)
cv.imshow('im1_rotated_180',im1_rotated4)
cv.imshow('im1',im1)
cv.waitKey(0)
cv.destroyAllWindows()

im2=cv.imread('test2.jpg',1)
im2_rotated1=rotate(im2,22)
cv.imwrite('im2_rotate_22.jpg',im2_rotated1)
cv.imshow('im2_rotated_22',im2_rotated1)
im2_rotated2=rotate(im2,59)
cv.imwrite('im2_rotated_59.jpg',im2_rotated2)
cv.imshow('im2_rotated_59',im2_rotated2)
im2_rotated3=rotate(im2,111)
cv.imwrite('im2_rotated_111.jpg',im2_rotated3)
cv.imshow('im2_rotated_111',im2_rotated3)
im2_rotated4=rotate(im2,166)
cv.imwrite('im2_rotated_166.jpg',im2_rotated4)
cv.imshow('im2_rotated_166',im2_rotated4)
cv.imshow('im2',im2)
cv.waitKey(0)
cv.destroyAllWindows()
