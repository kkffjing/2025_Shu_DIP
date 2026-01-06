import cv2 as cv
import numpy as np
import time

def gamma_with_LUT(img,gamma):
    lut=np.zeros(256,np.uint8)#创建查找表
    for index in range(256):
        lut[index]=np.clip(((index/255.0)**gamma)*255.0,0,255).astype(np.uint8)
        #gamma变换  #边界检测保护
    im_gamma=cv.LUT(img,lut)#用opencv自带函数进行映射
    return im_gamma

def gamma_without_LUT(img,gamma):
    im_gamma=np.array(np.clip(((img/255.0)**gamma)*255.0,0,255),np.uint8)
    #直接使用gamma变换
    return im_gamma

im_light=cv.imread("light.tif",1)
im_dark=cv.imread("dark.jpg",1)
gamma_dark=5#gamma>1 使亮图变暗
gamma_light=0.5#gamma<1 使暗图变亮

start_time=time.time()
im_light_with_lut=gamma_with_LUT(im_light,gamma_dark)
time_light_with_lut=time.time()-start_time

start_time=time.time()
im_light_without_lut=gamma_without_LUT(im_light,gamma_dark)
time_light_without_lut=time.time()-start_time

start_time=time.time()
im_dark_with_lut=gamma_with_LUT(im_dark,gamma_light)
time_dark_with_lut=time.time()-start_time

start_time=time.time()
im_dark_without_lut=gamma_without_LUT(im_dark,gamma_light)
time_dark_without_lut=time.time()-start_time

print("im_light_with_LUT 's time :",time_light_with_lut)
print("im_light_without_LUT 's time :",time_light_without_lut)
res_light=np.hstack((im_light,im_light_with_lut,im_light_without_lut))
cv.imwrite("res_light.jpg",res_light)
cv.imshow("res_light",res_light)
cv.waitKey(0)
cv.destroyAllWindows()

print("im_dart_with_LUT 's time :",time_dark_with_lut)
print("im_dark_without_LUT 's time :",time_dark_without_lut)
res_dark_1=np.hstack((im_dark,im_dark_with_lut,im_dark_without_lut))
cv.imwrite("res_dark1.jpg",res_dark_1)
cv.imshow("res_dark",res_dark_1)
cv.waitKey(0)
cv.destroyAllWindows()




