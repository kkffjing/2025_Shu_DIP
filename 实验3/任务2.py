import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def self_equalize_hist(img_gray):
    img_gray=np.clip(img_gray,0,255).astype(np.uint8)
    #hist,bins = np.histogram(img_gray.flatten(),256,[0,256])#一直报错，怀疑是numpy的兼容性问题
    pixels = img_gray.flatten()#用flatten(）——将图像展为一维数组
    hist=np.zeros(256,dtype=np.int64)
    for x in pixels:
        hist[x]=hist[x]+1
    #不依赖函数计算hist值
    cdf = hist.cumsum()#计算累计分布函数
    cdf_min=cdf.min()
    cdf_max=cdf.max()
    cdf_normalized=((cdf-cdf_min)*255.0/(cdf_max-cdf_min)).astype(np.uint8)#对累计分布函数进行归一化，确保灰度为整形
    return cdf_normalized[img_gray]

def hist_plot(img,title):
    img_gray=np.clip(img,0,255).astype(np.uint8)
    #plt.hist(img.flatten(),256,[0,256])一直报错，怀疑是兼容性问题
    pixels = img_gray.flatten()
    hist=np.bincount(pixels,None,256).astype(np.int64)
    plt.bar(np.arange(len(hist)),hist)
    plt.title(title)
    plt.savefig(title+".png")
    plt.show()#绘制直方图

dark=cv.imread("dark.jpg",cv.IMREAD_GRAYSCALE)
hill=cv.imread("hill.jpg",cv.IMREAD_GRAYSCALE)
baby=cv.imread("baby.png",cv.IMREAD_GRAYSCALE)

dark_self=self_equalize_hist(dark)
hill_self=self_equalize_hist(hill)
baby_self=self_equalize_hist(baby)

dark_cv=cv.equalizeHist(dark)
hill_cv=cv.equalizeHist(hill)
baby_cv=cv.equalizeHist(baby)

res_dark_2=np.hstack((dark,dark_self,dark_cv))
cv.imwrite("res_dark.jpg",res_dark_2)
cv.imshow("dark",res_dark_2)
res_hill=np.hstack((hill,hill_self,hill_cv))
cv.imwrite("res_hill.jpg",res_hill)
cv.imshow("hill",res_hill)
res_baby=np.hstack((baby,baby_self,baby_cv))
cv.imwrite("res_baby.jpg",res_baby)
cv.imshow("baby",res_baby)
cv.waitKey(0)
cv.destroyAllWindows()

hist_plot(dark,"dark hist")
hist_plot(dark_self,"dark_self hist")
hist_plot(dark_cv,"dark_cv hist")
hist_plot(hill,"hill hist")
hist_plot(hill_self,"hill_self hist")
hist_plot(hill_cv,"hill_cv hist")
hist_plot(baby,"baby hist")
hist_plot(baby_self,"baby_self hist")
hist_plot(baby_cv,"baby_cv hist")

