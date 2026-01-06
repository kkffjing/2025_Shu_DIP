import cv2
import numpy as np
import matplotlib.pyplot as plt
def hist_plot(img,title):
    img_gray=np.clip(img,0,255).astype(np.uint8)
    #plt.hist(img.flatten(),256,[0,256])一直报错，怀疑是兼容性问题
    pixels = img_gray.flatten()
    hist=np.bincount(pixels,None,256).astype(np.int64)
    plt.bar(np.arange(len(hist)),hist)
    plt.title(title)
    plt.savefig(title+".png")
    plt.show()#绘制直方图
if __name__=='__main__':
    mus=cv2.imread('mushroom.png')
    mus_bgr=cv2.imread('mus_bgr.png')
    mus_hsv=cv2.imread('mus_hsv.png')
    hist_plot(mus,'mushroom_h')
    hist_plot(mus_bgr,'mus_bgr_h')
    hist_plot(mus_hsv,'mus_hsv_h')

    sky=cv2.imread('sky.bmp')
    sky_bgr=cv2.imread('sky_bgr.bmp')
    sky_hsv=cv2.imread('sky_hsv.bmp')
    hist_plot(sky,'sky_h')
    hist_plot(sky_bgr,'sky_bgr_h')
    hist_plot(sky_hsv,'sky_hsv_h')

