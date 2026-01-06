import cv2
def equalize_bgr(img):
    # OpenCV 读取的是 BGR 格式
    b, g, r = cv2.split(img)#拆分BGR通道
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)#对每个通道做直方图均衡化
    return cv2.merge([b_eq, g_eq, r_eq])#合并通道

def equalize_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转换到HSV空间
    h, s, v = cv2.split(hsv)#拆分HSV通道
    v_eq = cv2.equalizeHist(v)#仅对亮度通道进行直方图均衡化
    hsv_eq = cv2.merge([h, s, v_eq])#合并
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)#转化到BGR空间

if __name__ == '__main__':
    sky= cv2.imread('sky.bmp',1)
    sky_bgr = equalize_bgr(sky)
    sky_hsv = equalize_hsv(sky)
    cv2.imwrite('sky_bgr.bmp',sky_bgr)
    cv2.imwrite('sky_hsv.bmp',sky_hsv)
    cv2.imshow('sky', sky)
    cv2.imshow('sky_bgr', sky_bgr)
    cv2.imshow('sky_hsv', sky_hsv)

    mus=cv2.imread('mushroom.png',1)
    mus_bgr = equalize_bgr(mus)
    mus_hsv = equalize_hsv(mus)
    cv2.imwrite('mus_bgr.png',mus_bgr)
    cv2.imwrite('mus_hsv.png',mus_hsv)
    cv2.imshow('mushroom', mus)
    cv2.imshow('mus_bgr', mus_bgr)
    cv2.imshow('mus_hsv', mus_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


