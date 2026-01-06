import cv2
import numpy as np
def get_filter(img,kernel,i,j):
    #获取点周围与算子大小对应的图像区域
    half_kernel=kernel//2 #将卷积整除
    start_i=np.clip(i-half_kernel,0,img.shape[0]-1)
    start_j=np.clip(j-half_kernel,0,img.shape[1]-1)
    end_i=np.clip(i+kernel-half_kernel,0,img.shape[0])
    end_j=np.clip(j+kernel-half_kernel,0,img.shape[1])#边界检测
    return img[start_i:end_i,start_j:end_j]
def mean_filtering(img,kernel):
    im_filtered = np.zeros(img.shape,np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                im_filtered[i,j,k] = np.mean(get_filter(img[:,:,k],kernel,i,j))
    return im_filtered #计算点周围灰度的平均数
def median_filtering(img,kernel):
    im_filtered = np.zeros(img.shape,np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                im_filtered[i,j,k] = np.median(get_filter(img[:,:,k],kernel,i,j))
    return im_filtered #计算点周围灰度的中值

space=cv2.imread("Space.jpg")
mona=cv2.imread("Mona.jpg")
pcb=cv2.imread("pcb.png")

mean_space=mean_filtering(space,3)
median_space=median_filtering(space,3)
mean_mona=mean_filtering(mona,3)
median_mona=median_filtering(mona,3)
mean_pcb=mean_filtering(pcb,3)
median_pcb=median_filtering(pcb,3)

res_space=np.hstack((space,mean_space,median_space))
res_mona=np.hstack((mona,mean_mona,median_mona))
res_pcb=np.hstack((pcb,mean_pcb,median_pcb))

cv2.imwrite("res_space.jpg",res_space)
cv2.imwrite("res_mona.jpg",res_mona)
cv2.imwrite("res_pcb.jpg",res_pcb)
cv2.imshow("res_space",res_space)
cv2.imshow("res_mona",res_mona)
cv2.imshow("res_pcb",res_pcb)
cv2.waitKey(0)
cv2.destroyAllWindows()



