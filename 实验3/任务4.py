import cv2
import numpy as np

# 设置拉普拉斯算子
# 中心为-4，四周为1
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
def get_filter(img, laplacian, i, j):
    ##获取点周围与算子大小对应的图像区域
    kernel_size = laplacian.shape[0]
    half_kernel = kernel_size // 2
    start_i = i - half_kernel
    start_j = j - half_kernel
    end_i = i + half_kernel + 1
    end_j = j + half_kernel + 1
    return img[start_i:end_i, start_j:end_j]
def laplacian_filtering(img, laplacian):
    ##使用拉普拉斯算子对图像进行锐化
    ##公式: g(x,y) = f(x,y) - ∇²f(x,y) (当算子中心为负时)
    # 转换为浮点型进行计算，防止溢出
    im_sharpened = np.zeros_like(img.astype(np.float32))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
                for k in range(img.shape[2]):
                    roi = get_filter(img[:, :, k], laplacian, i, j)
                    lap_val = np.sum(roi * laplacian)
                    im_sharpened[i, j, k] = img[i, j, k] - lap_val
    im_sharpened = np.clip(im_sharpened, 0, 255).astype(np.uint8)
    return im_sharpened

bm = cv2.imread("blurry_moon.tif")
bm_sharpened = laplacian_filtering(bm, laplacian)
res_bm=np.hstack((bm,bm_sharpened))
cv2.imwrite("res_blurry_moon_sharpened.tif", res_bm)
cv2.imshow("res_blurry_moon_sharpened", res_bm)
cv2.waitKey(0)
cv2.destroyAllWindows()
