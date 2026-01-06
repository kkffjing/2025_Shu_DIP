import cv2
import numpy as np

def basic_global_thresholding(image, tolerance=0.5):
    #tolerance: 迭代停止的容差（两次阈值差小于该值则停止）。
    T = np.mean(image)  # 计算图像像素均值作为初始阈值

    while True:
        # 按当前阈值分割图像为前景(G1)和背景(G2)
        G1 = image[image > T]
        G2 = image[image <= T]

        # 计算前景和背景的平均灰度值（处理空数组避免均值计算报错）
        m1 = np.mean(G1) if len(G1) > 0 else 0
        m2 = np.mean(G2) if len(G2) > 0 else 0

        # 计算新阈值
        T_new = (m1 + m2) / 2

        # 判断是否满足停止条件
        if abs(T_new - T) < tolerance:
            T = T_new  # 最终阈值
            break

        # 更新阈值
        T = T_new

    # 应用最终阈值进行二值化（cv2.threshold返回retval和二值图像）
    _, img_thresholded = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)

    return T, img_thresholded #返回最终迭代得到的全局阈值和基于该阈值的二值化图像。

def otsu_global_thresholding(image):

    #  转换为uint8并严格裁剪到0-255
    img = np.clip(image.astype(np.float32), 0, 255).astype(np.uint8)

    # OpenCV计算直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # 展平为一维数组
    hist = hist.flatten()

    # 1. 归一化直方图，得到概率密度函数p(i)
    size = img.size
    p = hist / size  # 归一化直方图

    # 2. 计算累积概率P(k)
    P = np.cumsum(p)

    # 3. 计算累积均值m(k)
    intensity_levels = np.arange(256)
    m = np.cumsum(intensity_levels * p)

    # 4. 计算全局灰度均值m_G：所有灰度级的加权平均
    m_G = m[-1]  # 累积均值的最后一个值即为全局均值

    # 5. 计算类间方差sigma_B_sq(k)
    numerator = (m_G * P - m) ** 2
    denominator = P * (1 - P)
    sigma_B_sq = np.zeros_like(denominator)  # 初始化类间方差数组
    valid_mask = denominator > 0  # 筛选分母有效（>0）的位置
    sigma_B_sq[valid_mask] = numerator[valid_mask] / denominator[valid_mask]#计算有效位置

    # 6. 找到最优阈值k*：类间方差最大对应的灰度级
    max_sigma_B_sq = np.max(sigma_B_sq)  # 最大类间方差
    k_i = np.where(sigma_B_sq == max_sigma_B_sq)[0]  # 所有最大方差对应的k
    k = np.mean(k_i)  # 取平均值作为最优阈值（处理多峰值情况）

    # 7. 计算可分离性测度：类间方差/全局方差
    sigma_G_sq = np.sum(((intensity_levels - m_G) ** 2) * p) # 全局方差sigma_G_sq
    separability = max_sigma_B_sq / sigma_G_sq if sigma_G_sq > 0 else 0 # 避免除以零，全局方差为0时可分离性设为0

    # 应用最优阈值进行二值化（阈值取整，因为cv2.threshold需要整数阈值）
    k_final = int(k)
    _, img_thresholded = cv2.threshold(img, k_final, 255, cv2.THRESH_BINARY)

    return k, img_thresholded, separability
    # 返回Otsu最优阈值,使用最优阈值分割后的二值图像和可分离性测度（eta），类间方差/全局方差。

def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 两种方法分割
    T, bas_seg = basic_global_thresholding(img)
    k, otsu_seg, separability = otsu_global_thresholding(img)
    print(f"\n=== 图像 {img_path} 分割结果 ===")
    print(f"迭代式全局阈值：{T:.2f}")
    print(f"Otsu最优阈值：{k:.2f}")
    print(f"Otsu可分离性测度：{separability:.4f}")
    res_img = np.hstack((img, bas_seg, otsu_seg))
    cv2.imwrite(f"res_{img_path}", res_img)
    cv2.imshow(f"res_{img_path}", res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_image("rice.tif")
    process_image("finger.tif")
    process_image("poly.tif")
