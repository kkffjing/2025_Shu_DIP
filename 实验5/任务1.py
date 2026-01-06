
import cv2
import numpy as np
import heapq
from collections import Counter
from math import sqrt

class Node:#定义结点
    def __init__(self, val, freq):
        self.val = val
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other): # 重载运算符进行比较，用于最小堆排序
        return self.freq < other.freq

def build_huffman_tree(img): # 构建哈夫曼树，返回根结点
    freq = Counter(img.flatten()) # 统计像素频率
    heap = [Node(val, f) for val, f in freq.items()] # 创建结点列表
    heapq.heapify(heap) # 转换成最小堆
    while len(heap) > 1: # 循环直到只剩一个棵树
        l, r = heapq.heappop(heap), heapq.heappop(heap) # 取出权重最小的两棵树
        merged = Node(None, l.freq + r.freq)
        merged.left, merged.right = l, r# 合并为一个新二叉树
        heapq.heappush(heap, merged) # 放到堆顶
    return heap[0]

def generate_codes(node, cd="", cds=None): # 生成哈夫曼编码
    if cds==None:
        cds={}
    if node.val is not None:
        cds[node.val] = cd if cd else '0'
    else:
        if node.left: generate_codes(node.left, cd + "0", cds) # 左分支添0
        if node.right: generate_codes(node.right, cd + "1", cds) # 右分支添1
    return cds

def huffman_compress(img): # 压缩函数：返回比特流、树根、原图形状
    root = build_huffman_tree(img)
    codes = generate_codes(root)
    bits = "".join(codes[p] for p in img.flatten())
    # 将像素的多维数组转化为一维数组
    # 通过查表得到其哈夫曼编码
    # 利用join进行一次拼接转化为比特流
    return bits, root, img.shape

def huffman_decompress(bits, root, shape): # 解压函数：重建图像
    decd = [] #解码比特流
    r = root
    for b in bits:
        if b == '0':
           r = r.left
        else :
           r = r.right # 根据比特位移动
        if r.val is not None: # 到达叶子节点，找到原值
            decd.append(r.val)
            r = root # 重置回根节点，寻找下一个
    img_org=np.array(decd,dtype=np.uint16).reshape(shape)
    return img_org

def predictive_code(img): # 简单的一维无损预测（左侧预测）
    h, w = img.shape
    # 转为int16防止溢出，因为差值范围是 -255 到 255
    img_int = img.astype(np.int16)
    errors = np.zeros_like(img_int)
    for i in range (h):
        for j in range (w):
            if j==0:
                pred=0
            else:
                pred=img_int[i,j-1]
            errors[i,j]=img_int[i,j]-pred
    # 将误差平移到正数范围 (0-510)，以便进行哈夫曼编码
    errors_shifted = errors + 255
    return errors_shifted.astype(np.uint16)

def predictive_decompress(errors_shifted, shape): # 解码预测误差并解压
    h, w = shape
    errors = errors_shifted.astype(np.int16) - 255 # 还原误差值
    de = np.zeros(shape, dtype=np.int16)
    # 必须串行重建，因为当前像素依赖前一个已重建的像素
    for i in range(h):
        for j in range(w):
            if j > 0:
             pred = de[i, j-1]
            else :
             pred = 0
            de[i, j] = pred + errors[i, j]
    return np.clip(de, 0, 255).astype(np.uint8)

def calc_metrics(img, compressed_bits, dec_img):
    mse = np.mean((img - dec_img) ** 2) # 均方误差
    rmse= sqrt(mse) # 均方根误差
    orig_bits = img.size * 8 # 原图总比特数
    cal_rat = orig_bits / len(compressed_bits) # 压缩比
    return rmse, cal_rat

def process_image(path):
    img = cv2.imread(path, 0) # 以灰度模式读取

    # 1. 直接哈夫曼编码
    h_bits, h_root, shape = huffman_compress(img)
    h_dec = huffman_decompress(h_bits, h_root, shape)
    h_rmse, h_cal_rat = calc_metrics(img, h_bits, h_dec)
    print(f" 1.采用哈夫曼编码，实现 RMSE: {h_rmse:.4f}, 压缩比: {h_cal_rat:.2f}")

    # 2. 预测编码 + 哈夫曼
    err_img = predictive_code(img)
    # 将预测误差作为符号进行哈夫曼编码
    d_bits, d_root, d_shape = huffman_compress(err_img)
    d_dec_err = huffman_decompress(d_bits, d_root, d_shape) # 还原误差图
    d_rec = predictive_decompress(d_dec_err, shape) # 还原原图
    d_rmse, d_cal_rat = calc_metrics(img, d_bits, d_rec)
    print(f"  2.采用无损预测编码（预测模型自己设定），并对误差进行哈夫曼编码，实现 RMSE: {d_rmse:.4f}, 压缩比: {d_cal_rat:.2f}")

    # 验证逻辑正确性
    if d_rmse == 0: print("  > 无损压缩验证通过 (RMSE=0)")


if __name__ == '__main__':
    process_image('bridge.bmp')
    process_image('web.bmp')

