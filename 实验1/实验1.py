import numpy as np
import cv2 as cv
id = "24122365"
name = "JINGKAIFENG"
"""读取图像"""
im = cv.imread('letter.jpg',1)
"""转灰度"""
imgray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
"""用阈值将图像转为二进制图像"""
thresh = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 15, 4)  # 反转阈值+优化参数
"""提取二进制图像轮廓"""
contours,hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
"""绘制轮廓"""
contour_image = np.ones_like(im) * 255
cv.drawContours(contour_image,contours,-1,(0,255,0),2)
cv.imshow('cim',contour_image)
cv.waitKey(0)
cv.destroyAllWindows()
"""筛选出符合要求的轮廓"""
# 过滤
min_area = 10000    # 最小面积（过滤微小噪声）
max_area = 50000  # 最大面积（过滤整个图像/大块背景）
min_aspect_ratio = 0.5
max_aspect_ratio = 1.5 # 字符宽高比范围（避免过窄/过宽的噪声)
char_box=[]  # 存储轮廓
for i, contour in enumerate(contours):
    x, y, w, h = cv.boundingRect(contour)
    area=cv.contourArea(contour)  # 计算轮廓面积（关键过滤依据）
    if h!=0:
        aspect_ratio = w/h
    else:
        aspect_ratio = 0# 宽高比
    # 多条件过滤：尺寸+面积+宽高比（确保只保留字符级轮廓）
    if (w > 20 and h > 20 and min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio):
        char_box.append([x, y, w, h])
# 验证轮廓
verify_im = im.copy()
for i,(x,y,w,h) in enumerate(char_box):
    cv.rectangle(verify_im, (x,y), (x+w,y+h), (0,255,0), 3)
    cv.putText(verify_im, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)
cv.imshow('字符轮廓验证', verify_im)
cv.waitKey(0)
cv.destroyAllWindows()
"""根据所绘制的轮廓，截取数字和字母图像"""
char_box_count = len(char_box)
print("请按轮廓序号的顺序，输入每个序号对应的字符.要求输入大写字母或数字，不要有空格")

# 读取并验证用户输入
letter_list = input(f"\n请输入 {len(char_box)} 个字符：").strip()
#建立映射：字符 → 轮廓ROI
char_to_roi = {}
for idx, (contour, letter) in enumerate(zip(char_box, letter_list)):
    x,y,w,h =contour
    char_to_roi[letter] = im[y:y+h,x:x+w] # 存储字符对应的图像区域
"""拼接图像图像"""
char_space = 5
line_space = 5 # 设置字符间隔和行间距
max_height = max(le.shape[0] for le in char_to_roi.values())
def count_width(string):
    width = 0
    for char in string:
        if char in char_to_roi:
            width += char_to_roi[char].shape[0]
        else:
            print("未找到字符")
    return width
total_width = max(count_width(id), count_width(name))
total_height =max_height*2+line_space*2# 计算new_im高度和宽度
# 选择原图中的一个区域作为背景颜色的参考
reference_region = im[0:10, 0:10]
background_color = np.mean(reference_region, axis=(0, 1)).astype(int)
background_color = np.clip(background_color, 0, 255).astype(np.uint8)
# 创建一个空白的背景图像（使用提取的背景颜色）
new_im = np.ones((total_height, total_width, 3), dtype=np.uint8)*background_color
current_x = char_space # 从字符间隔开始
current_y = char_space # 从字符间隔开始
#拼接学号
for char in id:
    if char in char_to_roi:
        char_image = char_to_roi[char]  # 获取字符图像
        char_height = char_image.shape[0]
        char_width = char_image.shape[1] # 将字符图像粘贴到新图像的指定位置
        if char_width > (total_width - current_x):
            char_image = cv.resize(char_image, (total_width - current_x, char_height))
        new_im[current_y:current_y + char_height, current_x:current_x + char_width] = char_image
        current_x += char_width + char_space  # 更新x坐标
    else:
        print(char, '字符未找到')

current_x = char_space  # 重置x坐标
current_y += max_height + line_space # 更新y坐标加上行间距

for char in name:
    if char in char_to_roi:
        char_image = char_to_roi[char]  # 获取字符图像
        char_height = char_image.shape[0]
        char_width = char_image.shape[1] # 将字符图像粘贴到新图像的指定位置
        if char_width > (total_width - current_x):
            char_image = cv.resize(char_image, (total_width - current_x, char_height))

        new_im[current_y:current_y + char_height, current_x:current_x + char_width] = char_image
        current_x += char_width + char_space # 更新x坐标
    else:
        print(char, '字符未找到')
"""显示图像并保存"""
cv.imshow('New Image', new_im)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('New Image.jpg', new_im)