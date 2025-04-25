import os.path

import numpy as np
import cv2
import math

from matplotlib import pyplot as plt
from skimage import exposure


# reference_image_paths用于存放调整a通道的参考图像，reference_image_paths_L用于存放调整V通道的参考图像，file_name2用于存放待调整图像
def ret_img(reference_image_paths, reference_image_paths_L, file_name2):
    name = os.path.basename(file_name2)
    print(file_name2)
    fold_name_save = 'D:/File/crop3/6_res/'  # 存放结果的文件夹
    name_new = fold_name_save + name
    reference_a = []
    ref_light = []
    img_resized = None
    img_resized_L = None
    print('origin_size:', img_resized)
    print('origin_size_L:', img_resized_L)
    # 读取用于调整V通道的参考图像，并保存参考图像的光照分量
    for i in range(len(reference_image_paths_L)):
        reference_image_path_L = os.path.join(c_folder_L, reference_image_paths_L[i])
        print('ref_L_path:', reference_image_path_L)
        image_ref_L = cv2.imread(reference_image_path_L)  # 参考图像
        if img_resized_L is None:
            img_resized_L = image_ref_L.shape[:2]
        print('ref_L_size:', img_resized_L)
        image_ref_L = cv2.resize(image_ref_L, (img_resized_L[1], img_resized_L[0]))
        im_hsv = cv2.cvtColor(image_ref_L, cv2.COLOR_BGR2HSV)
        im_H, im_s, im_v = cv2.split(im_hsv)
        b_im = cv2.GaussianBlur(im_v, (0, 0), 20)  # 高斯滤波处理，获取光照分量
        print('b_im_size:', b_im.shape)
        ref_light.append(b_im)
    # 读取用于调整a通道的参考图像，并保存参考图像的a通道数据
    for i in range(len(reference_image_paths)):
        reference_image_path = os.path.join(c_folder, reference_image_paths[i])
        print('ref_a_path:', reference_image_path)
        image_ref = cv2.imread(reference_image_path)  # 参考图像
        if img_resized is None:
            img_resized = image_ref.shape[:2]
        print('ref_size:', img_resized)
        image_ref = cv2.resize(image_ref, (img_resized[1], img_resized[0]))
        im_lab = cv2.cvtColor(image_ref, cv2.COLOR_BGR2Lab)
        im_L, im_a, im_b = cv2.split(im_lab)
        reference_a.append(im_a)
    # 求平均光照
    ref_light_stack = np.array(ref_light)
    ref_light_mean = np.mean(ref_light_stack, axis=0)
    # 处理参考图a通道
    matched_a = reference_a[0]
    for m in range(3):
        for i in range(len(reference_a)):
            matched_a = exposure.match_histograms(matched_a, reference_a[i])
    # 对目标图像进行处理
    # 转变为HSV颜色空间
    image2 = cv2.imread(file_name2)  # 目标图像
    im2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    im2_H, im2_s, im2_v = cv2.split(im2_hsv)
    # 对每张图V通道做高斯滤波处理,得到光照图像
    b_im2 = cv2.GaussianBlur(im2_v, (0, 0), 20)
    # 将目标图的光照图像与平均光照图像匹配
    b_match2 = exposure.match_histograms(b_im2, ref_light_mean)
    # 利用log运算，从原图中减去匹配结果，使用匹配后的光照图像进行retinex增强
    im2_f = np.float32(im2_v)
    r_im2 = np.log10(im2_f + 0.01) - np.log10(b_match2 + 0.01)
    r_im2 = (r_im2 - np.min(r_im2)) / (np.max(r_im2) - np.min(r_im2)) * 255 * 0.9
    r_im2_v = np.uint8(np.minimum(np.maximum(r_im2, 0), 255))
    # 用新V通道数值替换原来
    im2_hsv[:, :, 2] = r_im2_v
    print('H:', np.sum(im2_hsv[:, :, 0] != im2_H))
    print('S:', np.sum(im2_hsv[:, :, 1] != im2_s))
    print('V:', np.sum(im2_hsv[:, :, 2] == im2_v))
    # 将hsv转变到Lab
    im2_new = cv2.cvtColor(im2_hsv, cv2.COLOR_HSV2BGR)
    im2_lab = cv2.cvtColor(im2_new, cv2.COLOR_BGR2Lab)
    im2_L, im2_a, im2_b = cv2.split(im2_lab)
    # 修改a通道
    a_matched = exposure.match_histograms(im2_a, matched_a)
    im2_lab[:, :, 1] = a_matched
    print('L:', np.sum(im2_lab[:, :, 0] != im2_L))
    print('a:', np.sum(im2_lab[:, :, 1] == im2_a))
    print('b:', np.sum(im2_lab[:, :, 2] != im2_b))
    print(name_new)
    # # 保存图片
    im2_save = cv2.cvtColor(im2_lab, cv2.COLOR_Lab2BGR)
    cv2.imwrite(name_new, im2_save)


# Example usage
if __name__ == "__main__":
    import cv2
    import numpy as np

    input_folder = 'D:/File/crop3/6_crop'  # 待调整图片文件夹路径
    c_folder = 'D:/File/crop3/6_can2'  # 用于调整a通道的参考图像文件夹路径
    c_folder_L = 'D:/File/crop3/background_can'  # 用于调整V通道的参考图像文件夹路径
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_c = [f for f in os.listdir(c_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_c_L = [f for f in os.listdir(c_folder_L) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(input_folder, image_file)
        ret_img(image_c, image_c_L, image_path)
