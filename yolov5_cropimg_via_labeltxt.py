import os
import cv2

# 检测结果和原图的文件夹路径
labels_dir = 'D:/File/new_yaocai/yolo_res/8/labels'  # 通过YOLOV5获取的Label文件夹
images_dir = 'D:/File/new_yaocai/yaocai_split/fuling'  # 存放含有对应类别药材的图像块文件夹
output_dir = 'D:/File/new_yaocai/yolo_res/8_res'  # 存放裁剪结果的文件夹

# 创建保存切割图像的文件夹
os.makedirs(output_dir, exist_ok=True)

# 遍历检测结果文件
for label_file in os.listdir(labels_dir):
    image_file = label_file.replace('.txt', '.jpg')  # 假设图像为 .jpg 格式
    image_path = os.path.join(images_dir, image_file)
    label_path = os.path.join(labels_dir, label_file)

    # 加载图像
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    k = 1.2  # 膨胀系数
    # 读取标签文件中的检测信息
    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            # 太小的块不要
            if width*w < 10 or height*h < 10:
                continue
            # 计算检测框的绝对坐标
            abs_x_center = int(x_center * w)
            abs_y_center = int(y_center * h)
            abs_width = int(width * w * k)  # 进行膨胀
            abs_height = int(height * h * k)
            # 计算检测框的左上角和右下角坐标
            x1 = max(0, abs_x_center - abs_width // 2)
            y1 = max(0, abs_y_center - abs_height // 2)
            x2 = min(w, abs_x_center + abs_width // 2)
            y2 = min(h, abs_y_center + abs_height // 2)
            # 从图像中切割检测区域
            cropped_image = image[y1:y2, x1:x2]
            # 保存切割后的图像
            output_path = os.path.join(output_dir, f"{image_file.split('.')[0]+'_'+str(i)}.jpg")
            cv2.imwrite(output_path, cropped_image)
            print(f"Saved cropped image: {output_path}")
