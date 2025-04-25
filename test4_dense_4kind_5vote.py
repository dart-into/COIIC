import argparse
import csv
import glob
import os
import random
import torch
import re
import pandas as pd
from torch import optim, nn
import torchvision.models as models
from torch.utils.data import DataLoader
from MyDataset import HerbalMedicineDataset
import numpy as np
from torchvision import transforms
from sklearn.metrics import recall_score

best_acc = 0
best_epoch = 0
batch_size = 1
epochs = 30
pepochs = 20
nepochs = 20
extend_epochs = 10
num_classes = 20
pretrained = True
learning_rate = 0.00002
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test2(model, loader, csv_file_path, r):
    correct = 0
    total = 0
    model.eval()
    predictions_dict = {}  # 初始化字典
    pred_result = []
    # 标签到索引的映射
    label_to_index = {6: 0, 8: 1, 15: 2, 3: 3}  # 根据你的标签定义映射
    num_classes = len(label_to_index)  # 类别数量
    Out = []
    Small_path = []
    Labels = []
    Labels_True = []
    image_pred_all = {}  # 存放不同参考图调整的概率
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            negative_images, labels, small_image_paths, number = data
            negative_images, labels = negative_images.to(device), labels.to(device)
            outputs = model(negative_images)
            probabilities = torch.softmax(outputs, dim=1)  # 计算预测概率
            # print(outputs)
            value_max, predicted = torch.max(outputs.data, dim=1)
            Small_path.append(small_image_paths)
            Out.append(probabilities.cpu().numpy().flatten().tolist())
            Labels.append(labels.item())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if number.item() not in image_pred_all:
                Labels_True.append(labels.item())
                image_pred_all[number.item()] = []
            image_pred_all[number.item()].append(probabilities.cpu().numpy().flatten().tolist())

    print('image_pred_all_len:', len(image_pred_all))
    print('image_pred_all[1]:', len(image_pred_all[1]))
    # 获取最大值
    image_pred_all[1] = np.array(image_pred_all[1])
    max_value = np.max(image_pred_all[1])
    pred_images = []
    for num in range(len(image_pred_all)):
        pred_array = np.array(image_pred_all[num + 1])
        # 平均
        mean_per_row = np.mean(pred_array, axis=0)
        # 获取最大值所在的位置（索引）
        max_position = np.argmax(mean_per_row)
        pred_max = max_position
        pred_images.append(pred_max)
        # 取最大
        # max_position = np.unravel_index(np.argmax(pred_array), pred_array.shape)
        # pred_max = max_position[1]
    num = 0
    pred_fin = []
    print('pred_images_len:', len(pred_images))
    while num < len(image_pred_all):
        pred_line = [0] * 20
        for i in range(10):
            # print('image_path:', Small_path[num])
            # print('pred:', pred_images[num])
            pred_line[pred_images[num]] += 1
            num = num + 1
        line_array = np.array(pred_line)
        max_position = np.argmax(line_array)
        # print('max_p:', max_position)
        for i in range(10):
            pred_fin.append(max_position)
    print('pred_fin_len:', len(pred_fin))
    new_total = len(pred_fin)
    new_correct = 0
    for nid in range(len(pred_fin)):
        if pred_fin[nid] == Labels_True[nid]:
            new_correct = new_correct + 1
    for num in range(len(pred_images)):
        pred_max = pred_fin[num]
        small_image_paths = Small_path[num]
        for i in range(len(small_image_paths)):  # 遍历当前样本的所有小图路径
            # print(len(small_image_paths))
            print('small_path:', small_image_paths[i][0])
            small_image_path = small_image_paths[i][0].split('/')[-1]
            # pred_example.append(small_image_path)
            # print(type(small_image_path))
            if small_image_path not in predictions_dict:
                predictions_dict[small_image_path] = [0] * num_classes  # 初始化计数列表，长度为类别数
            # # 使用映射更新计数
            index = label_to_index[pred_max]  # 获取实际标签的索引
            # print(index)
            predictions_dict[small_image_path][index] += 1
    index = 0
    print('Outs:', len(Out))
    for path, counts in predictions_dict.items():
        print(f"Path: {path}, Counts: {counts}")
        index = index + 1
    print('index:', index)
    df = pd.DataFrame.from_dict(predictions_dict, orient='index',
                                columns=[f'Count_Label_{label}' for label in label_to_index.keys()])
    df.index.name = 'Small_Image_Path'  # 设置索引名称
    df.reset_index(inplace=True)  # 重置索引以便将其作为列

    # 添加最大计数所在的标签和真实标签列
    df['Max_Count'] = df[[f'Count_Label_{label}' for label in label_to_index.keys()]].max(axis=1)  # 最大计数
    df['Max_Label'] = df[[f'Count_Label_{label}' for label in label_to_index.keys()]].idxmax(axis=1).apply(
        lambda x: int(x.split('_')[-1]))  # 最大计数对应的标签

    # 提取真实标签
    def extract_real_label(file_path):
        file_name = file_path.split('/')[-1]
        first_number = file_name.split('_')[0]
        # print(first_number)
        if first_number == '62747-1-58':
            first_number = 3
        # print(file_name)
        return int(first_number)  # 假设真实标签是第一个数字

    df['Real_Label'] = df['Small_Image_Path'].apply(extract_real_label)  # 从小图路径提取真实标签

    # 添加 Is_Correct 列，判断 Max_Label 和 Real_Label 是否相等
    df['Is_Correct'] = (df['Max_Label'] == df['Real_Label']).astype(int)  # 相等为 1，不相等为 0
    c = (df['Max_Label'] == df['Real_Label']).astype(int)
    accuracy = c.sum() / len(df)
    print(f"投票的准确率: {accuracy:.3f}")
    df['acc_single'] = round(correct / total, 3)
    # 保存到 CSV 文件
    df.to_csv(csv_file_path, index=False)  # 保存 DataFrame 到 CSV 文件
    print('correct:', new_correct, 'total:', new_total)
    novote_recall = recall_score(Labels_True, pred_fin, average='macro')
    new_recall = recall_score(df['Real_Label'], df['Max_Label'], average='macro')
    print('novote_recall:', novote_recall)
    print('new_recall:', new_recall)
    return round(new_correct / new_total, 3), round(accuracy, 3), round(novote_recall, 3), round(new_recall, 3)


# dataset(data_path)
acc_single = []
acc_vote = []
recall1 = []
recall2 = []
for i in range(1):
    print('range' + str(i + 1))

    transform_big = transforms.Compose([
        # transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_small = transforms.Compose([
        transforms.Resize((224, 224)),  # 这里将小图调整为112x112
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # 用于存放结果的csv
    csv_file_path = '4kind_csv/test_quan/Va2to4kind_pred/dense_pred/predictions_counts' + str(
        i + 1) + '_Va2light_allto4kind_4round_densenet169_quan_trainshai_10vote_10round_recall_time.csv'  # 指定 CSV 文件路径
    # csv_file='4kind_csv/test_quan/Va2light_csv/test_4kind_new_Va2to4kind.csv' #
    # 使用Va通道调整，测试时每幅图像使用四种参考图调整，5张大图为一组**** csv_file='4kind_csv/test_quan/Va2light_csv/test_4kind_10round_Va2to4kind
    # .csv' # 使用Va通道调整，测试时每幅图像使用四种参考图调整，10张大图为一组********
    # 修改csv_file，调整为自己生成的测试csv文件
    test_dataset = HerbalMedicineDataset(csv_file='4kind_csv/test_quan/Va2light_csv/test_4kind_10round_Va2to4kind.csv',
                                         transform_big=transform_big, transform_small=transform_small)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=8,
                             pin_memory=True,
                             prefetch_factor=6)

    # 加载预训练的 DenseNet
    model = models.densenet169(pretrained=False)
    # 修改分类头为 20 类输出
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    # 将模型移动到 GPU
    model = model.to(device)
    # print(model)
    print('new4_dense169_test')
    # 训练好的模型参数
    model_path = "model_4kind_4round_Valight_dense/model_4kind_Va2light2_4round_dense169_shai3_shai1_" + str(
        i + 1) + ".mdl"
    print(model_path)
    model.load_state_dict(torch.load(model_path))
    test_acc, acc_all, novote_recall, new_recall = test2(model, test_loader, csv_file_path, i + 1)
    acc_single.append(test_acc)
    acc_vote.append(acc_all)
    recall1.append(novote_recall)
    recall2.append(new_recall)
    print("Accuracy Of Test Set:", test_acc * 100.0, "%")

import csv
from itertools import zip_longest

# 用于存放多轮统计结果
# 打开CSV文件（如果文件不存在，它将创建一个新文件）
with open('4kind_csv/test_quan/Va2to4kind_pred/dense_pred'
          '/output_acc_Va2light_densenet169_quan_trainshai_allto4kind_10vote_10round_recall_time.csv',
          mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入列名
    writer.writerow(['acc_single', 'acc_vote', 'recall_single', 'recall_vote'])
    # 使用 zip_longest 填充较短的数组
    for item1, item2, item3, item4 in zip_longest(acc_single, acc_vote, recall1, recall2, fillvalue='N/A'):
        writer.writerow([item1, item2, item3, item4])
