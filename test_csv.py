import os
import csv
import pandas as pd
import numpy as np
import random


# 根据药材对应的csv生成对应的test.csv
def my_csv_lie(folder_name, num, id_num):
    for w in range(10):
        print(folder_name)
        # 获取文件夹中的所有文件名
        small_file = [f for f in os.listdir(folder_name) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
        print(len(small_file))
        small_num = list(range(0, len(small_file)))
        random.shuffle(small_num)
        test_samll_index = small_num[0:len(small_num)]
        with open('test_4kind_10round.csv', 'a', newline='', encoding='gbk') as file:
            writer = csv.writer(file)
            small_co = 0
            i = 0
            # 将所有图像都用一遍
            while i < len(test_samll_index):
                id_num = id_num + 1
                i = i + 4
                data = []
                for j in range(4):
                    data.append(folder_name + '/' + small_file[test_samll_index[small_co % len(test_samll_index)]])
                    small_co = small_co + 1
                data.append(num)
                data.append(id_num)
                writer.writerow(data)
            print(small_co)
    return id_num


folder_path = 'D:/File/crop3/light_test_Va24kind_test'
with open('test_4kind_10round.csv', 'w', newline='', encoding='gbk') as file:
    writer = csv.writer(file)
    writer.writerow(['Small Image 1', 'Small Image 2', 'Small Image 3', 'Small Image 4', 'Label', 'number'])

for filename in os.listdir(folder_path):
    id_num = 0
    file_path = folder_path + '/' + filename
    print(file_path)
    for filename1 in os.listdir(file_path):
        file_path1 = file_path + '/' + filename1
        print(file_path1)
        kind = filename1
        print(kind)
        id_num = my_csv_lie(file_path1, kind, id_num)

