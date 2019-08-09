import numpy as np
import csv

csv_path = '/home/yoiker/PycharmProjects/DataSet/人民币训练集/train_id_label.csv'
txt_path = '/home/yoiker/PycharmProjects/DataSet/人民币训练集/450train_id_label.txt'
txt_path2 = '/home/yoiker/PycharmProjects/DataSet/人民币训练集/50validation_id_label.txt'

def loadCSVfile(csvPath, txtPath, txtPath2 = None):
    list_file = []
    with open(csvPath, 'rt') as csv_file:       # 因为是读取的是文本文件，所以是'rt'or'r';若为二进制文件应为'rb'
        all_lines = csv.reader(csv_file)
        # print(all_lines)
        for one_line in all_lines:
            list_file.append(one_line)
        print(list_file[0], list_file[1])
        list_file.remove(list_file[0])
        arr_file = np.array(list_file)
        label = arr_file[:,0]
        j = 0
        for i in range(len(label)):
            if label[i] == 'NULL':      # 对null值做处理，若不需要处理，注释掉就可以了
                continue
            arr_file[j] = arr_file[i]
            j += 1
        # 将数据写入txt文件
        endIndex = len(arr_file)
        if txtPath2 != None:
            endIndex = round(len(arr_file)*0.7)
            file2 = open(txtPath2, 'w')
            for i in range(450, 501):
                file2.write(arr_file[i][0] + '\t' + arr_file[i][1] + '\n')
        file = open(txtPath, 'w')
        for i in range(450):
            file.write(arr_file[i][0] + '\t' + arr_file[i][1] + '\n')
    return True

if __name__ == "__main__":
    loadCSVfile(csv_path, txt_path, txt_path2)
    print("ok")
