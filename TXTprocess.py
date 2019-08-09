readfilename = '/home/yoiker/PycharmProjects/DataSet/gtFile.txt'
writefilename = '/home/yoiker/PycharmProjects/DataSet/gtResFile.txt'

def spe_pro1(strA, chinaOnly=None):  # 处理 O, Q, I特殊字符
    spe_char = ['O', 'Q', 'I']
    spe_num9 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']
    if chinaOnly is not None:
        if strA[0] != 'L':
            return False
    if len(strA) != 17 or strA[8] not in spe_num9:
        return False
    for i in range(13):
        if strA[i] in spe_char:
            return False
    spe_num9.remove('X')
    for i in range(13, 17):
        if strA[i] not in spe_num9:
            return False
    return True


def getMydata(filePath, resFilePth):        # 只筛选出我需要的车架号数据
    with open(filePath, 'r', encoding='utf-8') as data:
        dataList = data.readlines()
    nSamples = len(dataList)
    res = []
    file = open(resFilePth, 'w')
    for i in range(nSamples):
        p_tmp = [j for j in dataList[i].strip('\n').split(',')]
        # print(p_tmp[-1])
        # p_tmp.remove(p_tmp[8])
        # res.append(p_tmp)
        if spe_pro1(p_tmp[-1]):
            for j in range(len(p_tmp) - 1):
                file.write(p_tmp[j] + ',')
            file.write(p_tmp[-1] + '\n')
    print("over!")

def getitsdata(filePath, resFilePth):       # 加0 ，1 标签
    with open(filePath, 'r', encoding='utf-8') as data:
        dataList = data.readlines()
    nSamples = len(dataList)
    res = []
    file = open(resFilePth, 'w')
    for i in range(nSamples):
        p_tmp = [j for j in dataList[i].strip('\n').split(',')]
        # print(p_tmp)
        # p_tmp = changeName(p_tmp)
        # print(p_tmp[0])
        if spe_pro1(p_tmp[-1]):
            for j in range(len(p_tmp) - 1):
                file.write(p_tmp[j] + ',')
            file.write('1' + ',' + p_tmp[-1] + '\n')
        else:
            for j in range(len(p_tmp) - 1):
                file.write(p_tmp[j] + ',')
            file.write('0' + ',' + p_tmp[-1] + '\n')

def changeName(listA, index=0):     # 更改txt文件index处的文件名
    a = (list(listA[0]))
    a.insert(-4, 'C')       # 在index=-4处插入'b'
    listA[0] = ''.join(a)
    return listA

def getRenamedTxt(filePath, resFilePth):
    with open(filePath, 'r', encoding='utf-8') as data:
        dataList = data.readlines()
    nSamples = len(dataList)
    res = []
    file = open(resFilePth, 'w')
    for i in range(nSamples):
        p_tmp = [j for j in dataList[i].strip('\n').split('\t')]
        # print(p_tmp[-1])
        p_tmp = changeName(p_tmp)
        # print(p_tmp[0])
        for j in range(len(p_tmp) - 1):
            file.write(p_tmp[j] + '\t')
        file.write(p_tmp[-1] + '\n')
    print("over!")

if __name__ == "__main__":
    # getitsdata(readfilename,writefilename)
    getRenamedTxt(readfilename, writefilename)