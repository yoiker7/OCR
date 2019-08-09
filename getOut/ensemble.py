#-*-coding:utf-8-*-
from collections import OrderedDict
import re
from collections import Counter
def ensemble(Attnse_result_path, ctc_result1_path, Attn_result2_path, save_path):

    def load_result(txt_path):
        result_dict = OrderedDict()
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            name, code = line.strip().split('\t')
            result_dict[name] = code.strip()
        return result_dict

    def valid_str(strA, rep_str='?'):   # 对读取的字符串做处理
        def get_num(str_, ch='num'):
            if ch == 'num':
                return len(re.compile('[0-9]').findall(str_))
            else:
                return len(re.compile('[A-Z]').findall(str_))

        def spe_pro1(strA, chinaOnly=None):  # 处理 O, Q, I特殊字符
            spe_char0 = ['O', 'Q']
            spe_char1 = ['I']
            if chinaOnly is not None:
                if strA[0] != 'L':
                    strA = 'L' + strA[1:]
            for i in range(len(strA)):
                if strA[i] in spe_char0:
                    strA = strA[:i] + '0' + strA[i + 1:]
                elif strA[i] in spe_char1:
                    strA = strA[:i] + '1' + strA[i + 1:]
            return strA

        def spe_pro2(strA):  # 处理第九位与第十位特殊字符
            spe_num9 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']
            if strA[8] not in spe_num9:
                strA = strA[:8] + rep_str + strA[9:]
            if strA[9] == 'Z':
                strA = strA[:9] + rep_str + strA[10:]
            return strA

        if len(strA) == 17:
            strA = spe_pro1(strA)
            strA = spe_pro2(strA)
            last_num = get_num(strA[-4:], 'num')  # 最后四位是数字
            if last_num == 4:
                return strA
            else:
                last_str = re.sub(re.compile('[A-Z]'), rep_str, strA[-4:])
                strA = strA[:-4] + last_str
                return strA

        if len(strA) > 17:  # 长度不为17时，全部变为17
            strA = rep_str * 17
            return strA
        elif get_num(spe_pro1(strA)[-4:], 'num') == 4:
            strA = spe_pro1(strA)
            header_len = len(strA) - 4
            tail = rep_str * (13 - header_len)
            strA = strA[:-4] + tail + strA[-4:]
        else:
            strA = spe_pro1(strA)
            tail = rep_str * (17 - len(strA))
            strA = strA + tail

        strA = spe_pro2(strA)  # 处理位置特殊字符

        return strA + ' !'

    def get_compare(str1, str2, str3):
        def is_char_same(strA, strB, strC):
            return [(strA[i] == strB[i] and strB[i] == strC[i]) for i in range(17)]

        str1 = valid_str(str1)
        str2 = valid_str(str2)
        str3 = valid_str(str3)
        is_same = is_char_same(str1, str2, str3)
        combined_str = []
        for idx, same in enumerate(is_same):
            if same:  # 如果三个字符都相同
                temp_ch = 'A' if idx < 13 else '2'
                temp_ch= temp_ch if str1[idx] == '?' else str1[idx]
                combined_str.append(temp_ch)
            else:  # 如果三个字符都不同，获取每个字符的个数
                count = Counter([str1[idx], str2[idx], str3[idx]])      # 输出：Counter({'c': 3, 'a': 2, 'b': 2})
                temp = sorted(count.items(), key=lambda x: x[1], reverse=True)
                ch = temp[1][0] if temp[0][0] == '?' else temp[0][0]
                if temp[0][1] == 1:  # 如果个数都是1个，先考虑str3,在考虑str2,最后设为0
                    ch=str1[idx] if str1[idx] !='?' else str2[idx] if str2[idx] !='?' else '0'
                combined_str.append(ch)
        return ''.join(combined_str)

    Attnse_dict = load_result(Attnse_result_path)
    ctc_dict = load_result(ctc_result1_path)
    Attn_dict = load_result(Attn_result2_path)
    new_dict = OrderedDict()
    new_dict['name'] = 'label'
    default_value = '?' * 17
    for key, value in Attnse_dict.items():
        if key == 'name': continue
        Attnse_value = value  # 初始化
        ctc_value = ctc_dict.get(key, default_value)    # ctpn_dict1的长度可能偏少，有些图片没有找到ROI
        Attn_value = Attn_dict.get(key, default_value)
        if ctc_value.endswith('!'):  # 如果结果最后一位为Z，代表结果预测不准，全部用?代替。
            ctc_value = default_value
        if Attn_value.endswith('!'):
            Attn_value = default_value
        real_value = get_compare(Attnse_value, ctc_value, Attn_value)
        new_dict[key] = real_value

    def write_dict_result(dict_result, save_path):
        if 'name' in dict_result.keys():
            dict_result.pop('name')
        with open(save_path, 'w') as file:
            # file.write('name\tlabel\n')
            for key, value in dict_result.items():
                file.write(key + '\t' + value + '\n')
                print(key + '\t' + value + '\n')

    write_dict_result(new_dict, save_path) # 将最终结果保存到save_path中
    print('finished. Ensembled result is saved to {}'.format(save_path))

if __name__ == "__main__":
    Attnse_result_path = 'testTxt0.txt'
    ctc_result1_path = 'testTxt.txt'
    Attn_result2_path = 'testTxt1.txt'
    save_path = 'testTxtRes.txt'
    ensemble(Attnse_result_path, ctc_result1_path, Attn_result2_path, save_path)