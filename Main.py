#-*-coding:utf-8-*-
import sys,os,shutil
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示error信息
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       #仅适用第一个GPU
import warnings
warnings.filterwarnings('ignore')

import configparser
from getOut import demoToTxt0, demoToTxt1, demoToTxt2, ensemble
if __name__ == '__main__':
    # 读取配置信息
    config_file = os.path.join('./config.cfg')
    if not os.path.exists(config_file):
        raise FileNotFoundError('Config file {} not exist'.format(os.path.abspath(config_file)))
    CONFIG = configparser.ConfigParser()
    CONFIG.read(config_file)
    src_imgs_folder=CONFIG.get('PATH', 'src_imgs_folder')   # test pic folder
    if not os.path.exists(src_imgs_folder):
        raise FileNotFoundError('src_imgs_folder {} not exist'.format(os.path.abspath(src_imgs_folder)))
    final_txt_path=CONFIG.get('PATH', 'final_txt_path')     # result folder
    if not os.path.exists(os.path.dirname(final_txt_path)):
        os.makedirs(os.path.dirname(final_txt_path))
    # print(src_imgs_folder) # 检查配置信息是否加载正确，路径是否正常
    # print(final_csv_path)

    # 新建临时文件夹，用于存储中间数据
    temp_folder=os.path.abspath('./TEMP')
    if os.path.exists(temp_folder):     # 如果存在则清空
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    ### 使用Attn_sentisive_模型来预测保存到临时文件夹中
    # from demoToTxt_sen import demoToTxt0
    print('start to predict IDs of TestPic with CV_roi setX.npy,this will take some time...')
    txt_result_folder = os.path.join(temp_folder, 'txt_result')
    os.makedirs(txt_result_folder)
    AttnSe_txt_path=os.path.join(txt_result_folder,'AttnSe_result.txt')
    saved_model = 'we_need_you/150_coll_data/Attn_Sensitive/best_accuracy.pth'
    demoToTxt0(src_imgs_folder, saved_model, AttnSe_txt_path)

    # 用ctc_model来预测
    # from demoToTxtCTC import demoToTxt1
    print('start to predict IDs of RMB with ctpn_setX.npy,this will take some time...')
    CTC_txt_path = os.path.join(txt_result_folder, 'CTC_result.txt')
    saved_model = 'we_need_you/150_coll_data/CTC/best_accuracy.pth'
    demoToTxt1(src_imgs_folder, saved_model, CTC_txt_path)

    # from demoToTxt1 import demoToTxt2
    Attn_txt_path = os.path.join(txt_result_folder, 'Attn_result.txt')
    saved_model = 'we_need_you/150_coll_data/Attn_NoSensitive/best_accuracy.pth'
    demoToTxt2(src_imgs_folder, saved_model, Attn_txt_path)

    ### 对得到的三个csv结果进行集成，得到最终结果
    # from ensemble import ensemble
    ensemble(AttnSe_txt_path,CTC_txt_path,Attn_txt_path,final_txt_path)

    ### 善后：删除临时文件夹
    shutil.rmtree(temp_folder)

    print('GOOD!. All Finished!!!')
