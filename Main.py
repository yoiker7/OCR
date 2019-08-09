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

    # ### 第一步：使用CV_ROI模块中的extract_ROI.py来提取出图片中的ROI，并保存到一个临时文件夹中
    # print('start to extract_CV_roi imgs by CV_ROI module')
    # from CV_ROI import get_save_roi
    # CV_roi_imgs=os.path.join(temp_folder,'cv_roi_imgs')
    # os.makedirs(CV_roi_imgs)
    # get_save_roi(src_imgs_folder,CV_roi_imgs)   # CV_ROI get the roi
    # print('extraxt CV_roi imgs DONE!')
    #
    # ### 第二步：使用CTPN_ROI模块中的extract_ROI.py来提取出图片中的ROI，并保存到另一个临时文件夹中
    # # 下面的代码必须在Linux中运行，Windows下会出错
    # print('start to extract CTPN_roi imgs by CTPN_ROI module')
    # from CTPN_ROI import get_save_ctpn_roi
    # CTPN_roi_imgs=os.path.join(temp_folder,'ctpn_roi_imgs')
    # os.makedirs(CTPN_roi_imgs)
    # get_save_ctpn_roi(src_imgs_folder,CTPN_roi_imgs)    # CTPN_ROI get the roi
    # print('extract CTPN_roi imgs DONE!')
    #
    # ### 第三步：将提取的ROI图片整理为npy格式的数据集，并保存该数据集到临时文件夹中，用于后续CTC_Models的编码预测
    # print('start to load CV_rois to setX.npy')
    # from CTC_Models import load_roi_setX
    # roiset_folder=os.path.join(temp_folder,'roiset')
    # os.makedirs(roiset_folder)
    #
    # # 对CV_ROI模块得到的roi_imgs进行提取并保存为npy格式的setX
    # cv_setX_path=os.path.join(roiset_folder,'cv_setX.npy')
    # cv_names_path=os.path.join(roiset_folder,'cv_names.npy')
    # load_roi_setX(CV_roi_imgs,cv_setX_path,cv_names_path,size=(80,240))
    # print('load CV_rois DONE!')
    #
    # # 对CTPN_ROI模块得到的roi_imgs提取并保存为npy格式的setX
    # ctpn_setX_path=os.path.join(roiset_folder,'ctpn_setX.npy')
    # ctpn_names_path=os.path.join(roiset_folder,'ctpn_names.npy')
    # load_roi_setX(CTPN_roi_imgs, ctpn_setX_path, ctpn_names_path, size=(60, 240))
    # print('load CTPN_rois DONE!')

    ### 第四步：使用Attn_sentisive_模型来预测保存到临时文件夹中
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

    ### 第五步：对得到的三个csv结果进行集成，得到最终结果
    # from ensemble import ensemble
    ensemble(AttnSe_txt_path,CTC_txt_path,Attn_txt_path,final_txt_path)

    ### 善后：删除临时文件夹
    shutil.rmtree(temp_folder)

    print('GOOD!. All Finished!!!')
