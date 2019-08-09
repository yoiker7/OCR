# OCR
识别部分代码，通过训练好的三个模型进行对比产生识别结果；包含部分处理的代码，比较杂。本代码运行于ubuntu1604，基于deep-text-recognition-benchmark修改，参照:https://github.com/clovaai/deep-text-recognition-benchmark
训练时需要使用create_lmdb_dataset.py文件将训练集转换为lmdb数据集，而后进行训练
运行示例代码如下：
# 创建lmdb数据集
python3 create_lmdb_dataset.py --inputPath /home/yoiker/daraset/vali
d_data/ --gtFile /home/yoiker/daraset/valid_data/gt.txt --outputPath valid_data/

# demo.py
CUDA_VISIBLE_DEVICES=0 python3 demo.py --Transformation TPS --Featur
eExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder demo_image1/ --saved_model TPS-ResNet-BiLSTM-Attn-case-sensitive.pth --sensitive

CUDA_VISIBLE_DEVICES=0 python demo.py --Transformation TPS --Feature
Extraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder /home/yoiker/PycharmProjects/private_test_data --saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth --sensitive

# train.py
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data result/training
 --valid_data result/validation --select_data data --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --sensitive
