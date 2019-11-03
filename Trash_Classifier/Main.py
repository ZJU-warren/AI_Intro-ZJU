import sys ;sys.path.append('../')
import Trash_Classifier.DataLinkSet as DLSet
import Trash_Classifier.DataGenerator as DGen

# import Trash_Classifier.Model.Model_Sample_DNN as DNN
import Trash_Classifier.Model.Model_Sample_CNN as CNN
from keras.layers import Input, Dense, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import glob, os, cv2, random, time
import numpy as np


# 读取数据集
def FetchDataList():
    # 获取数据名称列表
    imgList = glob.glob(os.path.join(DLSet.dataSet_link, '*/*.jpg'))
    return imgList


# 训练测试集生成器
def GetDGener():
    # 图像数据的行数和列数
    height, width = 384, 512
    # 获取训练数据和验证数据集
    train_generator, validation_generator = DGen.processing_data(DLSet.dataSet_link, height, width)
    # 通过属性class_indices可获得文件夹名与类的序号的对应字典。 (类别的顺序将按照字母表顺序映射到标签值)。
    labels = train_generator.class_indices
    print(labels)

    # 转换为类的序号与文件夹名对应的字典
    labels = dict((v, k) for k, v in labels.items())
    print(labels)


def Train():
    # 开始时间
    start = time.time()
    # 图像数据的行数和列数
    height, width = 384, 512
    # 获取训练数据和验证数据
    train_generator, validation_generator = DGen.processing_data(DLSet.dataSet_link, height, width)
    # 定义模型输入大小
    input_shape = (384, 512, 3)
    # 训练模型，获取训练过程和训练后的模型
    res, model = CNN.cnn_model(input_shape, train_generator, validation_generator)
    # 打印模型概况和模型训练总数长
    model.summary()
    print("模型训练总时长：", time.time() - start)
    return res, model


def plot_training_history(res):
    """
    绘制模型的训练结果
    :param res: 模型的训练结果
    :return:
    """
    # 绘制模型训练过程的损失和平均损失
    # 绘制模型训练过程的损失值曲线，标签是 loss
    plt.plot(res.history['loss'], label='loss')

    # 绘制模型训练过程中的平均损失曲线，标签是 val_loss
    plt.plot(res.history['val_loss'], label='val_loss')

    # 绘制图例,展示出每个数据对应的图像名称和图例的放置位置
    plt.legend(loc='upper right')

    # 展示图片
    plt.show()

    # 绘制模型训练过程中的的准确率和平均准确率
    # 绘制模型训练过程中的准确率曲线，标签是 acc
    plt.plot(res.history['accuracy'], label='accuracy')

    # 绘制模型训练过程中的平均准确率曲线，标签是 val_acc
    plt.plot(res.history['val_accuracy'], label='val_accuracy')

    # 绘制图例,展示出每个数据对应的图像名称，图例的放置位置为默认值。
    plt.legend()

    # 展示图片
    plt.show()


# 主函数
def Main():
    # imgList = FetchDataList()
    # GetDGener()
    res, model = Train()
    # 绘制模型训练过程曲线
    plot_training_history(res)


if __name__ == '__main__':
    Main()
