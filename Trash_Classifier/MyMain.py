from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
from keras import backend as K
import sys;

sys.path.append('../')

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


def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    """
    conv2d -> batch normalization -> relu activation
    """
    x = Conv2D(nb_filter, kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def shortcut(input, residual):
    """
    shortcut连接，也就是identity mapping部分。
    """

    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    identity = input
    # 如果维度不同，则使用1x1卷积进行调整
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_regularizer=regularizers.l2(0.0001))(input)

    return add([identity, residual])


def basic_block(nb_filter, strides=(1, 1)):
    """
    基本的ResNet building block，适用于ResNet-18和ResNet-34.
    """

    def f(input):
        conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
        residual = conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))

        return shortcut(input, residual)

    return f


def residual_block(nb_filter, repetitions, is_first_layer=False):
    """
    构建每层的residual模块，对应论文参数统计表中的conv2_x -> conv5_x
    """

    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2)
            input = basic_block(nb_filter, strides)(input)
        return input

    return f


def resnet_18(input_shape=(224, 224, 3), nclass=1000):
    """
    build resnet-18 model using keras with TensorFlow backend.
    :param input_shape: input shape of network, default as (224,224,3)
    :param nclass: numbers of class(output shape of network), default as 1000
    :return: resnet-18 model
    """
    input_ = Input(shape=input_shape)

    conv1 = conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    conv2 = residual_block(64, 2, is_first_layer=True)(pool1)
    conv3 = residual_block(128, 2, is_first_layer=True)(conv2)
    conv4 = residual_block(256, 2, is_first_layer=True)(conv3)
    conv5 = residual_block(512, 2, is_first_layer=True)(conv4)

    pool2 = GlobalAvgPool2D()(conv5)
    output_ = Dense(nclass, activation='softmax')(pool2)

    model = Model(inputs=input_, outputs=output_)
    model.summary()

    return model


def processing_data(data_path, batch_size=32, validation_split=0.1):
    """
    数据处理
    :param data_path: 数据集路径
    :return: train, test:处理后的训练集数据、测试集数据
    """
    # -------------------------- 实现数据处理部分代码 ----------------------------
    train_data = ImageDataGenerator(
        # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
        rescale=1. / 225,
        # 浮点数，剪切强度（逆时针方向的剪切变换角度）
        shear_range=0.1,
        # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        zoom_range=0.1,
        # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        width_shift_range=0.1,
        # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        height_shift_range=0.1,
        # 布尔值，进行随机水平翻转
        horizontal_flip=True,
        # 布尔值，进行随机竖直翻转
        vertical_flip=True,
        # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
        validation_split=validation_split
    )

    # 接下来生成测试集，可以参考训练集的写法
    test_data = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=validation_split)

    height, width = 384, 512
    train_generator = train_data.flow_from_directory(
        # 提供的路径下面需要有子目录
        data_path,
        # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
        target_size=(height, width),
        # 一批数据的大小
        batch_size=batch_size,
        # "categorical", "binary", "sparse", "input" 或 None 之一。
        # 默认："categorical",返回one-hot 编码标签。
        class_mode='categorical',
        # 数据子集 ("training" 或 "validation")
        subset='training',
        seed=0)

    validation_generator = test_data.flow_from_directory(
        data_path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=0)

    # ------------------------------------------------------------------------
    return train_generator, validation_generator


def model(train_data, test_data, model_save_path):
    """
    创建、训练和保存深度学习模型
    :param train_data: 训练集数据
    :param test_data: 测试集数据
    :param save_model_path: 保存模型的路径和名称
    :return:
    """
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------
    ### 创建模型
    input_shape = (384, 512, 3)
    model = resnet_18(input_shape, 6)

    # 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
    model.compile(
        # 是优化器, 主要有Adam、sgd、rmsprop等方式。
        optimizer='Adam',
        # 损失函数,多分类采用 categorical_crossentropy
        loss='categorical_crossentropy',
        # 是除了损失函数值之外的特定指标, 分类问题一般都是准确率
        metrics=['accuracy'])

    model.fit_generator(
        # 一个生成器或 Sequence 对象的实例
        generator=train_data,
        # epochs: 整数，数据的迭代总轮数。
        epochs=10,
        # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        steps_per_epoch=2259 // 32,
        # 验证集
        validation_data=test_data,
        # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        validation_steps=248 // 32)

    # 模型保存
    model.save(model_save_path)
    # 保存模型（请写好保存模型的路径及名称）
    # -------------------------------------------------------------------------

    return model


def evaluate_mode(test_data, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param test_data: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------
    # 加载模型
    model = load_model(save_model_path)
    # 获取验证集的 loss 和 accuracy
    loss, accuracy = model.evaluate_generator(test_data)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
    # ---------------------------------------------------------------------------


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    data_path = "../../DataSets/dataset-resized"  # 数据集路径
    save_model_path = 'results/ResNet18.h5'  # 保存模型路径和名称

    # 获取数据
    train_data, test_data = processing_data(data_path)

    # 创建、训练和保存模型
    model(train_data, test_data, save_model_path)

    # 评估模型
    evaluate_mode(test_data, save_model_path)


if __name__ == '__main__':
    main()