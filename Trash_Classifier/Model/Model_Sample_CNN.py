# 导入相关包
from keras.layers import Input, Dense, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import TensorBoard
import time
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta


def cnn_model(input_shape, train_generator, validation_generator, model_save_path='results/cnn.h5',
              log_dir="results/logs/"):
    """
    该函数实现 Keras 创建深度学习模型的过程
    :param input_shape: 模型数据形状大小，比如:input_shape=(384, 512, 3)
    :param train_generator: 训练集
    :param validation_generator: 验证集
    :param model_save_path: 保存模型的路径
    :param log_dir: 保存模型日志路径
    :return: 返回已经训练好的模型
    """
    # Input 用于实例化 Keras 张量。
    # shape: 一个尺寸元组（整数），不包含批量大小。 例如，shape=(32,) 表明期望的输入是按批次的 32 维向量。
    # inputs = Input(shape=input_shape)
    #
    # cnn = Conv2D(96, (5, 5), activation='relu')(inputs)
    # cnn = MaxPool2D(pool_size=(2, 2))(cnn)
    # cnn = Conv2D(256, (3, 3), activation='relu')(cnn)
    # cnn = MaxPool2D(pool_size=(2, 2))(cnn)
    #
    # cnn = Flatten()(cnn)
    #
    # cnn = Dropout(0.5)(cnn)
    # cnn = Dense(128, activation='relu')(cnn)
    # cnn = Dropout(0.5)(cnn)
    # cnn = Dense(6, activation='softmax')(cnn)
    #
    # outputs = cnn
    #
    # # 生成一个函数型模型
    # model = Model(inputs=inputs, outputs=outputs)
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        MaxPool2D(pool_size=2),

        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2),

        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2),

        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2),

        Flatten(),

        Dense(64, activation='relu'),

        Dense(6, activation='softmax')
    ])

    # 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
    model.compile(
            # 是优化器, 主要有Adam、sgd、rmsprop等方式。
            optimizer='Adam',
            # 损失函数,多分类采用 categorical_crossentropy
            loss='categorical_crossentropy',
            # 是除了损失函数值之外的特定指标, 分类问题一般都是准确率
            metrics=['accuracy'])

    # 可视化，TensorBoard 是由 Tensorflow 提供的一个可视化工具。
    tensorboard = TensorBoard(log_dir)

    # 训练模型, fit_generator函数:https://keras.io/models/model/#fit_generator
    # 利用Python的生成器，逐个生成数据的batch并进行训练。
    # callbacks: 实例列表。在训练时调用的一系列回调。详见 https://keras.io/callbacks/。
    d = model.fit_generator(
            # 一个生成器或 Sequence 对象的实例
            generator=train_generator,
            # epochs: 整数，数据的迭代总轮数。
            epochs=8,
            # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
            steps_per_epoch=2259 // 32,
            # 验证集
            validation_data=validation_generator,
            # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
            validation_steps=248 // 32,
            callbacks=[tensorboard])
    # 模型保存
    model.save(model_save_path)

    return d, model
