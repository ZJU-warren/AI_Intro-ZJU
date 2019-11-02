# 方式一: 使用 .add() 方法将各层添加到模型中
from keras.models import Sequential
from keras.layers import Dense, Activation

def GenModel():
    # 选择模型，选择序贯模型（Sequential())
    model = Sequential()
    # 构建网络层
    # 添加全连接层，输入784维,输出空间维度32
    model.add(Dense(32, input_shape=(784,)))
    # 添加激活层，激活函数是 relu
    model.add(Activation('relu'))
    # 打印模型概况
    model.summary()


if __name__ == '__main__':
    GenModel()
