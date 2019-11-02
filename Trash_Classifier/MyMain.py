from keras.models import load_model
from keras.preprocessing import image
import os


def processing_data(data_path):
    """
    数据处理
    :param data_path: 数据集路径
    :return: train, test:处理后的训练集数据、测试集数据
    """
    # -------------------------- 实现数据处理部分代码 ----------------------------

    # ------------------------------------------------------------------------
    train_data, test_data = None, None
    return train_data, test_data


def model(train_data, test_data, save_model_path):
    """
    创建、训练和保存深度学习模型
    :param train_data: 训练集数据
    :param test_data: 测试集数据
    :param save_model_path: 保存模型的路径和名称
    :return:
    """
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------

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

    # ---------------------------------------------------------------------------


def predict(img):
    """
    加载模型和模型预测
    主要步骤:
        1.加载模型(请加载你认为的最佳模型)
        2.图片处理
        3.用加载的模型预测图片的类别
    :param img: PIL.Image 对象
    :return: string, 模型识别图片的类别,
            共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    """
    # 把图片转换成为numpy数组
    img = image.img_to_array(img)

    # 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
    # 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/dnn.h5'
    model_path = ''
    try:
        # 作业提交时测试用, 请勿删除此部分
        model_path = os.path.realpath(__file__).replace('main.py', model_path)
    except NameError:
        model_path = './' + model_path

    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 加载模型
    model = None

    # -------------------------------------------------------------------------
    predict = None

    # 返回图片的类别
    return predict


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    data_path = None  # 数据集路径
    save_model_path = None  # 保存模型路径和名称

    # 获取数据
    train_data, test_data = processing_data(data_path)

    # 创建、训练和保存模型
    model(train_data, test_data, save_model_path)

    # 评估模型
    evaluate_mode(test_data, save_model_path)


if __name__ == '__main__':
    # 输入图片路径和名称
    img_path = 'test.jpg'

    # 打印该张图片的类别
    img = image.load_img(img_path)
    print(predict(img))
