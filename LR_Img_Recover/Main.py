from matplotlib import pyplot as plt  # 展示图片
import numpy as np  # 数值处理
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析
import math


def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)

    # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)

    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()


def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)

    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()

    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max

        # 转换图片数组数据类型
        img = img.astype(np.uint8)

    # 生成图片
    cv2.imwrite(filename, img)


def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)

    # 图像数组数据放缩在 0-1 之间
    return image.astype(np.double) / info.max


def noise_mask_image(img, noise_ratio):
    """
    根据题目要求生成受损图片
    :param img: 图像矩阵，一般为 np.ndarray
    :param noise_ratio: 噪声比率，可能值是0.4/0.6/0.8
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------
    R = np.random.random(img.shape)
    I = np.array(R >= noise_ratio, dtype='double')
    M, N, C = img.shape
    while True:
        total = I.sum() / (M * N * C)
        if abs(total - noise_ratio) <= 3.5:
            break
        R = np.random.random(img.shape)
        I = np.array(R >= noise_ratio, dtype='double')
    noise_img = img * I

    # -----------------------------------------------
    return noise_img


def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')


def compute_error(res_img, img):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像
    :param img:原始图像
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    # 初始化
    error = 0.0

    # 将图像矩阵转换成为np.narray
    res_img = np.array(res_img)
    img = np.array(img)

    # 如果2个图像的形状不一致，则打印出错误结果，返回值为 None
    if res_img.shape != img.shape:
        print("shape error res_img.shape and img.shape %s != %s" % (res_img.shape, img.shape))
        return None

    # 计算图像矩阵之间的评估误差
    error = np.sqrt(np.sum(np.power(res_img - img, 2)))

    return round(error, 3)


def CalDis(i, j, ci, cj):
    return max(ci - i, i - ci) + max(cj - j, j - cj)


def LocalValue(i, j, imgSlide, mskSlide, size):
    rowU = max(i - size, 0)
    rowD = min(i + size, imgSlide.shape[0] - 1)

    colL = max(j - size, 0)
    colR = min(j + size, imgSlide.shape[1] - 1)

    y = []
    totalDis = 0
    for ci in range(rowU, rowD + 1):
        for cj in range(colL, colR + 1):
            if mskSlide[ci, cj] != 0.0:
                dis = size * 2 - CalDis(i, j, ci, cj) + 1  # Laplace
                dis = dis ** 8
                totalDis += dis
                y.append(imgSlide[ci, cj] * dis)
    y = np.array(y, ndmin=1)
    if y.shape[0] == 0:
        return 0.5
    return y.sum()/totalDis


def Predict(i, j, imgSlide, mskSlide, localValue, size):
    rowD = min(i + size, imgSlide.shape[0] - 1) + 1
    colR = min(j + size, imgSlide.shape[1] - 1) + 1

    X = []
    y = []
    for ci in range(i, rowD):
        for cj in range(j, colR):
            # print(type(X))
            X.append([ci, cj])
            y.append(localValue[ci, cj])

    X = np.array(X, ndmin=2)
    y = np.array(y, ndmin=1)

    clf = LinearRegression()
    clf.fit(X, y)

    for ci in range(i, rowD):
        for cj in range(j, colR):
            if mskSlide[ci, cj] == 0.0:
                XPred = np.array([i, j], ndmin=2)
                imgSlide[i, j] = clf.predict(XPred)[0]
                imgSlide[i, j] = min(max(imgSlide[i, j], 0), 1)
    return rowD, colR, imgSlide[i:i+rowD, j:j+colR]


def restore_image(noise_img, size=4):
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------
    # res_img = normalization(res_img)

    for c in range(3):
        # print('slide =', c)
        imgSlide = res_img[:, :, c]
        mskSlide = noise_mask[:, :, c]
        localValue = res_img[:, :, c]

        M, N = imgSlide.shape
        for i in range(M):
            for j in range(N):
                if mskSlide[i, j] == 0.0:
                    # 求localValue
                    localValue[i, j] = LocalValue(i, j, imgSlide, mskSlide, 2)
                else:
                    localValue[i, j] = imgSlide[i, j]

        for i in range(0, M, size):
            for j in range(0, N, size):
                rowD, colR, temp = Predict(i, j, imgSlide, mskSlide, localValue, size)
                res_img[i:i+rowD, j:j+colR, c] = temp
    # ---------------------------------------------------------------
    return res_img


def Main():
    load_img_path = 'A.png'
    # load_img_path = '/home/zju-warren/Pictures/1.jpg'

    for noiseRatio in [0.4, 0.6, 0.8]:
        # 读取图片\标准化
        orgImg = read_image(load_img_path)
        normImg = normalization(orgImg)

        noiseImg = noise_mask_image(normImg, noiseRatio)
        # plot_image(noiseImg, 'noiseImg')

        restoreImg = restore_image(noiseImg)
        plot_image(restoreImg, 'restoreImg')
        # save_image(img_store_path % noiseRatio, restoreImg)

        basicErr = compute_error(noiseImg, normImg)
        imgErr = compute_error(restoreImg, normImg)
        score = -math.log(imgErr/basicErr) * (1 + 210/basicErr)
        print('imgErr = %f, score = %f' % (imgErr, score))


if __name__ == '__main__':
    Main()
