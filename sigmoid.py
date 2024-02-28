import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, fsolve
import torch
from torch.optim import Adam
from torch import optim
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

global em_x, em_y, cut_ratio

"""
拉格朗日插值公式
"""


def L(xinput: int, k: int, xcord: list):  # 拉格朗日基函数
    return np.prod([(xinput - xcord[i]) / (xcord[k] - xcord[i]) for i in range(len(xcord)) if i != k])


def lagrange(xinput: int, xcord: list, ycord: list):
    res = 0
    for i in range(len(ycord)):
        res += ycord[i] * L(xinput, i, xcord)
    return res


"""
第一次拟合sigmoid拟合函数(体外通路活性->体外浓度)
"""


def sigmoid(x, L, x0, k, b):  # sigmoid基函数
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def sigmoid_fit(x_data, y_data, x_range):
    # if min(x_range)<min(x_data) or max(x_range)>max(x_data):
    # return "输入数据范围错误!"
    # 初始参数猜测（L，x0，k，b）
    p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
    # 使用curve_fit进行拟合
    popt, _ = curve_fit(sigmoid, x_data, y_data, p0, bounds=(0, 10000) maxfev=1000)
    print("sigmoid参数:", popt)
    return sigmoid(x_range, *popt)


"""
线性拟合函数(体外浓度->体内浓度)
"""


def linear_function(x, a, b):  # 线性函数基函数
    return a * x + b


def linear_fit(x_data, y_data, x_range):
    params, _ = curve_fit(linear_function, x_data, y_data)
    print("线性拟合参数:", params)
    # 提取拟合参数值
    slope, intercept = params

    return slope * x_range + intercept


"""
第二次拟合Sigmoid函数(体内浓度->体内通路活性)
"""


def eq(x):
    return (x * math.log(x / (cut_ratio * cut_ratio))) - 4 * (em_y)


def Sigmoid(x_range, x_data, d):  # Sigmoid基函数
    x_guess = 2.0
    L = 4 * (fsolve(eq, x_guess)[0])
    b = (x_data[-1] * math.log(L / cut_ratio) - x_data[0] * math.log(cut_ratio)) / em_x
    k = math.log(L / (cut_ratio * cut_ratio)) / em_x
    y = L / (1 + torch.exp(b - k * x_range)) + d
    return y


def Sigmoid_fit(x_data, y_data, x_range):
    device = "cpu"
    d = torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device))
    params = [d]
    optimizer = optim.Adam(params, lr=0.01)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.6, step_size=5000)
    xdata = torch.Tensor(x_data)
    ydata = torch.Tensor(y_data)
    pre_loss = float(math.inf)
    for i in range(30000):
        optimizer.zero_grad()
        predict_y = Sigmoid(xdata, xdata, *params)
        loss = torch.mean(torch.pow(ydata - predict_y, 2))
        if torch.abs(loss - pre_loss) < 0.001:
            break
        pre_loss = loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if i % 100 == 0:
            print("step=%s loss = %.5f" % (i, loss))
    print(params)
    predict_y = Sigmoid(xdata, xdata, *params)
    loss = torch.mean(torch.pow(ydata - predict_y, 2))
    print(predict_y)
    print("final loss = ", loss, )
    xrange = torch.Tensor(x_range)
    # xrange = np.linspace(np.min(x_data), np.max(y_data), 1000)
    y_predict = Sigmoid(xrange, xdata, *params)
    print(y_predict)
    # fig, ax = plt.subplots()
    # ax.plot(xrange.numpy(), y_predict.detach().numpy(), label="predict")
    # ax.plot(xdata.numpy(), ydata.numpy(), label="true")
    # ax.legend()
    # plt.show(block=True)
    return y_predict.detach().numpy()


"""三次指数平滑时间序列预测处理数据不等长问题"""


def time_seq(seqInput: list, length: int):
    global a, b, c
    alpha = [0.1 * i for i in range(1, 10)]  # 加权系数选择
    if len(seqInput) <= 3:
        return "Warning：Insufficient Data!"
    init = (seqInput[0] + seqInput[1] + seqInput[2]) / 3
    best_w = 0
    error = float(np.inf)
    output = [x for x in seqInput]
    A, B, C = 0, 0, 0
    S1, S2, S3 = 0, 0, 0
    for w in alpha:
        # print("==>加权系数{}<==:".format(w))
        res = [0] * (len(seqInput) - 1)
        s1 = s2 = s3 = init
        cur_error = 0
        for i in range(len(seqInput) - 1):
            s1 = w * seqInput[i] + (1 - w) * s1
            s2 = w * s1 + (1 - w) * s2
            s3 = w * s2 + (1 - w) * s3
            a = 3 * s1 - 3 * s2 + s3
            b = ((6 - 5 * w) * s1 - 2 * (5 - 4 * w) * s2 + (4 - 3 * w) * s3) * w / (2 * np.power(1 - w, 2))
            c = (s1 - 2 * s2 + s3) * np.power(w, 2) / (2 * np.power(1 - w, 2))
            # print("第{}次预测".format(i + 1))
            # print("一次平滑值:{},二次平滑值:{},三次平滑值:{}".format(s1, s2, s3))
            res[i] = a + b + c
        for i in range(1, len(seqInput)):
            cur_error += np.power((res[i - 1] - seqInput[i]), 2)
        # cur_error = sum([abs(res[i-1] - seqInput[i]) for i in range(1,len(seqInput))])
        if cur_error < error:
            seqoutput = res
            error = cur_error
            A, B, C = a, b, c
            S1, S2, S3 = s1, s2, s3
            best_w = w
    last_id = seqInput[-1]
    S1 = best_w * last_id + (1 - best_w) * S1
    S2 = best_w * S1 + (1 - best_w) * S2
    S3 = best_w * S2 + (1 - best_w) * S3
    A = 3 * S1 - 3 * S2 + S3
    B = ((6 - 5 * best_w) * S1 - 2 * (5 - 4 * best_w) * S2 + (4 - 3 * best_w) * S3) * best_w / (
            2 * np.power(1 - best_w, 2))
    C = (S1 - 2 * S2 + S3) * np.power(best_w, 2) / (2 * np.power(1 - best_w, 2))
    for i in range(length):
        pred = A + B * (i + 1) + C * (i + 1) * (i + 1)
        output.append(pred)
    print("最优加权系数:", best_w)
    # print(seqoutput)
    print("预测序列:", output)
    return output


"""
排序
"""


def bubble_sort(arr1, arr2):
    if len(arr1) != len(arr2):
        return "输入的两个数组长度不一致"
    n = len(arr1)
    for i in range(n):
        # 经过i轮冒泡后，后i个元素已经有序，不需要再比较
        for j in range(n - i - 1):
            if arr1[j] > arr1[j + 1]:
                # 如果前一个元素比后一个元素大，则交换它们的位置
                arr1[j], arr1[j + 1] = arr1[j + 1], arr1[j]
                arr2[j], arr2[j + 1] = arr2[j + 1], arr2[j]
    return arr1, arr2


def data_test(m, n, x, y, show_ratio: int):
    global em_x, em_y, cut_ratio
    x, y = bubble_sort(x, y)
    # test_mol_n = np.linspace(np.min(n), np.max(n) + 20, 1000)  # 测试的体外通路活性
    # test_mol_n = np.linspace(np.min(n), np.max(n), 1000)
    test_mol_n = np.linspace(np.min(n), np.max(n), 1000)  # 测试的体外通路活性

    """拟合n->m"""
    # test_mol_m = np.array([lagrange(n0, list(n), list(m)) for n0 in test_mol_n])  # 映射n—>m
    test_mol_m = np.array(sigmoid_fit(n, m, test_mol_n))  # sigmoid拟合映射n—>m

    """时间序列预测补全数据"""
    if len(m) != len(x):
        if len(m) < len(x):
            n = time_seq(list(n), len(x) - len(n))
            m = time_seq(list(m), len(x) - len(m))
        else:
            x = time_seq(list(x), len(m) - len(x))
            y = time_seq(list(y), len(m) - len(y))

    """拟合m->x"""
    test_mol_x = np.array(linear_fit(m, x, test_mol_m))
    max_x = max(x)
    min_x = min(y)
    max_test_x = max(test_mol_x)
    min_test_x = min(test_mol_x)
    test_mol_x = np.array(
        [(min_x + (x - min_test_x) * (max_x - min_x) / (max_test_x - min_test_x)) for x in
         test_mol_x])  # 统一x与test_mol_x的尺度

    em_x = x[-1] - x[0]
    em_y = y[-1] - y[0]

    """拟合x->y"""
    test_mol_y = np.array(Sigmoid_fit(x, y, list(test_mol_x)))

    plt.rcParams['font.sans-serif'] = [u'simHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(2, 2, 1)
    plt.title("P1")
    plt.xlabel("体外通路活性")
    plt.ylabel("体外浓度")
    # plt.scatter(test_mol_n, test_mol_m, color='violet')
    plt.scatter(n, m, color='violet')

    plt.subplot(2, 2, 2)
    plt.title("P2")
    plt.xlabel("体外浓度")
    plt.ylabel("体内浓度")
    plt.scatter(m, x, color='violet')

    plt.subplot(2, 2, 3)
    plt.title("P3")
    plt.xlabel("体内浓度")
    plt.ylabel("体内通路活性")
    plt.scatter(x, y, color='violet')

    plt.subplot(2, 2, 4)
    plt.title("拟合")
    plt.xlabel("体外通路活性")
    plt.ylabel("体内通路活性")
    plt.scatter(test_mol_n, test_mol_y, color='violet', s=5)
    ratio(show_ratio=show_ratio, test_mol_n=test_mol_n, test_mol_y=test_mol_y)
    plt.show()


"""
show_ratio:设置需要查看纵坐标的比例,取值在0-100
"""


def ratio(show_ratio: int, test_mol_n, test_mol_y):
    if show_ratio < 0 or show_ratio > 100:
        return "Invalid show_ratio value!"
    flag = False
    for i in range(len(test_mol_n) - 1):

        sign = min(test_mol_y) + (max(test_mol_y) - min(test_mol_y)) * show_ratio * 0.01
        if test_mol_y[i] <= sign <= test_mol_y[i + 1]:
            plt.plot(test_mol_n,
                     np.array([sign for _ in range(len(test_mol_y))]), "--", color="black", )
            plt.plot(np.array([test_mol_n[i] for _ in range(len(test_mol_y))]), test_mol_y, "--", color="black", )
            plt.text(test_mol_n[i], sign, "纵坐标{}%".format(show_ratio), fontdict={'size': 12, 'color': "red"})
            flag = True
            print("纵坐标为{}%时对应的坐标为:".format(show_ratio), (test_mol_n[i], sign))
            break
    if not flag:
        return "Invalid Error"


if __name__ == "__main__":
    #想看清清穿黑丝
    """这里输入数据"""
    m = np.array([0, 1.256, 6.28, 31.4])  # 体外浓度
    n = np.array([21.178, 50, 19.635, 18.589])  # 体外通路活性
    m, n = bubble_sort(m, n)

    # x = np.array([0, 5,8, 10])
    x = np.array([50, 30, 100, 500])  # 体内浓度
    y = np.array([28.18, 27.69, 33.39, 36.98])  # 体内通路活性

    cut_ratio = 0.00001  # 这个参数越小最后输出的曲线保留的部分越多设置在0.01以下均可
    data_test(m, n, x, y, show_ratio=80)  # show_ratio 是指显示的比例
