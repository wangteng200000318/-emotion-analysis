import numpy as np
import time
import re
import random
import matplotlib.pyplot as plt

file1 = open("neg.txt", "r", encoding='UTF-8')
file2 = open("pos.txt", "r", encoding='UTF-8')
file3 = open("test_n.txt", "r", encoding='UTF-8')
file4 = open("test_p.txt", "r", encoding='UTF-8')
positive_accuracy = []
negative_accuracy = []


# 先将文本按照行分割，再将句子按按照,.!等符号进行分割，并将每个分割后的结果保存到一个list中
def pre(file):
    lst1 = []
    # The splitlines() method returns a list with all the lines in string,
    # optionally including the line breaks (if num is supplied and is true).
    # 按行分割！
    content = file.read().splitlines()

    # compile(pattern, flags=0)
    # Compile a regular expression pattern, returning a pattern object.
    # re.I忽略大小写
    for i in content:
        lst = re.compile(r"[a-z]+[a-z]+", re.I).findall(i)
        str = " ".join(lst)
        lst1.append([str])
    dataset = np.array(lst1)
    return dataset


def File_to_Vector_bool(d1, d2, d3, d4):  # 文档向量化(布尔权重)
    lst = []
    data1 = []
    data2 = []
    dataset2 = np.concatenate([d1, d2])
    dataset3 = np.concatenate([d3, d4])
    """
    dataset2 :
    [['simplistic silly and tedious']
     ['it so laddish and juvenile only teenage boys could possibly find it funny']
     ['exploitative and largely devoid of the depth or sophistication that would make watching such graphic treatment of the crimes bearable']
     ...
     ['standing in the shadows of motown is the best kind of documentary one that makes depleted yesterday feel very much like brand new tomorrow']
     ['it nice to see piscopo again after all these years and chaykin and headly are priceless']
     ['provides porthole into that noble trembling incoherence that defines us all']] 

     dataset3 :
     [['this is amusing for about three minutes']
     ['klein charming in comedies like american pie and dead on in election delivers one of the saddest action hero performances ever witnessed']
     ['it rare to see movie that takes such speedy swan dive from promising to interesting to familiar before landing squarely on stupid']
     ...
     ['informative intriguing observant often touching gives human face to what often discussed in purely abstract terms']
     ['once the true impact of the day unfolds the power of this movie is undeniable']
     ['an honest sensitive story from vietnamese point of view']]
    """
    # 把训练样本中的每个词都加入到lst中
    for i in range(len(dataset2)):
        for j in range(len(dataset2[i][0].split())):
            lst.append(dataset2[i][0].split()[j])
    # 通过调用set()再调用list()来实现去除重复项，最后通过sort()来进行排序，并将最后结果赋给dlst
    dlst = sorted(list(set(lst)))
    lenth = len(dlst)
    dic = dict(zip(dlst, [i for i in range(lenth)]))
    # 找出测试集中与训练集中公共部分
    for i in range(len(dataset3)):
        j = dataset3[i][0].split()
        lst2 = list(set(j) & set(dlst))
        str = " ".join(lst2)
        data2.append([str])
    dataset4 = np.array(data2)
    """
    训练集与测试集的交集的最后输出结果
    [['is three about for amusing minutes this']
     ['and of in comedies delivers hero action on witnessed klein ever dead the saddest one election performances like pie american charming']
     ['takes that see before interesting on swan it dive to from movie familiar landing rare such speedy stupid promising squarely']
     ...
     ['purely often what gives informative intriguing to abstract in human terms touching face observant discussed']
     ['impact power is day undeniable unfolds of true the movie once this']
     ['vietnamese sensitive of view from an point story honest']]
    """
    dataset = np.concatenate([dataset2, dataset4])  # 合并两个矩阵
    matrix = np.zeros(shape=(len(dataset), lenth))
    for i in range(len(dataset)):
        data1.append(dataset[i][0].split())

    datamat = np.array(data1)
    """
    [list(['simplistic', 'silly', 'and', 'tedious'])
     list(['it', 'so', 'laddish', 'and', 'juvenile', 'only', 'teenage', 'boys', 'could', 'possibly', 'find', 'it', 'funny'])
     list(['exploitative', 'and', 'largely', 'devoid', 'of', 'the', 'depth', 'or', 'sophistication', 'that', 'would', 'make', 'watching', 'such', 'graphic', 'treatment', 'of', 'the', 'crimes', 'bearable'])
     ...
     list(['touching', 'discussed', 'abstract', 'what', 'observant', 'intriguing', 'face', 'to', 'terms', 'gives', 'human', 'often', 'in', 'informative', 'purely'])
     list(['the', 'unfolds', 'this', 'movie', 'of', 'is', 'power', 'day', 'impact', 'true', 'once', 'undeniable'])
     list(['view', 'point', 'vietnamese', 'story', 'of', 'sensitive', 'an', 'from', 'honest'])]
     """
    for k in range(len(datamat)):
        for r in range(len(datamat[k])):
            matrix[k][dic[datamat[k][r]]] = 1
    return matrix


def ChoosebyVar(matrix, p):  # 方差选择法特征选择
    delst = []
    # 二项分布方差 = p * (1 - p)
    var = p * (1 - p)
    lst = np.var(matrix, axis=0)
    for i in range(len(lst)):
        if lst[i] <= var:
            delst.append(i)
    matrix = np.delete(matrix, delst, axis=1)
    return matrix


# data_array训练样本,
def train(data_array, y, eta, iters):
    # 返回一个以0填充的数组，数组的大小为data_array的行数
    w = np.zeros(data_array.shape[1])

    # w = np.random.uniform(-0.5, 0.5, data_array.shape[1])
    # 这样正确率太低了
    b = 0
    iter = 0
    while iter < iters:
        random_x = random.randint(0, len(data_array) - 1)  # 随机挑选一个x
        if y[random_x] * (np.dot(data_array[random_x], w) + b) <= 0:
            w += eta * y[random_x] * data_array[random_x]
            b += eta * y[random_x]
        iter += 1
    return (w, b)


def accuracy(d1, d2, d3, d4):
    start = time.time()
    len1 = len(d1)
    len2 = len(d2)
    len3 = len(d3)
    len4 = len(d4)
    count_n, count_p = 0, 0
    labels = [-1 for i in range(len(d1))] + [1 for j in range(len(d2))]

    matrix_0 = File_to_Vector_bool(d1, d2, d3, d4)
    end = time.time()
    print("文本向量化完成.......")
    print("用时：%.4f" % (end - start), "s")
    matrix_1 = ChoosebyVar(matrix_0, 1.0)
    # matrix_1 = np.loadtxt("matrix.txt", delimiter=',')
    matrix_test_n = matrix_1[len1 + len2:len1 + len2 + len3]
    matrix_test_p = matrix_1[len1 + len2 + len3:len1 + len2 + len3 + len4]
    matrix_train = matrix_1[0:len1 + len2]
    next_end = time.time()
    print("特征选择完成..........")
    print(matrix_1.shape)
    print("用时：%.4f" % (next_end - end), 's')

    (w, b) = train(matrix_train, labels, 0.00001, 13000)

    print("感知机学习完成，开始分类......")
    for i in range(len(d3)):
        if (np.dot(matrix_test_n[i], w) + b) < 0:
            count_n += 1
    for j in range(len(d4)):
        if (np.dot(matrix_test_p[j], w) + b) > 0:
            count_p += 1
    final_end = time.time()
    print("分类完成，用时：%.4f" % (final_end - next_end), "s")
    print("总用时：%.4f" % (final_end - start), "s")
    positive_accuracy.append(count_p / len(d4))
    negative_accuracy.append(count_n / len(d3))
    print("negative测试集的正确率为:%.4f" % (count_n / len(d3)))
    print("positive测试集的正确率为:%.4f" % (count_p / len(d4)))
    # return (count_n / len(d3), count_p / len(d4))


if __name__ == "__main__":
    accuracy(pre(file1), pre(file2), pre(file3), pre(file4))

# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.plot(positive_accuracy)
# plt.plot(negative_accuracy)
# plt.title("测试样本正确率图")
# plt.legend(['positive','negative'])
# plt.show()
