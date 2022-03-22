import cv2
import numpy as np
from numpy import histogram
from sklearn.metrics import confusion_matrix, classification_report
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

array_of_img = []
# 图片类别
array_of_label = []
# data下具体类别
labels = ["apple", "banana", "durian", "grape", "orange", "pineapple", "pomegranate", "strawberry"]
# 将所有图片都变成同样大小，不足的用黑边填充，过大的缩放
# ref:
# https://blog.csdn.net/weixin_45885074/article/details/113656535

img_size = 128

data_dir = './data'

# 读取文件，将图像全部转为灰度图像，并缩放成img_size*img_size
def read_img(filename):
    for label in labels:
        count = 0
        path = os.path.join(filename, label)
        print("start reading {}...".format(label))
        for imgname in os.listdir(path):
            img = cv2.imread(path + "/" + imgname)
            resized_img = cv2.resize(img, (img_size, img_size))
            gray = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints = sift.detect(gray)
            _, desc = sift.compute(gray, keypoints)
            sample = np.mean(desc,axis=0)
            array_of_img.append(sample)
            array_of_label.append(label)
            count += 1
        print("reading success!")
        print("{} dataset size is {}".format(label, count))


read_img(data_dir)

# 打乱数据集
index = [i for i in range(len(array_of_img))]
np.random.shuffle(index)
array_of_img = np.array(array_of_img)
array_of_label = np.array(array_of_label)
array_of_img = array_of_img[index]
array_of_label = array_of_label[index]

# 数据集类别转换
def changeLabel(array_of_label):
    print("strat changing...")
    for Y_label in range(len(array_of_label)):
        if array_of_label[Y_label] == 'apple':
            array_of_label[Y_label] = 0
        elif array_of_label[Y_label] == 'strawberry':
            array_of_label[Y_label] = 1
        elif array_of_label[Y_label] == 'orange':
            array_of_label[Y_label] = 2
        elif array_of_label[Y_label] == 'banana':
            array_of_label[Y_label] = 3
        elif array_of_label[Y_label] == 'durian':
            array_of_label[Y_label] = 4
        elif array_of_label[Y_label] == 'pineapple':
            array_of_label[Y_label] = 5
        elif array_of_label[Y_label] == 'pomegranate':
            array_of_label[Y_label] = 6
        elif array_of_label[Y_label] == 'grape':
            array_of_label[Y_label] = 7
        else:
            print("unknow label!")
    print("change success")
changeLabel(array_of_label)

array_of_img = np.array(array_of_img)
array_of_label = np.array(array_of_label)
print(array_of_img.shape,array_of_label.shape)
print(array_of_img[0][0])

array_of_img_train, array_of_img_test, array_of_label_train, array_of_label_test = train_test_split(array_of_img,
                                                                                                    array_of_label,
                                                                                                    test_size=0.3,
                                                                                                    random_state=42)

model = SVC(kernel="poly",degree=2,gamma="auto",probability=True)
model.fit(array_of_img_train,array_of_label_train)
predictions1 = model.predict(array_of_img_test)
print(confusion_matrix(array_of_label_test, predictions1))
print (classification_report(array_of_label_test, predictions1))

with open('./SVC-SIFT.pkl', 'wb') as f:
    pickle.dump(model, f)