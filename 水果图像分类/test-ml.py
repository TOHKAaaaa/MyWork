import os
import torch
import pickle
import easygui
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


model = pickle.load(open('SVC-SIFT.pkl','rb'))
# model.eval()
print("load finished!")

def showImg(img,label,name):
    plt.figure(figsize = (4,4),num = name)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
    plt.show()

# 绘制混淆矩阵
def plot_confusion(cm,f1,acc):
    plt.figure()
    plot_confusion_matrix(cm,figsize = (12,8),cmap = plt.cm.Blues)
    plt.xticks(range(8), ['Apple','Strawberry','Orange','Banana','Durian','Pineapple','Pomegranate','Grape'],fontsize = 10)
    plt.yticks(range(8), ['Apple','Strawberry','Orange','Banana','Durian','Pineapple','Pomegranate','Grape'],fontsize = 14)
    # plt.xticks(range(2),['predict','actual'],fontsize = 14)
    # plt.yticks(range(2),['predict','actual'],fontsize = 14)
    plt.xlabel('Predicted Label',fontsize = 16)
    plt.ylabel('True Label',fontsize = 16)
    plt.title('f1-score: ' + str(f1) + ' Accurary: ' + str(acc))
    plt.show()

def accuracy(outputs, labels):
    correct = np.sum(outputs == labels) / len(labels)
    return correct

def metrics(outputs,labels):
    cm = confusion_matrix(labels,outputs)
    f1 = f1_score(labels,outputs,labels=[0,1,2,3,4,5,6,7],average='macro')
    f1 = round(f1,4)
    acc = accuracy(preLabel,sub['label'])
    plot_confusion(cm,f1,acc)
    
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 * ((precision * recall) / (precision + recall))
    # return precision, recall, f1

# 得到文件夹地址
def getDirectory():
    return easygui.diropenbox()

# 得到csv文件地址
def getFile():
    return easygui.fileopenbox()

image_transforms = {
    'valid': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
}

preLabel = []

# 开始预测
def predict():
    fruit = ['苹果','草莓','橙子','香蕉','榴莲','菠萝','石榴','葡萄']
    with torch.no_grad():
        for batch_id,(images,labels) in enumerate(dataloaders['valid']):
            fileName = (os.listdir(os.path.join(str(dataloaders['valid'].dataset.root),'test')))[batch_id]
            outputs = model(images)
            print(batch_id + 1,end = '\t')
            _,predicted = torch.max(outputs,dim = 1)
            if(batch_id < 9):
                img_name = '0' + str(batch_id + 1) + '.jpg'
            else:
                img_name = str(batch_id + 1) + '.jpg'
            img = Image.open(dataloaders['valid'].dataset.root + '\\test\\'+ img_name)
            preLabel.append(int(predicted))
            if((batch_id + 1) % 5 == 0):
                print(fruit[int(predicted)])
            else:
                print(fruit[int(predicted)],end = ' ')
            showImg(img,fruit[int(predicted)],fileName)

precisions = []
recalls = []
f1s = []
accuracies = []
def getMetrics():
    # precision,recall,f1 = 
    metrics(preLabel,sub['label'])
    # acc = accuracy(preLabel,sub['label'])
    # print(acc)
    # precisions.append(precision)
    # recalls.append(recall)
    # f1s.append(f1)
    # accuracies.append(acc.item())

while(1):
    if(easygui.buttonbox(choices=['加载测试图像文件夹','加载csv文件','关闭程序']) == '加载测试图像文件夹'):
        img_path = getDirectory()
        datasets = {
            'valid': datasets.ImageFolder(img_path,transform = image_transforms['valid'])   
        }
        dataloaders = {
            'valid': DataLoader(datasets['valid'])
        }
        predict()
    elif(easygui.buttonbox(choices=['加载测试图像文件夹','加载csv文件','关闭程序']) == '加载csv文件'):
        file_path = getFile()
        sub = pd.read_csv(file_path)
        df_sub = pd.DataFrame()
        df_sub['id'] = sub['id']
        df_sub['label'] = preLabel
        df_sub.to_csv('myValid.csv',index = False)
        getMetrics()
    else:
        break