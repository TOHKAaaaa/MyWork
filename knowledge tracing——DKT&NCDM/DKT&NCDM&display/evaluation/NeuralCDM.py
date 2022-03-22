import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
import os
from sklearn.metrics import roc_auc_score
sys.path.append('..')
import Hyparams as params
from data.NeuralCD_dataLoader import TrainDataLoader, ValTestDataLoader
from model.NeuralCDM import NeuralCDM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    dataloader =TrainDataLoader()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = NeuralCDM(params.student_num, params.exercise_num, params.knowledge_num)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    validate(model, 0)

    print('Training...')
    print('Device:', torch.cuda.get_device_name(0), torch.cuda.current_device(), 'count:', torch.cuda.device_count())
    loss_function = nn.NLLLoss()
    for epoch in range(params.epoch_num):
        dataloader.reset()
        running_loss = 0.0
        batch_count = 0
        while not dataloader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = dataloader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels =input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = model(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            # output_0 = torch.ones(output_1.size()) - output_1

            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output), labels)
            loss.backward()
            optimizer.step()
            model.apply_clipper()

            running_loss += loss.item()

            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 100))
                running_loss = 0.0

        validate(model, epoch)
    return model

def validate(model, epoch):
    dataloader = ValTestDataLoader('validation')
    # print('Validating...')
    dataloader.reset()
    model.eval()
    correct = 0
    exercise = 0
    batch_count = 0
    batch_avg_loss = 0
    pred = []
    label = []
    with torch.no_grad():
        while not dataloader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = dataloader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            output = model(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output = output.view(-1)
            for i in range(len(labels)):
                if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                    correct += 1
            exercise += len(labels)
            pred += output.to(torch.device('cpu')).tolist()
            label += labels.to(torch.device('cpu')).tolist()
        pred = np.array(pred)
        label = np.array(label)
        acc = correct / exercise
        rmse = np.sqrt(np.mean((label - pred) ** 2))
        auc = roc_auc_score(label, pred)
        print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, acc, rmse, auc))

        # index = str(len(os.listdir('../log/')))
        with open('../log/NeuralCDM_val.txt', 'a', encoding='utf8') as f:
            f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch + 1, acc, rmse, auc))

def test(model):
    dataloader = ValTestDataLoader('test')
    print('Testing...')
    dataloader.reset()
    model.eval()
    correct = 0
    exercise = 0
    pred = []
    label = []
    with torch.no_grad():
        while not dataloader.is_end():
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = dataloader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            output = model(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output = output.view(-1)
            # print(labels, output)
            for i in range(len(labels)):
                # print(labels[i], output[i])
                if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                    correct += 1
            exercise += len(labels)
            pred += output.tolist()
            label += labels.tolist()
        pred = np.array(pred)
        label = np.array(label)

        # print(correct, exercise)
        acc = correct / exercise
        rmse = np.sqrt(np.mean((pred - label) ** 2))
        auc = roc_auc_score(label, pred)
        print('accuracy= %f, rmse= %f, auc= %f' % (acc, rmse, auc))
        with open('../log/NeuralCDM_test.txt', 'a', encoding='utf8') as f:
            f.write('accuracy= %f, rmse= %f, auc= %f\n' % (acc, rmse, auc))

def saveModel(model, filename):
    f = open(filename, 'wb')
    torch.save(model, filename)
    f.close()

def loadModel(filename):
    f = open(filename, 'rb')
    model = torch.load(f)
    f.close()
    return model

if __name__ =='__main__':
    model = train()
    saveModel(model, '../Models/NeuralCDM1.pth')
    model = loadModel('../Models/NeuralCDM1-0.752583.pth')
    test(model)

