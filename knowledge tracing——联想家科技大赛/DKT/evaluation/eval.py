import sys
sys.path.append('..')
import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
import Hyparams as params

def performance(gt, pred):
    fpr, tpr, thresholds = metrics.roc_curve(gt.detach().numpy(), pred.detach().numpy())
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(gt.detach().numpy(), torch.round(pred).detach().numpy())
    recall = metrics.recall_score(gt.detach().numpy(), torch.round(pred).detach().numpy())
    precision = metrics.precision_score(gt.detach().numpy(), torch.round(pred).detach().numpy())
    acc = metrics.accuracy_score(gt.detach().numpy(), torch.round(pred).detach().numpy())
    print('auc:' + str(auc) + '\nf1:' + str(f1) + '\nrecall:' + str(recall) + '\nprecision:' + str(precision) + '\nacc:' + str(acc))

class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    def forward(self, pred, batch):
        loss = torch.Tensor([0.0])
        for student in range(pred.shape[0]):
            delta = batch[student][:, :params.NUM_OF_QUESTIONS] + batch[student][:, params.NUM_OF_QUESTIONS:]
            temp = pred[student][:params.MAX_STEP - 1].mm(delta[1:].t()) # n x n 每次step做对概率
            index = torch.LongTensor([[i for i in range(params.MAX_STEP - 1)]])
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, :params.NUM_OF_QUESTIONS] - batch[student][:, params.NUM_OF_QUESTIONS:]).sum(1) + 1) // 2)[1:]
            for i in range(len(p)):
                if p[i] > 0:
                    loss -= (a[i] * torch.log(p[i]) + (1 - a[i]) * torch.log(1 - p[i]))
        return loss


def train_epoch(model, trainLoader, optimizer, loss_func):
    for batch in tqdm.tqdm(trainLoader, desc='Training...'):
        pred = model(batch)
        loss = loss_func(pred, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, optimizer

def test_epoch(model, testLoader):
    gt_epoch = torch.Tensor([])
    pred_epoch = torch.Tensor([])
    for batch in tqdm.tqdm(testLoader, desc='Testing...'):
        pred = model(batch) #pred是预测t+1时刻每道题做对的概率

        for student in range(pred.shape[0]):
            temp_pred = torch.Tensor([])
            temp_gt = torch.Tensor([])
            delta = batch[student][:, :params.NUM_OF_QUESTIONS] + batch[student][:, params.NUM_OF_QUESTIONS:]  # 表示写了哪一题的onehot
            temp = pred[student][:params.MAX_STEP-1].mm(delta[1:].t()) # n x n的矩阵，正对角线为接下来每一步做对的概率
            index = torch.LongTensor([[i for i in range(params.MAX_STEP - 1)]])
            p = temp.gather(0, index)[0]
            # 真实答题情况，因为是预测下一个时间点的结果，因此从1开始
            a = (((batch[student][:, :params.NUM_OF_QUESTIONS] - batch[student][:, params.NUM_OF_QUESTIONS:]).sum(1) + 1) // 2)[1:]

            for i in range(len(p)): #先前有对step padding 0，因此用0区别出padding，虽然有极小可能性出现预测结果为0的情况但对模型整体并没有太大影响
                if p[i] > 0:

                    temp_pred = torch.cat([temp_pred, p[i:i+1]])
                    temp_gt = torch.cat([temp_gt, a[i:i+1]])
            pred_epoch = torch.cat([pred_epoch, temp_pred])
            gt_epoch = torch.cat([gt_epoch, temp_gt])
    return pred_epoch, gt_epoch



def test(testLoaders, model):
    gt = torch.Tensor([])
    pred = torch.Tensor([])
    for i in range(len(testLoaders)):
        pred_epoch, gt_epoch = test_epoch(model, testLoaders[i])
        pred = torch.cat([pred, pred_epoch])
        gt = torch.cat([gt, gt_epoch])
    performance(gt, pred)

def train(trainLoaders, model, optimizer, lossFunc):
    for i in range(len(trainLoaders)):
        model, optimizer = train_epoch(model, trainLoaders[i], optimizer, lossFunc)
    return model, optimizer