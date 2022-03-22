import sys
sys.path.append('../')
from model.RNN import DKT
from data.dataloader import getDataLoader, getLoader
import Hyparams as params
from evaluation import eval
import torch.optim as optim
import torch

import warnings	
warnings.filterwarnings(action="ignore")

savePath = 'E:\\winR\\college\\专业课\\大三下\\大数据技术\\final\\DKT\\Models\\NCDM.pth'
print('DataSet: ' + params.DATASET + ', learning_rate: ' + str(params.LR) + '\n')

model = DKT(params.INPUT, params.HIDDEN, params.LAYERS, params.OUTPUT)

optimizer_adam = optim.Adam(model.parameters(), lr=params.LR)
optimizer_adgd = optim.Adagrad(model.parameters(), lr=params.LR)
loss_func = eval.lossFunc()
trainloaders, testloaders = getLoader(params.DATASET)

for epoch in range(params.EPOCH):
    print('\nepoch: ' + str(epoch))
    model, optimizer = eval.train(trainloaders, model, optimizer_adgd, loss_func)
    eval.test(testloaders, model)
    torch.save(model, savePath)
