import sys
sys.path.append('../')
from model.RNN import DKT
from data.dataloader import getDataLoader, getLoader
import Hyparams as params
from evaluation import eval
import torch.optim as optim
import torch

savePath = '../Models/DKT.pth'
print('DataSet: ' + params.DATASET + ', learning_rate: ' + str(params.LR) + '\n')

# model = torch.load(savePath)
model = DKT(params.INPUT, params.HIDDEN, params.LAYERS, params.OUTPUT)
# torch.save(model, savePath)
# torch.save(model.state_dict(), savePath)
# print(model.state_dict())
# sys.exit(0)

optimizer_adam = optim.Adam(model.parameters(), lr=params.LR)
optimizer_adgd = optim.Adagrad(model.parameters(), lr=params.LR)
loss_func = eval.lossFunc()
trainloaders, testloaders = getLoader(params.DATASET)

for epoch in range(params.EPOCH):
    print('\nepoch: ' + str(epoch))
    model, optimizer = eval.train(trainloaders, model, optimizer_adgd, loss_func)
    eval.test(testloaders, model)
    torch.save(model, savePath)
