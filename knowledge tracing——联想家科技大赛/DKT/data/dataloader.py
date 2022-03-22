import sys
sys.path.append('../')
import torch
import torch.utils.data as Data
import Hyparams as params
from data.DataSet import DataSet
from data.readdata import DataReader

def getDataLoader(path):
    handle = DataReader(path, params.MAX_STEP, params.NUM_OF_QUESTIONS)
    ques, ans = handle.getData()
    data = DataSet(ques, ans)
    dataloader = Data.DataLoader(data, batch_size=params.BATCH_SIZE, shuffle=True)
    return dataloader

def getLoader(dataset):
    trainloader = []
    testloader = []
    root = params.Datapath
    if dataset == 'assist2009':
        trainloader.append(getDataLoader(root + '/builder_train.csv'))
        testloader.append(getDataLoader(root + '/builder_test.csv'))
    elif dataset == 'assist2009_updated':
        trainloader.append(getDataLoader("E:\winR\college\专业课\大三上\认知\dataset\DKT\data\ssist2009_updated_train.csv"))
        testloader.append(getDataLoader("E:\winR\college\专业课\大三上\认知\dataset\DKT\data\ssist2009_updated_test.csv"))
    # elif dataset == 'assist2015':

    return trainloader, testloader