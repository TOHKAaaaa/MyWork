#   DKT
Datapath = "E:\winR\college\专业课\大三上\认知\dataset\DKT\data"

datasets = {
    'assist2009' : 'assist2009',
    'assist2015' : 'assist2015',
    'assist2017' : 'assist2017',
    'static2011' : 'static2011',
    'kddcup2010' : 'kddcup2010',
    'synthetic' : 'synthetic',
    'assist2009_updated' : 'assist2009_updated'
}

# 各个数据集的问题数
numbers = {
    'assist2009' : 124,
    'assist2015' : 100,
    'assist2017' : 102,
    'static2011' : 1224,
    'kddcup2010' : 661,
    'synthetic' : 50,
    'assist2009_updated' : 110,
}

DATASET = datasets['assist2009_updated']
NUM_OF_QUESTIONS = numbers['assist2009_updated']

# RNN的最大步长
MAX_STEP = 50
BATCH_SIZE = 32
# LR 0.002
LR = 0.002
EPOCH = 50
#input dimension
INPUT = NUM_OF_QUESTIONS * 2
# embedding dimension
EMBED = NUM_OF_QUESTIONS
# hidden layer dimension
HIDDEN = 200
# nums of hidden layers
LAYERS = 1
# output dimension
OUTPUT = NUM_OF_QUESTIONS
#   DKT

#   NeuralCDM

exercise_num = 17746
knowledge_num = 123
student_num = 4163

epoch_num = 2
NEURALCD_BATCHSIZE = 32


#   NeuralCDM
