import json
import torch
import sys
sys.path.append('..')
import Hyparams as params

class TrainDataLoader(object):
    def __init__(self):
        self.batch_size = params.NEURALCD_BATCHSIZE
        self.ptr = 0
        self.data = []
        data_path = '../dataset/NeuralCDM-assist2009/train_set.json'
        with open(data_path, encoding='utf-8') as file:
            self.data = json.load(file)
        self.knowledge_dim = params.knowledge_num

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for index in range(self.batch_size):
            log = self.data[self.ptr + index]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowledge_embs.append(knowledge_emb)
            ys.append(y)
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.LongTensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size >len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

class ValTestDataLoader(object):
    def __init__(self, type='validation'):
        self.ptr = 0
        self.data = []
        self.type = type

        if self.type == 'validation':
            data_path = '../dataset/NeuralCDM-assist2009/val_set.json'
        else:
            data_path = '../dataset/NeuralCDM-assist2009/test_set.json'
        with open(data_path, encoding='utf-8') as file:
            self.data = json.load(file)
        self.knowledge_dim = params.knowledge_num

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_exer_ids.append(log['exer_id'] - 1)
            input_stu_ids.append(user_id - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.LongTensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + 1 > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
