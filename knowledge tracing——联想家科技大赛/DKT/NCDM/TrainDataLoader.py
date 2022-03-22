import json
import torch

class TrainDataLoader(object):
    def __init__(self):
        self.batch_size = 32
        self.ptr = 0
        self.data = []

        data_dir = './data/train_set.json'
        config_file ='./data/config.txt'

        with open(data_dir,encoding='utf8') as f:
            self.data = json.load(f)
        with open(config_file,encoding='utf8') as f:
            _,_,knowledge_n = f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def next_batch(self):
        if self.is_end():
            return None,None,None,None
        input_stu_ids, input_problem_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            problem = self.data[self.ptr+count]
            knowledge_emb = [0.] * self.knowledge_dim
            for concept in problem['concept']:
                knowledge_emb[concept-1] = 1.0
            y = problem['label']
            input_stu_ids.append(problem['student_id']-1)
            input_problem_ids.append(problem['problem_id']-1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids),torch.LongTensor(input_problem_ids),torch.Tensor(input_knowedge_embs),torch.LongTensor(ys)

class ValTestDataLoader(object):
    def __init__(self,d_type='validation'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if self.d_type == 'validation':
            data_dir = './data/val_set.json'
        else:
            data_dir = './data/test_set.json'
        config_dir = './data/config.txt'
        with open(data_dir,encoding='utf8') as f:
            self.data = json.load(f)
        with open(config_dir,encoding='utf8') as f:
            _,_, knowledge_n = f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def next_batch(self):
        if self.is_end():
            return None,None,None,None
        problems = self.data[self.ptr]['problem']
        student_id = self.data[self.ptr]['student_id']
        input_stu_ids, input_problem_ids, input_knowedge_embs, ys = [], [], [], []
        # print("我是学生id:",student_id)
        # print("i am problem:",problems)
        # print(self.knowledge_dim)
        for problem in problems:
            input_stu_ids.append(student_id - 1)
            input_problem_ids.append(problem['problem_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            for concept in problem['concept']:
                knowledge_emb[concept - 1] = 1.0
            input_knowedge_embs.append(knowledge_emb)
            y = problem['label']
            ys.append(y)
        print(input_stu_ids)
        print(input_problem_ids)
        print(input_knowedge_embs)
        print(torch.LongTensor(input_stu_ids))
        print(torch.LongTensor(input_problem_ids))
        print(torch.Tensor(input_knowedge_embs))

        self.ptr += 1
        return torch.LongTensor(input_stu_ids),torch.LongTensor(input_problem_ids),torch.Tensor(input_knowedge_embs),torch.LongTensor(ys)