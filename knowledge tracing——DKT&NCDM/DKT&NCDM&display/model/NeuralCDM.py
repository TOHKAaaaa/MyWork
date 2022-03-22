import torch
import torch.nn as nn
import sys

class NeuralCDM(nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num):
        super(NeuralCDM, self).__init__()

        self.knowledge_dim = knowledge_num
        self.exercise_dim = exercise_num
        self.emb_dim = student_num

        self.fc_input_dim = self.knowledge_dim
        self.fc_dim1 = 512
        self.fc_dim2 = 256

        self.stu_emb = nn.Embedding(self.emb_dim, self.knowledge_dim)
        self.k_diff = nn.Embedding(self.exercise_dim, self.knowledge_dim)
        self.e_disc = nn.Embedding(self.exercise_dim, 1)
        self.fc1 = nn.Linear(self.fc_input_dim, self.fc_dim1)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(self.fc_dim2, 1)

        # print(self.fc1.weight)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        # print(self.fc1.weight)
        # sys.exit(0)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor, 一个或多个stu_id
        :param exer_id:  LongTensor, 多个exer_id
        :param kn_emb:  FloatTensor, 每一个exer_id对应的知识点相关性的，每个元素为[1，knowledge_num]，值为1表示该试题包含该知识点
        :return:
        '''
        # print(stu_id.shape)
        # print(exer_id.shape)
        # print(kn_emb.shape)
        h_s = torch.sigmoid(self.stu_emb(stu_id))
        h_diff = torch.sigmoid(self.k_diff(exer_id))
        h_disc = torch.sigmoid(self.e_disc(exer_id)) * 10

        input_x = h_disc * (h_s - h_diff) * kn_emb
        # print(input_x.shape)
        input_x = self.drop1(torch.sigmoid(self.fc1(input_x)))
        # print(input_x.shape)
        input_x = self.drop2(torch.sigmoid(self.fc2(input_x)))
        # print(input_x.shape)
        output = torch.sigmoid(self.fc3(input_x))
        # sys.exit(0)

        return output # 一个scalar表达score

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.fc1.apply(clipper)
        self.fc2.apply(clipper)
        self.fc3.apply(clipper)



class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add(a)