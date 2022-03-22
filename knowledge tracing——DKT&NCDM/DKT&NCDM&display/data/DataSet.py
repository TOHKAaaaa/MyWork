import numpy as np
from torch.utils.data.dataset import Dataset
import sys
sys.path.append('..')
import Hyparams as params
import torch


class DataSet(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        onehot = self.onehot(questions, answers)
        return torch.FloatTensor(onehot.tolist())

    def onehot(self, questions, answers): #每个time step时各个题目的掌握程度
        result = np.zeros(shape=[params.MAX_STEP, 2 * params.NUM_OF_QUESTIONS])
        # print(result, result.shape)
        # print(questions)
        for i in range(params.MAX_STEP):
            if answers[i] > 0:
                result[i][questions[i]] = 1
            elif answers[i] == 0:
                # print(questions[i])
                # print(params.NUM_OF_QUESTIONS)
                result[i][questions[i] - 1 + params.NUM_OF_QUESTIONS] = 1
        return result

