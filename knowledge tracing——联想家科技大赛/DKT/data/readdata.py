import numpy as np
import itertools
import tqdm
from data.DataSet import DataSet

class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getData(self):
        ques = np.array([])
        ans = np.array([])
        with open(self.path, 'r') as data:
            for seq_len, seq_ques, seq_ans in tqdm.tqdm(itertools.zip_longest(*[data]*3), desc = 'loading data: '):
                seq_len = int(seq_len.strip().strip(','))
                seq_ques = np.array(seq_ques.strip().strip(',').split(',')).astype(np.int)
                seq_ans = np.array(seq_ans.strip().strip(',').split(',')).astype(np.int)
                pad = 0 if seq_len % self.maxstep == 0 else (self.maxstep - seq_len % self.maxstep)
                zeros = np.zeros(pad) - 1
                seq_ques = np.append(seq_ques, zeros)
                seq_ans = np.append(seq_ans, zeros)
                ques = np.append(ques, seq_ques).astype(np.int)
                ans = np.append(ans, seq_ans).astype(np.int)
        return ques.reshape([-1, self.maxstep]), ans.reshape([-1, self.maxstep])