import torch
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from TrainDataLoader import ValTestDataLoader
from NCDM import NeuralCDM

device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
# get from config.txt
problem_n = 840
knowledge_n = 74
student_n = 5728


def test(epoch):
    device = torch.device('cpu')
    data_loader = ValTestDataLoader('test')
    net = NeuralCDM(student_n, problem_n, knowledge_n)
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, 'model/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
        out_put = out_put.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and out_put[i] > 0.5) or (labels[i] == 0 and out_put[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += out_put.tolist()
        label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
    with open('result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch, accuracy, rmse, auc))

def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()

def get_status():
    net = NeuralCDM()
    load_snapshot(net, 'model/model_epoch12')       # load model
    net.eval()
    with open('result/student_stat.txt', 'w', encoding='utf8') as output_file:
        for stu_id in range(student_n):
            # get knowledge status of student with stu_id (index)
            status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
            output_file.write(str(status) + '\n')

def get_exer_params():
    net = NeuralCDM()
    load_snapshot(net, 'model/model_epoch12')    # load model
    net.eval()
    exer_params_dict = {}
    for exer_id in range(problem_n):
        # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
        k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id]))
        exer_params_dict[exer_id + 1] = (k_difficulty.tolist()[0], e_discrimination.tolist()[0])
    with open('result/exer_params.txt', 'w', encoding='utf8') as o_f:
        o_f.write(str(exer_params_dict))

if __name__ == '__main__':
    device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
    if (len(sys.argv) != 2) or (not sys.argv[1].isdigit()):
        print('please run the predict model in Terminal\nexample:\n\tpython predict.py 70')
        exit(1)

    with open('./data/config.txt') as i_f:
        student_n, problem_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    test(int(sys.argv[1]))
