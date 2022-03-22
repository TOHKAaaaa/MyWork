import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from TrainDataLoader import TrainDataLoader, ValTestDataLoader
from NCDM import NeuralCDM

# 超参
# device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
# epoch_n = 5
learning_rate = 0.002

# f.write(str(len(student2id)))
# f.write(str(len(problem2id)))
# f.write(str(len(knowledgepoints2id)))
# 5728,840,74
problem_n = 840
knowledge_n = 74
student_n = 5728

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = TrainDataLoader()
    net = NeuralCDM(student_n,problem_n,knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print("training model...")

    loss_function = nn.NLLLoss()
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss =0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_problem_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_problem_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_problem_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
            output_1 = net.forward(input_stu_ids,input_problem_ids,input_knowledge_embs)
            # output_0 = torch.ones(output_1.size()).to(device) - output_1
            output_0 = torch.ones(output_1.size()) - output_1
            output = torch.cat((output_0,output_1),1)

            loss = loss_function(torch.log(output), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            rmse, auc = validate(net, epoch)
            # print('epoch:%d, loss: %.3f,rmse:%.3f,auc:%.3f' % (epoch + 1, running_loss, rmse, auc))
            if batch_count % 200 == 199:
                print('epoch:%d, loss: %.3f,rmse:%.3f,auc:%.3f' % (epoch + 1, running_loss / 200, rmse, auc))
                running_loss = 0.0
            validate(net,epoch)
            save_snapshot(net, 'model/model_epoch' + str(epoch + 1))

def validate(model, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = ValTestDataLoader('validation')
    net = NeuralCDM(student_n,problem_n,knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, problem_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_problem_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_problem_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_problem_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        # print("test:",input_stu_ids)
        # print("test:",input_problem_ids)
        output = net.forward(input_stu_ids, input_problem_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        problem_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / problem_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('please run the train model in Terminal\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    # global student_n, exer_n, knowledge_n, device
    with open('./data/config.txt') as i_f:
        student_n, problem_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    train()