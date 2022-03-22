from flask import Flask, render_template, request, session, g
import torch
import json
import Hyparams as params
from flask_cors import CORS, cross_origin


app = Flask(__name__)

mapPath = 'E:\\winR\\college\\专业课\\大三上\\认知\\dataset\\DKT\\data\\assist2009_updated_skill_mapping.txt'
ques_map = {}
with open(mapPath, 'r') as file:
    for eachOne in file.readlines():
        index = int(eachOne.split()[0])
        title = eachOne.split()[1]
        ques_map[index] = title

info = {'steps_ans': torch.LongTensor([])}

modelPath = './Models/DKT2-0.05-32.pth'
model = torch.load(modelPath)
model.eval()
ques_index = 0

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def jsonData(pred):
    index2Name = {
        7: 'Mode',
        8: 'Mean',
        15: 'Fraction of',
        92: 'Rotations',
        59: 'Exponents',
        50: 'Pythagorean Theorem'
    }
    returnList = []
    for step in range(pred.shape[0]):
        dic = {}
        dic['time'] = step + 1
        for k, v in index2Name.items():
            dic[v] = float(pred[step][k - 1])
        returnList.append(dic)
    return {'data': returnList}

@app.route('/predict', methods=['POST', 'GET'])
def predict():

    # result = request.form.get('result')
    # print(request.method)
    if request.method == 'GET':
        # print(7777)
        ques_index = int(request.args.get('quesIndex'))
        result = request.args.get('result')
        # print(ques_index, result)
        onehot = torch.zeros([params.NUM_OF_QUESTIONS * 2])

        if result == 'true':
            print(ques_index, 'True')
            onehot[ques_index] = 1
        else:
            print(ques_index, 'False')
            onehot[ques_index + params.NUM_OF_QUESTIONS] = 1

        info['steps_ans'] = torch.cat([info['steps_ans'].squeeze(0), onehot.reshape( 1, params.NUM_OF_QUESTIONS * 2)]).unsqueeze(0)

        # print(info['steps_ans'], info['steps_ans'].shape)
        pred = model(info['steps_ans'])
        # print(pred, pred.shape)
        # print(pred[0][-1])
    # return render_template('index.html', index=ques_index, result=result)
    return jsonData(pred[0])


if __name__ == '__main__':
    app.run(debug=True) #刷新即可