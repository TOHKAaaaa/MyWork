import json

# 读取数据集并转换成需要的样子
min_log = 5

def data_change():
    # problem = []
    problem_activity = []
    # with open('../Task2_data_0804/Task2_data_0804/problem_act_train.json',encoding='utf-8') as f:
    #     for line in f:
    #         item = json.loads(line)
    #         problem.append(item)
    # with open('../Task2_data_0804/Task2_data_0804/problem_act_train_2.json',encoding='utf-8') as f:
    #     for line in f:
    #         item = json.loads(line)
    #         problem.append(item)
    with open('../Task2_data_0804/Task2_data_0804/problem_activity.json',encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            problem_activity.append(item)
    stu_i = 0
    all_data = sorted(problem_activity, key=lambda x: int(x['student_id'][2:]))
    # print(all_data[0])
    # print(type(all_data))
    # print(len(problem))
    # print(len(problem_activity))
    # print(len(all_data))
    problem2id = {}  # 储存问题id
    idx = 1
    knowledgepoints = []
    with open('../Task2_data_0804/Task2_data_0804/problem_info.json',encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            problem2id[item['problem_id']] = idx
            knowledgepoints.extend(item['concept'])
            idx += 1
    knowledgepoints = set(knowledgepoints)
    knowledgepoints2id = {}
    idy = 1
    for i in knowledgepoints:
        knowledgepoints2id[i] = idy
        idy += 1
    student2id = {}  # 存储学生id
    idz = 1
    temp = []
    train_data = []  # 训练数据
    print(type(train_data))
    for i in range(len(all_data) - 1):
        temp.append(all_data[i])
        if all_data[i + 1]['student_id'] != all_data[i]['student_id']:  # 判定单个学生答题序列是否终止
            if len(temp) >= min_log:  # 答题序列长度低于min_log的数据过滤
                student_info = {}
                problem_info_list = []
                problem_info = {}
                student_info['student_id'] = idz
                student_info['problem_num'] = len(temp)
                for j in range(len(temp)):
                    concept = []
                    for k in range(len(temp[j]['concept'])):
                        concept.append(knowledgepoints2id[temp[j]['concept'][k]])
                    # concept.extend(temp[j]['concept'])
                    problem_info['problem_id'] = problem2id[temp[j]['problem_id']]
                    problem_info['label'] = temp[j]['label']
                    problem_info['concept'] = concept
                    problem_info_list.append(problem_info)
                student_info['problem'] = problem_info_list
                # 记录学生id字典
                student2id[all_data[i]['student_id']] = idz
                idz += 1
                train_data.append(student_info)
            temp = []
    print("train example:",train_data[0])
    # print(temp[-1])
    print("训练集长度为:",len(train_data))
    with open('./data/all_data.json','w',encoding='utf8') as f:
        json.dump(train_data,f,indent=2,ensure_ascii=False)
    # knowledgepoints2id problem2id student2id
    with open('./data/problem2id.txt','w',encoding='utf8') as f:
        f.write(str(problem2id))
    with open('./data/knowledgepoints2id.txt','w',encoding='utf8') as f:
        f.write(str(knowledgepoints2id))
    with open('./data/student2id.txt','w',encoding='utf8') as f:
        f.write(str(student2id))
    with open('./data/config.txt','w',encoding='utf8') as f:
        f.write(str(len(student2id)))
        f.write(",")
        f.write(str(len(problem2id)))
        f.write(",")
        f.write(str(len(knowledgepoints2id)))
    print("save successfully!")

if __name__ == '__main__':
    data_change()
