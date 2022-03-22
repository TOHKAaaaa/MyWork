import json
import random

# 划分数据集
def divide_data():
    with open('./data/all_data.json',encoding='utf8') as f:
        stus = json.load(f)
    train_slice, train_set, val_set, test_set = [], [], [], []
    for stu in stus:
        student_id =stu['student_id']
        stu_train = {'student_id': student_id}
        stu_val = {'student_id': student_id}
        stu_test = {'student_id': student_id}
        # train:val:test = 7:1:2
        train_size = int(stu['problem_num'] * 0.7)
        val_size =int(stu['problem_num'] * 0.1)
        test_size =stu['problem_num'] - train_size - val_size
        problems = []
        for problem in stu['problem']:
            problems.append(problem)
        random.shuffle(problems)
        # print("problem length:",len(problems))
        # 划分数据集
        stu_train['problem_num'] = train_size
        stu_train['problem'] =problems[:train_size]
        stu_val['problem_num'] = val_size
        stu_val['problem'] = problems[train_size:train_size+val_size]
        stu_test['problem_num'] = test_size
        stu_test['problem'] = problems[-test_size:]
        train_slice.append(stu_train)
        val_set.append(stu_val)
        test_set.append(stu_test)
        # 打乱train
        for problem in stu_train['problem']:
            train_set.append({'student_id':student_id,'problem_id':problem['problem_id'],'label':problem['label'],
                              'concept':problem['concept']})
        random.shuffle(train_set)

    with open('./data/train_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    with open('./data/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('./data/val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)
    with open('./data/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)
    print("save successfully!")

if __name__ == '__main__':
    divide_data()