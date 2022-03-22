# import json
# import sys
#
# mapPath = '../../dataset/assist2009_updated/assist2009_updated_skill_mapping.txt'
# ques_map = {}
# with open(mapPath, 'r') as file:
#     for eachOne in file.readlines():
#         index = int(eachOne.split()[0])
#         title = eachOne.split()[1]
#         ques_map[index] = title
#
#
# print(ques_map)

# #判断是否安装了cuda
# import torch
# print(torch.cuda.is_available())  #返回True则说明已经安装了cuda
# #判断是否安装了cuDNN
# from torch.backends import  cudnn
# print(cudnn.is_available())  #返回True则说明已经安装了cuDNN

import os
print(os.listdir('./log/'))