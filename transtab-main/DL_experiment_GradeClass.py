import transtab
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transtab.evaluator import predict_multi_task
from Result_process.metrics_process import metrics_with_youden, metrics_multiclass
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np
import itertools
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer, TabTransformer
import torch.optim as optim
info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

df_PHLF_valid_grade = pd.read_csv('../data_process/df_PHLF_valid_grade.csv')
df_PHLF_extvalid_grade = pd.read_csv('../data_process/df_PHLF_extvalid_grade.csv')
df_PHLF_train_with_grade = pd.read_csv('../data_process/df_PHLF_train_with_grade.csv')

train_data, valid_data = train_test_split(df_PHLF_train_with_grade, test_size=0.2, random_state=42)

train_set = [train_data.iloc[:,2:], [train_data.iloc[:,1], train_data.iloc[:,0]]]
valid_set = [valid_data.iloc[:,2:], [valid_data.iloc[:,1], valid_data.iloc[:,0]]]

param_grid = {
    'batch_size': [8, 16, 32,64],
    'lr': [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, ],
    'weight_decay': [1e-4, 1e-5, 1e-3],
    'eval_batch_size': [64, 128, 256],
    'imb_weight' :[1.5, 2, 2.5, 3., 3.5,]
}

training_arguments = {
    'batch_size':8,
    'lr':0.0005,
    'weight_decay':1e-05,
    'patience' :5,
    'eval_batch_size':128,
    'num_epoch' :200,
    'imb_weight': 2
}



param_combinations = list(itertools.product(*param_grid.values()))
def evaluate_params(params):
    params_dict = dict(zip(param_grid.keys(), params))
    training_arguments.update(params_dict)


    model = transtab.build_classifier_multi_task(cate_cols, num_cols, bin_cols, imb_weight=training_arguments['imb_weight'])
    model = model.to('cuda')
    transtab.train_multi_task(model, train_set, valid_set, **training_arguments)
    print(params)

    y_test = pd.concat([df_PHLF_valid_grade.iloc[:, 1], df_PHLF_valid_grade.iloc[:, 0]], axis=1)
    prob_int_parent, prob_int_child = predict_multi_task(model, df_PHLF_valid_grade.iloc[:, 2:], y_test)

    result_int_parent, parent_con = metrics_with_youden(df_PHLF_valid_grade.iloc[:, 1], prob_int_parent[:, 1])
    print('\n')
    result_int_child, child_con = metrics_multiclass(df_PHLF_valid_grade.iloc[:, 0], prob_int_child)

    return result_int_parent, result_int_child,parent_con, child_con, model

PATH = 'LF_bio_250220_best_0'
best_params = None
best_score = -float('inf')  # 我们是在最大化某个指标
count = 0
for params in param_combinations:
    result_int_parent, result_int_child, parent_con, child_con,model = evaluate_params(params)
    score = (float(result_int_parent[0])+float(result_int_child[0]))/2
    if score > best_score:
        best_score = score
        best_params = params
        best_model = model

        model.save('./checkpoint/'+PATH + str(count))
        count += 1
        print('./checkpoint/'+PATH + str(count))
        with open('./results/'+PATH+'/parameters.txt', 'a') as f:
            f.write(str(best_params) + '\n' +
                    str(parent_con) + '\n' +
                    str(result_int_parent) + '\n'+
                    str(child_con)+'\n'+
                    str(result_int_child) + '\n' +
                    '\n')
        # torch.save(best_model, './checkpoint/LF_bio_240613_best.pth')
        print(best_params, '\n',
              '**************************************************************************************')
print("Best parameters found:", best_params,'\n', "Best score:",best_score)

# model = transtab.build_classifier_multi_task(cate_cols, num_cols, bin_cols)
# model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+PATH)







#
#
# model.save('./checkpoint/'+PATH)
#
# best_params = None
# best_score = -float('inf')  # 我们是在最大化某个指标
# count = 0
# for params in param_combinations:
#     score, model, result_int, result_ext12, result_ext3 = evaluate_params(params)
#
#     if score > best_score:
#         best_score = score
#         best_params = params
#         best_model = model
#
#         model.save('./checkpoint/'+PATH + str(count))
#         count += 1
#         print('./checkpoint/'+PATH + str(count))
#         with open('./results/'+PATH+'.txt', 'a') as f:
#             f.write(str(best_params) + '\n' +
#                     str(result_int) + '\n' +
#                     str(result_ext12) + '\n'+
#                     str(result_ext3)+'\n'+'\n')
#         # torch.save(best_model, './checkpoint/LF_bio_240613_best.pth')
#         print(best_params, '\n',
#               '**************************************************************************************')
# print("Best parameters found:", best_params,'\n', "Best score:",best_score)


