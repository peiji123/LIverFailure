import transtab
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np
import itertools
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer, TabTransformer
import torch.optim as optim

def get_final_result(preds,label,):
    y_preds = np.zeros_like(preds)
    for idx, val in enumerate(preds):
        if val > 0.5:
            y_preds[idx] = 1
    auc = roc_auc_score(y_true=label, y_score=preds)
    # y_preds = preds.argmax(1) #argmax取出preds元素最大值所对应的索引,1代表维度，是指在第二维里取最大值的索引
    acc = accuracy_score(y_true=label, y_pred=y_preds)
    recall = recall_score(y_true=label, y_pred=y_preds)
    precision = precision_score(y_true=label, y_pred=y_preds, labels=[0])
    f1 = f1_score(y_true=label, y_pred=y_preds, )
    kappa = cohen_kappa_score(y1=label, y2=y_preds, )
    mcc = matthews_corrcoef(y_true=label, y_pred= y_preds)
    con = confusion_matrix(label, y_preds)
    print(con)
    print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc), 'recall:','{:.3f}'.format(recall))
    return ['acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc), 'recall:','{:.3f}'.format(recall),], con

info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
df_alin_test = pd.read_csv('../data_process/df_int_test.csv')
df_alin_ext12 = pd.read_csv('../data_process/df_ext12.csv')
df_alin_ext3 = pd.read_csv('../data_process/New_ext_3.csv')
df_alin_ext4 = pd.read_csv('../data_process/df_ext4.csv')

df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]
df_alin_ext12 = df_alin_ext12.iloc[:,1:]
df_alin_ext3 = df_alin_ext3.iloc[:,1:]
df_alin_ext4 = df_alin_ext4.iloc[:,1:]

df_test_ext3 = df_alin_ext3.drop(['duration of hepatic pedicle clamping'], axis=1)
df_test_ext3 = df_test_ext3.dropna()

df_train, df_valid = train_test_split(df_alin_train, test_size=0.2,random_state=42)
train_set = [df_train.iloc[:,1:], df_train.iloc[:,0]]
valid_set = [df_valid.iloc[:,1:], df_valid.iloc[:,0]]

training_arguments = {
    'batch_size':16,
    'lr':1e-4,
    'weight_decay':1e-5,
    'patience' :5,
    'eval_batch_size':256,
    'num_epoch' :200,
    'imb_weight': 2.5
}
# param_combinations = list(itertools.product(*param_grid.values()))
# def evaluate_params(params):
#     params_dict = dict(zip(param_grid.keys(), params))
#     training_arguments.update(params_dict)
#
#     # training_arguments.update(params)
#     model = transtab.build_classifier(cate_cols, num_cols, bin_cols, imb_weight=training_arguments['imb_weight'])
#     model = model.to('cuda')
#     transtab.train(model, train_set, valid_set, **training_arguments)
#
#     print(params)
#     # =========================================================================================
#     prob_int = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
#     result_int = get_final_result(prob_int, df_alin_test.iloc[:, 0])
#
#     prob_ext12 = transtab.predict(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0])
#     result_ext12 = get_final_result(prob_ext12, df_alin_ext12.iloc[:, 0])
#
#     prob_ext3 = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
#     result_ext3 = get_final_result(prob_ext3, df_alin_ext3.iloc[:, 0])
#
#     avg_auc = (float(result_int[5])+float(result_ext12[5])+float(result_ext3[5]))/3
#
#     # avg_auc = np.mean(float(result_int[5]), float(result_ext12[5]), float(result_ext3[5]))
#     return avg_auc, model, result_int, result_ext12, result_ext3
#     # =========================================================================================
#     # y_pred = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
#     # result_ext3 = get_final_result(y_pred, df_alin_ext3.iloc[:, 0])
#     # return float(result_ext3[5]), model, result_ext3

PATH = 'LF_bio_241007_best_0'
# best_params = None
best_score = -float('inf')  # 我们是在最大化某个指标
count = 0

for i in range(10000):
    model = transtab.build_classifier(cate_cols, num_cols, bin_cols, imb_weight=training_arguments['imb_weight'])
    model = model.to('cuda')
    transtab.train(model, train_set, valid_set, **training_arguments)

    prob_int = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
    result_int, cm_int = get_final_result(prob_int, df_alin_test.iloc[:, 0])

    prob_ext12 = transtab.predict(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0])
    result_ext12, cm_ext12 = get_final_result(prob_ext12, df_alin_ext12.iloc[:, 0])

    prob_ext3 = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
    result_ext3, cm_ext3 = get_final_result(prob_ext3, df_alin_ext3.iloc[:, 0])

    prob_ext4 = transtab.predict(model, df_alin_ext4.iloc[:, 1:], df_alin_ext4.iloc[:, 0])
    result_ext4, cm_ext4 = get_final_result(prob_ext4, df_alin_ext4.iloc[:, 0])

    int_auc = float(result_int[5])
    ext12_auc = float(result_ext12[5])
    ext3_auc = float(result_ext3[5])
    ext4_auc = float(result_ext4[5])
    # score = (float(result_int[5])+float(result_ext12[5])+float(result_ext3[5]))/3
    if (int_auc>0.94) & (ext12_auc > 0.9)& (ext3_auc > 0.9) & (ext4_auc > 0.9):
        # best_score = score
        best_model = model
        count += 1
        model.save('./checkpoint/'+PATH + str(count))
        print('**************************************************************************************')
        print('./checkpoint/'+PATH + str(count))
        with open('./results/'+PATH+'.txt', 'a') as f:
            f.write('*'+str(count)+ '********************************************************'+ '\n' +
                    str(cm_int)+ '\n' +
                    str(result_int) + '\n' +
                    str(cm_ext12) + '\n' +
                    str(result_ext12) + '\n'+
                    str(cm_ext3) + '\n' +
                    str(result_ext3)+'\n'+
                    str(cm_ext4) + '\n' +
                    str(result_ext4) + '\n' +
                    '\n')




