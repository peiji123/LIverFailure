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
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc), 'recall:','{:.3f}'.format(recall),]

info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

# df_alin_train = pd.read_csv('../data_process/df_int_alin_train.csv')
# df_alin_test = pd.read_csv('../data_process/df_int_alin_test.csv')
# df_alin_ext12 = pd.read_csv('../data_process/df_ext12_alin.csv')
# df_alin_ext3 = pd.read_csv('../data_process/df_ext3_alin.csv')

df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
df_alin_test = pd.read_csv('../data_process/df_int_test.csv')
df_alin_ext12 = pd.read_csv('../data_process/df_ext12.csv')
df_alin_ext3 = pd.read_csv('../data_process/df_ext3.csv')

df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]
df_alin_ext12 = df_alin_ext12.iloc[:,1:]
df_alin_ext3 = df_alin_ext3.iloc[:,1:]

df_test_ext3 = df_alin_ext3.drop(['duration of hepatic pedicle clamping'], axis=1)
df_test_ext3 = df_test_ext3.dropna()

# df_alin_train = df_alin_train.fillna(0)
# df_alin_test = df_alin_test.fillna(0)
# df_alin_ext12 = df_alin_ext12.fillna(0)
# df_alin_ext3 = df_alin_ext3.fillna(0)

df_train, df_valid = train_test_split(df_alin_train, test_size=0.2,random_state=42)
train_set = [df_train.iloc[:,1:], df_train.iloc[:,0]]
valid_set = [df_valid.iloc[:,1:], df_valid.iloc[:,0]]

# =============================================================================
# batch_size = [16, 32, 64, 128, 256]
# lr = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
# weight_decay = [1e-4, 1e-5, 1e-3]
# patience = [3, 5, 10, 15]
# eval_batch_size = [32, 64, 128, 256]
param_grid = {
    'batch_size': [8, 16, 32,64],
    'lr': [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, ],
    'weight_decay': [1e-4, 1e-5, 1e-3],
    'eval_batch_size': [64, 128, 256],
    'imb_weight' :[1.5, 2, 2.5, 3., 3.5,]
}
training_arguments = {
    'batch_size':64,
    'lr':2e-4,
    'weight_decay':1e-3,
    'patience' :5,
    'eval_batch_size':256,
    'num_epoch' :200,
    'imb_weight': 2
}
param_combinations = list(itertools.product(*param_grid.values()))
def evaluate_params(params):
    params_dict = dict(zip(param_grid.keys(), params))
    training_arguments.update(params_dict)

    # training_arguments.update(params)
    model = transtab.build_classifier(cate_cols, num_cols, bin_cols, imb_weight=training_arguments['imb_weight'])
    model = model.to('cuda')
    transtab.train(model, train_set, valid_set, **training_arguments)

    print(params)
    # =========================================================================================
    prob_int = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
    result_int = get_final_result(prob_int, df_alin_test.iloc[:, 0])

    prob_ext12 = transtab.predict(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0])
    result_ext12 = get_final_result(prob_ext12, df_alin_ext12.iloc[:, 0])

    prob_ext3 = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
    result_ext3 = get_final_result(prob_ext3, df_alin_ext3.iloc[:, 0])

    avg_auc = (float(result_int[5])+float(result_ext12[5])+float(result_ext3[5]))/3

    # avg_auc = np.mean(float(result_int[5]), float(result_ext12[5]), float(result_ext3[5]))
    return avg_auc, model, result_int, result_ext12, result_ext3
    # =========================================================================================
    # y_pred = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
    # result_ext3 = get_final_result(y_pred, df_alin_ext3.iloc[:, 0])
    # return float(result_ext3[5]), model, result_ext3

PATH = 'LF_bio_241006_best_0'
best_params = None
best_score = -float('inf')  # 我们是在最大化某个指标
count = 0
for params in param_combinations:
    score, model, result_int, result_ext12, result_ext3 = evaluate_params(params)

    if score > best_score:
        best_score = score
        best_params = params
        best_model = model

        # y_pred = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
        # result_ext3 = get_final_result(y_pred, df_alin_ext3.iloc[:, 0])
        # y_pred = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
        # result_test = get_final_result(y_pred, df_alin_test.iloc[:, 0])

        model.save('./checkpoint/'+PATH + str(count))
        count += 1
        print('./checkpoint/'+PATH + str(count))
        with open('./results/'+PATH+'.txt', 'a') as f:
            f.write(str(best_params) + '\n' +
                    str(result_int) + '\n' +
                    str(result_ext12) + '\n'+
                    str(result_ext3)+'\n'+'\n')
        # torch.save(best_model, './checkpoint/LF_bio_240613_best.pth')
        print(best_params, '\n',
              '**************************************************************************************')
print("Best parameters found:", best_params,'\n', "Best score:",best_score)

# model_path = './checkpoint/'+PATH
# model = transtab.build_classifier(checkpoint=model_path,device='cuda:0')
# model.eval()
# y_pred = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
# result_ext3 = get_final_result(y_pred, df_alin_ext3.iloc[:, 0])

# =========================================================
# '''
# Transtab
# '''
# training_arguments = {
#     'num_epoch':200,
#     'batch_size':64,
#     'lr':2e-4,
#     'output_dir':'./checkpoint/LF_bio_240512_0',
#     'weight_decay':1e-3,
#     'patience' :5,
#     'eval_batch_size':256,
# }
# model = transtab.build_classifier(cate_cols, num_cols, bin_cols)
# model = model.to('cuda')
# transtab.train(model, train_set,valid_set, **training_arguments)
#
# y_pred = transtab.predict(model, df_alin_test.iloc[:,1:], df_alin_test.iloc[:,0])
# result_int = get_final_result(y_pred, df_alin_test.iloc[:,0])
#
# y_pred = transtab.predict(model, df_alin_ext12.iloc[:,1:], df_alin_ext12.iloc[:,0])
# result_ext12 = get_final_result(y_pred, df_alin_ext12.iloc[:,0])
#
# y_pred = transtab.predict(model, df_alin_ext3.iloc[:,1:], df_alin_ext3.iloc[:,0])
# result_3 = get_final_result(y_pred, df_alin_ext3.iloc[:,0])
# =============================================================================
# '''
# TabTransformer
# '''
# def data_pre(df):
#     X_df_train, y_df_train = df.iloc[:,1:], df.iloc[:,0]
#     index_of_age = X_df_train.columns.get_loc('Age')
#     categories = tuple(range(0,index_of_age))
#     num_continuous = X_df_train.shape[1] - len(categories)
#     x_categ = X_df_train.iloc[:,:index_of_age]
#     x_numer = X_df_train.iloc[:,index_of_age:]
#
#     x_categ = X_df_train.iloc[:,:index_of_age]
#     x_numer = X_df_train.iloc[:,index_of_age:]
#
#     x_categ_ten = torch.tensor(np.array(x_categ), dtype=torch.long).to('cpu')
#     x_numer_ten = torch.tensor(np.array(x_numer), dtype=torch.float).to('cpu')
#
#     return x_categ_ten, x_numer_ten
#
# x_categ_ten, x_numer_ten = data_pre(df_alin_train)
# x_categ_test, x_numer_test = data_pre(df_alin_test)
# x_categ_ex12, x_numer_ex12 = data_pre(df_alin_ext12)
# x_categ_ex3, x_numer_ex3 = data_pre(df_alin_ext3)
#
#
#
# X_df_train, y_df_train = df_alin_train.iloc[:,1:], df_alin_train.iloc[:,0]
# index_of_age = X_df_train.columns.get_loc('Age')
# categories = tuple(range(0,index_of_age))
# num_continuous = X_df_train.shape[1] - len(categories)
# x_categ = X_df_train.iloc[:,:index_of_age]
# x_numer = X_df_train.iloc[:,index_of_age:]
#
# categories_list = []
# for col in categories:
#     col_unique = len(x_categ.iloc[:,col].unique())
#     categories_list.append(col_unique)
#
# model = TabTransformer(
#     categories = categories_list,      # tuple containing the number of unique values within each category
#     num_continuous = num_continuous,                # number of continuous values
#     dim = 32,                           # dimension, paper set at 32
#     dim_out = 2,                        # binary prediction, but could be anything
#     depth = 6,                          # depth, paper recommended 6
#     heads = 8,                          # heads, paper recommends 8
#     attn_dropout = 0.1,                 # post-attention dropout
#     ff_dropout = 0.1                    # feed forward dropout
# ).to(torch.float32)
# model.to(device='cpu')
# # model = model.float()
#
# # criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
# # criterion= nn.BCELoss
# # optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# for epoch in range(200):
#     optimizer.zero_grad()
#     pred = model(x_categ_ten, x_numer_ten)
#     label = torch.tensor(np.array(X_df_train.iloc[:,0])).to('cpu').to(torch.float32)
#     loss = criterion(pred[:,1], label)
#     loss.backward()
#     optimizer.step()
#     print(loss)
#
# model.eval()
# pred = model(x_categ_test, x_numer_test)
# result = get_final_result(pred[:,0].detach().cpu().numpy(), df_alin_test.iloc[:,0])
#
# pred = model(x_categ_ex12, x_numer_ex12)
# result = get_final_result(pred[:,0].detach().cpu().numpy(), df_alin_ext12.iloc[:,0])
#
# pred = model(x_categ_ex3, x_numer_ex3)
# result = get_final_result(pred[:,0].detach().cpu().numpy(), df_alin_ext3.iloc[:,0])



