import matplotlib.pyplot as plt
import datetime
import transtab
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np
import pickle
import torch
from torch import nn
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN,BorderlineSMOTE
import shap
from transtab.modeling_transtab import TransTabClassifier, TransTabFeatureExtractor, TransTabFeatureProcessor
from transformers import BertTokenizer, BertTokenizerFast, GPT2TokenizerFast, AutoTokenizer
from imblearn.under_sampling import RandomUnderSampler

inter = 11

def get_final_result(y_test, preds):
    y_preds2 = 1-preds
    y_preds = np.concatenate([ y_preds2.reshape(-1,1), preds.reshape(-1,1),],axis=1)
    y_preds = y_preds.argmax(1)
    auc = roc_auc_score(y_true=y_test, y_score=preds, )
    acc = accuracy_score(y_true=y_test, y_pred=y_preds)
    recall = recall_score(y_true=y_test, y_pred=y_preds, )
    # precision = precision_score(y_true=y_test, y_pred=y_preds, labels=[0])
    precision = precision_score(y_true=y_test, y_pred=y_preds)
    f1 = f1_score(y_true=y_test, y_pred=y_preds, )
    kappa = cohen_kappa_score(y1=y_test, y2=y_preds, )
    mcc = matthews_corrcoef(y_true=y_test, y_pred=y_preds)
    cm = confusion_matrix(y_true=y_test, y_pred=y_preds)
    print(cm)
    print({'acc:': round(acc, 3), 'f1:': round(f1, 3), 'auc:': round(auc, 3),
           'kappa:': round(kappa, 3), 'MCC:': round(mcc, 3), 'recall:': round(recall, 3),
           'precision:': round(precision, 3)})
    return y_preds


def X_y_split(df, lal):
    X = df.drop([lal], axis=1)
    y = df[lal]
    return X, y

info = json.load(open('../data_process/DropLiverseg_info.json'))

df_test2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_DropLiver_0508_test.csv')
df_test2 = df_test2.iloc[:,1:]
df_train2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_DropLiver_0508_train.csv')
df_train2 = df_train2.iloc[:,1:]

X_test2, y_test2 = X_y_split(df_test2, info['target'][0])
X_train2, y_train2 = X_y_split(df_train2, info['target'][0])

file_path = 'LF_bio_240531_06'

model = transtab.build_classifier(checkpoint='./checkpoint/'+file_path,device='cuda:0')
# model = nn.DataParallel(model).to('cuda')

ypred = transtab.predict(model, X_test2, y_test2)
y_pred = get_final_result(y_test2, ypred)

ypred_series = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
incorrect_samples = X_test2.loc[y_test2 != ypred_series, :]
incorrect_samples.fillna(0, inplace=True)


# transtab.evaluate(ypred, y_test2, seed=123, metric='auc')
# print(transtab.evaluate(ypred, y_test2, seed=123, metric='auc'))

col_idx = json.load(open('../data_process/col_idx.json'))
# shap_values = np.load('./shap_values/'+str(inter)+'.npy')
# new_cols = list(col_idx.values())
# new_cols = new_cols[1:]

def shap_predict(data):
    return transtab.predict_fun(model, data, X_test2.columns)

train_clean = df_train2.dropna()

x_train_clean, y_train_clean = X_y_split(train_clean, info['target'][0])

# x_train_clean = X_train2.dropna()
x_test_clean = X_test2.dropna()


# data = x_test_clean.iloc[:150,:]
data = x_train_clean.iloc[:100,:]

data = pd.concat([incorrect_samples,data])

ypred_clean = transtab.predict(model, x_train_clean, y_train_clean)
result = get_final_result(y_train_clean, ypred_clean)


explainer = shap.KernelExplainer(model=shap_predict, data=data, )
# file_path = '240531_ShapValue_01'

# =================================================================
# shap_values = explainer.shap_values(data, nsamples=100)
# with open('./shap_values/'+file_path+'.pkl','wb') as f:
#     pickle.dump(shap_values, f)
# =================================================================

# incorrest_shap_values = shap_values[:incorrect_samples.shape[0],:]
# shap_exp = shap.Explanation(shap_values, data)
# for i in range(incorrect_samples.shape[0]):
#     # base_value = shap_values.base_values[i]
#     shap_exp = shap.Explanation(shap_values[i,:], data.iloc[i,:])
#     shap.plots.waterfall(
#         shap_exp,  # shap_values的第一个元素对应于第一个（也是唯一）样本  # 使用matplotlib绘制，以便于自定义和保存图表
#         show=True  # 显示图表
#     )




with open('./shap_values/'+file_path+'.pkl','rb') as f:
    shap_values = pickle.load(f)



# np.save('./shap_values/'+file_path, shap_values)
# print(shap_values.shape)
# shap_values = np.load('./shap_values/'+file_path+'.npy')

# for idx, col in enumerate(data.columns):
#     if col == 'Ascites':
#         print(idx)
#         break
# Ascites_values = shap_values[:,idx]
# Ascites_values>0
#
# data.iloc[Ascites_values>0,:]

for col in data.columns:
    if col in col_idx.keys():
        data= data.rename(columns={col:col_idx[col]})
        # print(col)
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})
data['Methods'] = data['Methods'].map({'Laparoscopic surgery': 0, 'Transfer to laparotomy': 1,
                                             'Laparotomy': 2})
idx2cat = {'Gender': [0, 1], 'Methods': [0, 1, 2]}


# data.columns = new_cols
force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test2.iloc[0,:],show=True)

summary_plot = shap.summary_plot(shap_values,data, show=False,)

# shap.plots.text(shap_values, data,)
# text_data = data.iloc[:,:2]
# shap.text_plot(text_data,shap_values[:,:2])

# fig = plt.gcf()
# fig.set_size_inches(12,16)
# shap.plots.heatmap(shap_values)
# plt.savefig('./shap_values'+file_path+'/.jpg')
plt.savefig('./results/'+file_path+'/shap_summary.png')

sim_E_num = []
for i in info['E_num']:
    sim_E_num.append(col_idx[i])
sim_D1_num = []
for i in info['D1_num']:
    sim_D1_num.append(col_idx[i])

# for e_col, d_col in zip(sim_E_num, sim_D1_num):
#     e_idx = data.columns.get_loc(e_col)
#     d1_idx = data.columns.get_loc(d_col)
#     print(e_col,d_col)
#     shap.dependence_plot(
#         e_idx,  # 只需提供主要特征的索引
#         shap_values,
#         data,
#         interaction_index=d1_idx,  # 指定交互特征的索引
#         alpha=0.65,
#         show=False
#     )
#     ax = plt.gca()
#     num_points = data.shape[0]
#     sizes = [50] * num_points
#     scatter = ax.collections[0]
#     scatter.set_sizes(sizes)
#
#     current_xlim = ax.get_xlim()
#     padding = (current_xlim[1] - current_xlim[0]) * 0.1  # 计算10%的范围作为padding
#     new_xlim = (current_xlim[0] - padding, current_xlim[1])
#     ax.set_xlim(new_xlim)
#
#     # plt.savefig('./shap_values/'+file_path+'/dependence'+ e_col + '.jpg')
#     plt.savefig('./results/' + file_path + '/dependence'+ e_col + '.jpg')


incorrest_shap_values = shap_values[:incorrect_samples.shape[0],:]
# shap_exp = shap.Explanation(shap_values, data)
for i in range(incorrect_samples.shape[0]):
    # base_value = shap_values.base_values[i]
    # shap_exp = shap.Explanation(shap_values[i,:], data.iloc[i,:])
    shap.plots.force(
        explainer.expected_value,
        shap_values[i],
        show=False# shap_values的第一个元素对应于第一个（也是唯一）样本  # 使用matplotlib绘制，以便于自定义和保存图表# 显示图表
    )
    plt.savefig('./results/' + file_path + '/force'+ str(i) + '.jpg')




    # shap.partial_dependence_plot([data.loc[:,e_col], data.loc[:,d_col]], shap_predict, data,
    #                               show=True)
# shap.decision_plot( e_idx,  # 使用索引而不是列名
#     shap_values,  # 对应特征的shap值
#     feature_names=data.columns.tolist(),  # 提供所有特征的名称列表
#     highlight=[d1_idx],  # 高亮的特征索引
#     show=True)