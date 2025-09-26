import matplotlib.pyplot as plt
import datetime
import transtab
import pandas as pd
from child_class import GateChildHead,GateChildHead_eval
import pickle
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np
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
    return {'acc:': round(acc, 3), 'f1:': round(f1, 3), 'auc:': round(auc, 3),
            'kappa:': round(kappa, 3), 'MCC:': round(mcc, 3), 'recall:': round(recall, 3),
            'precision:': round(precision, 3)}
def X_y_split(df, lal):
    X = df.drop([lal], axis=1)
    y = df[lal]
    return X, y


df_PHLF_valid_grade = pd.read_csv('../data_process/df_PHLF_valid_grade_2.csv')
ori_PHLF_valid_grade = pd.read_csv('../data_process/ori_PHLF_valid_grade.csv')
df_PHLF_extvalid_grade = pd.read_csv('../data_process/df_PHLF_extvalid_grade.csv')
df_PHLF_train_with_grade = pd.read_csv('../data_process/df_PHLF_train_with_grade.csv')

train_data = df_PHLF_train_with_grade
# train_data, valid_data = train_test_split(df_PHLF_train_with_grade, test_size=0.2, random_state=42)

path = 'LF_bio_240625_best_018'
threshold=0.6
info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']
model = transtab.build_classifier_multi_task_onlyclss(cate_cols, num_cols, bin_cols)
model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+path)

weights = [1.0, 6.0, 16.0]
clf = GateChildHead_eval(
    parent_classes=2,
    weights=weights,
    child_classes=3)
clf.load_state_dict(torch.load('./checkpoint/LF_bio_240625_best_018_grade_420.pth'))


data = df_PHLF_valid_grade

x_data = df_PHLF_valid_grade.iloc[:,1:]

def fill_missing_values(df,cate_cols, num_cols,bin_cols):
    # Iterate over columns in the DataFrame
    for column in df.columns:
        if column in cate_cols or column in bin_cols:
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)

        else:
            # Fill missing values with the mean for continuous variables
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
    return df

def shap_predict(x_data_fill,):
    if isinstance(x_data_fill, np.ndarray):
        x_data_fill = pd.DataFrame(x_data_fill, columns=df_PHLF_valid_grade.columns[1:])
    prob_parent, encoder_output = transtab.evaluator.predict_all_prob(model, x_data_fill.iloc[:,1:],
                                                                      x_data_fill.iloc[:,0])
    logits_child= clf(encoder_output, prob_parent, threshold)
    y_pred = logits_child.detach().cpu().numpy()
    y_pred = y_pred.argmax(1)
    return y_pred

x_data_fill = fill_missing_values(x_data, cate_cols, num_cols,bin_cols)
data_fill = fill_missing_values(data, cate_cols, num_cols,bin_cols)

x_data_fill_sam = x_data_fill.iloc[:500,:]

sam_records = [954, 991, 1139]
sam_data = x_data_fill.loc[sam_records,:]
x_data_fill_sam = pd.concat([x_data_fill_sam, sam_data])

explainer = shap.KernelExplainer(model=shap_predict, data=x_data_fill_sam, )

shap_values = explainer.shap_values(x_data_fill_sam, nsamples=80)
with open('./results/' + path + '/Global_explanation/grde_shap.pkl','wb') as f:
    pickle.dump(shap_values, f)

with open('./results/' + path + '/Global_explanation/grde_shap.pkl','rb') as f:
    shap_values = pickle.load(f)

col_idx = json.load(open('../data_process/col_idx.json'))
for col in x_data_fill_sam.columns:
    if col in col_idx.keys():
        x_data_fill_sam= x_data_fill_sam.rename(columns={col:col_idx[col]})

with open('../data_process/simple2form_cols.json','r') as f:
    form_cols = json.load(f)
x_data_fill_sam = x_data_fill_sam.rename(columns=form_cols)

x_ori_sam = ori_PHLF_valid_grade.iloc[x_data_fill_sam.index,1:]

shap.initjs()
for i, idx in zip(range(x_ori_sam.shape[0]),x_ori_sam.index):
    shap.force_plot(explainer.expected_value, shap_values[i], np.array(x_ori_sam.iloc[i]),
                    list(x_data_fill_sam.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
    plt.savefig('./results/' + path + '/SHAP_grade/force_error'+str(idx)+'.eps')
    plt.close()








# info = json.load(open('../data_process/exterior_info.json'))

# df_test2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_ext_test.csv')
# df_test2 = df_test2.iloc[:,1:]
#
#
# X_test2, y_test2 = X_y_split(df_test2, info['target'][0])
# for col in info['bin_cols']:
#     idx = ~X_test2[col].isnull()
#     X_test2 = X_test2.loc[idx,:]
#     y_test2 = y_test2.loc[idx]
# X_test2[info['bin_cols']] = X_test2[info['bin_cols']].astype(int)
#
# model = transtab.build_classifier(checkpoint='./checkpoint/LF_bio_Non_240517_0',device='cuda:0')
# # model = nn.DataParallel(model).to('cuda')
#
# ypred = transtab.predict(model, X_test2, y_test2)
#
# result = get_final_result(y_test2, ypred)
#
# transtab.evaluate(ypred, y_test2, seed=123, metric='auc')
# print(transtab.evaluate(ypred, y_test2, seed=123, metric='auc'))
#
# col_idx = json.load(open('../data_process/col_idx.json'))
# shap_values = np.load('./shap_values/'+str(inter)+'.npy')
# # new_cols = list(col_idx.values())
# # new_cols = new_cols[1:]
#
# def shap_predict(data):
#     return transtab.predict_fun(model, data, X_test2.columns)
#
#
# x_test_clean = X_test2.dropna()
#
#
# # data = x_test_clean.iloc[:150,:]
# data = x_test_clean.iloc[:100,:]
#
# explainer = shap.KernelExplainer(model=shap_predict, data=data, )
# shap_values = explainer.shap_values(data, nsamples=80)
#
#
#
#
# # =================================================================
# nsamples=80
# shap_values = explainer.shap_values(data, nsamples=80)
# np.save('./shap_values/'+str(inter), shap_values)
# print(shap_values.shape)
# # =================================================================
#
# for col in data.columns:
#     if col in col_idx.keys():
#         data= data.rename(columns={col:col_idx[col]})
#         # print(col)
#
# # data.columns = new_cols
#
# summary_plot = shap.summary_plot(shap_values,data, show=False)
#
# # fig = plt.gcf()
# # fig.set_size_inches(16,16)
# # shap.plots.heatmap(shap_values)
# plt.savefig('./shap_values/Ex_240517_0.jpg')


