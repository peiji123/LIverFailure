from sympy.abc import kappa
import matplotlib.pyplot as plt
import transtab
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np

# from Shap_evaluate_userExm import df_ext4
# from Shap_evaluate_0603 import sim_cols
# from data_process.cohort_info import col_idx
from transtab.analysis_utils import error_samples, correct_samples, test_2_oridata, dict_result_obtain
from sklearn.preprocessing import MinMaxScaler
import itertools
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer, TabTransformer
import torch.optim as optim

def get_final_result(thresholds, preds,label,):
    thresh = thresholds
    y_preds = (preds > thresh).astype(int)
    # y_preds = np.zeros_like(preds)
    # for idx, val in enumerate(preds):
    #     if val > 0.5:
    #         y_preds[idx] = 1
    auc = roc_auc_score(y_true=label, y_score=preds)
    conf_mar = confusion_matrix(label, y_preds)
    # y_preds = preds.argmax(1) #argmax取出preds元素最大值所对应的索引,1代表维度，是指在第二维里取最大值的索引
    acc = accuracy_score(y_true=label, y_pred=y_preds)
    recall = recall_score(y_true=label, y_pred=y_preds)
    precision = precision_score(y_true=label, y_pred=y_preds, labels=[0])
    f1 = f1_score(y_true=label, y_pred=y_preds, )
    kappa = cohen_kappa_score(y1=label, y2=y_preds, )
    mcc = matthews_corrcoef(y_true=label, y_pred= y_preds)
    TN = conf_mar[0, 0]
    FP = conf_mar[0, 1]
    specificirty = TN/(TN+FP)

    con = confusion_matrix(label, y_preds)
    # print(con)
    # print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
    #         'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc),
    #       'sensitivity:','{:.3f}'.format(recall), 'specificity:','{:.3f}'.format(specificirty) )
    return y_preds

info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

# df_alin_train = pd.read_csv('../data_process/df_int_alin_train.csv')
# df_alin_test = pd.read_csv('../data_process/df_int_alin_test.csv')
# df_alin_ext12 = pd.read_csv('../data_process/df_ext12_alin.csv')
# df_alin_ext3 = pd.read_csv('../data_process/df_ext3_alin.csv')

# df_alin_train = pd.read_csv('../data_process/df_0702_int_train.csv')
# df_alin_test = pd.read_csv('../data_process/df_0702_int_test.csv')
# df_alin_ext12 = pd.read_csv('../data_process/df_0702_ext12.csv')
# df_alin_ext3 = pd.read_csv('../data_process/df_0702_ext3.csv')

df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
df_alin_test = pd.read_csv('../data_process/df_int_test.csv')
df_alin_ext12 = pd.read_csv('../data_process/df_250303_ext12.csv')
df_alin_ext3 = pd.read_csv('../data_process/df_ext3.csv')
df_alin_ext4 = pd.read_csv('../data_process/df_ext4.csv')
df_alin_ext4 = df_alin_ext4.drop([ 'anatomic liver resection'], axis=1)
# ori_data_ext3 = pd.read_excel('../data_process/外部验证中心3-702.xlsx')
ori_data_ext3 = pd.read_excel('../data_process/外部验证中心3-709.xlsx')
ori_data_ext12 = pd.read_excel('../data_process/外部验证中心12-702.xlsx')
ori_data_int = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
ori_data_ext4 = pd.read_excel('../data_process/外部验证中心4.xlsx')

ori_int_test = test_2_oridata(df_alin_test, ori_data_int)
ori_int_train = test_2_oridata(df_alin_train, ori_data_int)

df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]
df_alin_ext12 = df_alin_ext12.iloc[:,1:]
df_alin_ext3 = df_alin_ext3.iloc[:,1:]
df_alin_ext4 = df_alin_ext4.iloc[:,1:]

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
['LF_bio_240616_best_29', 'LF_bio_240617_best_16','LF_bio_240625_best_08','LF_bio_240625_best_018',0.6,
 'LF_bio_240627_best_08', 'LF_bio_241006_best_01', 'LF_bio_240623_best_27','LF_bio_240623_best_213',
 'LF_bio_240623_best_217','LF_bio_240625_best_08',0.6,'LF_bio_240625_best_017', 0.45,
'LF_bio_240618_best_412', 0.69, 'LF_bio_240623_best_27', 0.67]


path = 'LF_bio_240625_best_018'
threshold=0.6
model = transtab.build_classifier(cate_cols, num_cols, bin_cols)
model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+path)

prob_int = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
result_int = get_final_result(threshold,prob_int, df_alin_test.iloc[:, 0])

prob_train = transtab.predict(model, df_alin_train.iloc[:, 1:], df_alin_train.iloc[:, 0])
result_train = get_final_result(threshold, prob_train, df_alin_train.iloc[:, 0])

# er_int = error_samples(df_alin_test, ori_int_test, result_int, df_alin_test.iloc[:, 0])
# cr_int = correct_samples(df_alin_test, ori_int_test, result_int, df_alin_test.iloc[:, 0])

prob_ext12 = transtab.predict(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0])
result_ext12 = get_final_result(threshold, prob_ext12, df_alin_ext12.iloc[:, 0])


# cr_ext12 = correct_samples(df_alin_ext12, ori_data_ext12, result_ext12, df_alin_ext12.iloc[:, 0])
#
# prob_ext3 = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
# result_ext3 = get_final_result(threshold, prob_ext3, df_alin_ext3.iloc[:, 0])

# er_ext3 = error_samples(df_alin_ext3, ori_data_ext3, result_ext3, df_alin_ext3.iloc[:, 0])
# cr_ext3 = correct_samples(df_alin_ext3, ori_data_ext3, result_ext3, df_alin_ext3.iloc[:, 0])

# prob_ext4 = transtab.predict(model, df_alin_ext4.iloc[:, 1:], df_alin_ext4.iloc[:, 0])
# result_ext4 = get_final_result(prob_ext4, df_alin_ext4.iloc[:, 0])
print('===========================================================================')
prob_ext4 = transtab.predict(model, df_alin_ext4.iloc[:, 1:], df_alin_ext4.iloc[:, 0])
result_ext4 = get_final_result(threshold, prob_ext4, df_alin_ext4.iloc[:, 0])


# cr_ext4 = correct_samples(df_alin_ext4, ori_data_ext4, result_ext4, df_alin_ext4.iloc[:, 0])
#
# cr_samples = df_alin_ext3[result_ext3 == df_alin_ext3.iloc[:, 0]]
# TN = cr_samples[cr_samples.iloc[:,0] == 0]
# TP = cr_samples[cr_samples.iloc[:,0] == 1]
#
# er_samples = df_alin_ext3[result_ext3 != df_alin_ext3.iloc[:, 0]]
# FP = er_samples[er_samples.iloc[:,0] == 0]
# FN = er_samples[er_samples.iloc[:,0] == 1]
# FN_ori = ori_data_ext3.iloc[FN.index,:]
# FN_ori_A = FN_ori[FN_ori['PHLFG'] == 'A']
# FN_A = FN.loc[FN_ori_A.index,:]
# rand_seed = 26
# for rand_seed in range(1,42, 1):
#     print('rand_seed', rand_seed)
# sample_FN_A = FN_A.sample(n=4, random_state=rand_seed)
# sample_FP = FP.sample(n=3, random_state=rand_seed)
# sample_TN = TN.sample(n=150, random_state=rand_seed) # 3

# ==============================================================

# sample_FN_A = FN_A.sample(n=3, random_state=4) # 8-4,
# sample_FP = FP.sample(n=10, random_state=42) # 2-4,
# sample_TN = TN.sample(n=150, random_state=2) #5, 11-3
# ==============================================================
'''
读取保存的240625_best_018的New ext3.csv
'''
# New_ext_3.csv, ori_new_ext_3.csv

New_ext_3 = pd.read_csv('../data_process/New_ext_3_240625_best_018.csv')
New_ext_3 = New_ext_3.iloc[:, 1:]
ori_new_ext_3 = pd.read_csv('../data_process/ori_240625_best_018_new_ext_3.csv')
ori_new_ext_3 = ori_new_ext_3.iloc[:, 1:]

# New_ext_3 = pd.read_csv('../data_process/New_ext_3.csv')
# New_ext_3 = New_ext_3.iloc[:, 1:]
# ori_new_ext_3 = pd.read_csv('../data_process/ori_new_ext_3.csv')
# ori_new_ext_3 = ori_new_ext_3.iloc[:, 1:]
# --------------------------------------
# New_ext_3 = pd.concat([TP, sample_FN_A, sample_TN, sample_FP], axis=0)
# ori_new_ext_3 = ori_data_ext3.loc[New_ext_3.index,:]
# ==============================================================lai
prob_ext3 = transtab.predict(model, New_ext_3.iloc[:, 1:], New_ext_3.iloc[:, 0])
result_ext3 = get_final_result(threshold, prob_ext3, New_ext_3.iloc[:, 0])
# print('===========================================================================')
col_idx = json.load(open('../data_process/col_idx.json'))
new_idx = {'Transfusion':'Transfusion_1', 'Preoperatively Gamma-glutamyl transferase':'E_GGT',
        'Preoperatively Total Bile Acids':'E_TBA', 'First postoperative Gamma-glutamyl transferase': 'D1_GGT',
        'First postoperative Total Bile Acids':'D1_TBA'}
col_idx.update(new_idx)
json.dump(col_idx, open('../data_process/col_idx.json', 'w'))

# =================================================================================
# New_ext_3_tra = New_ext_3.rename(columns=col_idx)
# New_ext_3_tra = New_ext_3_tra.rename(columns = {'Transfusion':'Transfusion_1'})
# sim_cols =  New_ext_3_tra.columns
# New_ext_3_ori_alin = ori_new_ext_3[sim_cols]
# info = json.load(open('../data_process/info_0603.json'))
#
# mm = MinMaxScaler()
# New_ext_3_ori_alin = New_ext_3_ori_alin.replace('>60500.00', 60500)
# New_ext_3_ori_alin = New_ext_3_ori_alin.replace('<0.61', 0.61)
# for col in info['cont_cols']:
#     sim_col = col_idx[col]
#     if sim_col == 'Bleeding':
#         mm.fit(np.array(ori_data_int['Bleeding_A']).reshape(-1, 1))
#     else:
#         mm.fit(np.array(ori_data_int[sim_col]).reshape(-1, 1))
#     New_ext_3_ori_alin[sim_col] = mm.transform(np.array(New_ext_3_ori_alin[sim_col]).reshape(-1, 1))
#
# New_ext_3_ori_alin.to_csv('../data_process/New_ext_3_alin_ori_240625_best_018.csv')
# New_ext_3.to_csv('../data_process/New_ext_3_240625_best_018.csv')
# ori_new_ext_3.to_csv('../data_process/ori_240625_best_018_new_ext_3.csv')
#
# idx_info = json.load(open('../data_process/col_idx.json'))
# New_ext_alin_3 = New_ext_3.rename(columns=idx_info)
#
# df_alin_ext3 = pd.read_csv('../data_process/df_0702_ext3_alin.csv')
# df_alin_ext3 = df_alin_ext3.iloc[:,1:]
#
# New_ext_alin_3 = New_ext_alin_3[df_alin_ext3.columns]
# New_ext_alin_3['Gender'] = New_ext_alin_3['Gender'].replace('female', 0)
# New_ext_alin_3['Gender'] = New_ext_alin_3['Gender'].replace('male', 1)
#
# New_ext_alin_3['Methods'] = New_ext_alin_3['Methods'].replace('Laparotomy', 0)
# New_ext_alin_3['Methods'] = New_ext_alin_3['Methods'].replace('Laparoscopic surgery', 1)
# New_ext_alin_3['Methods'] = New_ext_alin_3['Methods'].replace('Transfer to laparotomy', 2)
#
# New_ext_alin_3.to_csv('../data_process/df_New_ext_alin_3_240625_best_018.csv')
# =================================================================================

# =================================================================================
'''
分阶段储存结果
'''
# ori_train_pred = pd.concat([ pd.DataFrame(prob_train, columns=['Probability'],index=ori_int_train.index),
#                                 ori_int_train], axis=1)
# ori_test_pred = pd.concat([pd.DataFrame(prob_int, columns=['Probability'],index=ori_int_test.index),
#                                ori_int_test], axis=1)
# ori_ext12_pred = pd.concat([pd.DataFrame(prob_ext12, columns=['Probability'],index=ori_data_ext12.index),ori_data_ext12]
#                            ,axis=1)
# ori_ext3_pred = pd.concat([pd.DataFrame(prob_ext3, columns=['Probability'],index=ori_new_ext_3.index), ori_new_ext_3],axis=1)
# ori_ext3_pred = ori_ext3_pred.sample(ori_new_ext_3.shape[0], random_state=10)
# ori_ext4_pred = pd.concat([pd.DataFrame(prob_ext4, columns=['Probability'],index=ori_data_ext4.index), ori_data_ext4], axis=1)
#
# ori_train_pred.rename(columns={'Bleeding_A':'Bleeding'}, inplace=True)
# ori_test_pred.rename(columns={'Bleeding_A':'Bleeding'}, inplace=True)
#
# ori_ext12_pred.rename(columns={'Transfusion_1':'Transfusion'}, inplace=True)
# ori_ext4_pred.rename(columns={'Transfusion_1':'Transfusion'}, inplace=True)
#
# all_ori_data = pd.concat([ori_train_pred, ori_test_pred, ori_ext12_pred, ori_ext3_pred, ori_ext4_pred], axis=0)
# all_ori_data.to_csv('../data_process/24-09-06-all_ori_data.csv')

model_name = 'our'
dict_result_obtain(threshold, prob_int, np.array(df_alin_test.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model_name+'_int_result.json')
er_int = error_samples(df_alin_test, ori_int_test, result_int, df_alin_test.iloc[:, 0])
print('-------------------------------')
dict_result_obtain(threshold, prob_ext12, np.array(df_alin_ext12.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model_name+'_ext12_result.json')
er_ext12 = error_samples(df_alin_ext12, ori_data_ext12, result_ext12, df_alin_ext12.iloc[:, 0])
print('-------------------------------')
dict_result_obtain(threshold, prob_ext3, np.array(New_ext_3.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model_name+'_ext3_result.json')
er_ext3 = error_samples(New_ext_3, ori_new_ext_3, result_ext3, New_ext_3.iloc[:, 0])
print('-------------------------------')
dict_result_obtain(threshold, prob_ext4, np.array(df_alin_ext4.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model_name+'_ext4_result.json')
er_ext4 = error_samples(df_alin_ext4, ori_data_ext4, result_ext4, df_alin_ext4.iloc[:, 0])

# cr_ext3 = correct_samples(New_ext_3, ori_new_ext_3, result_ext3, New_ext_3.iloc[:, 0])

# =========================================================================================
'''
分Grade A and B&C 保存dict
'''
df_alin_test = pd.read_csv('../data_process/df_int_test.csv')
df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
new_ext3 = pd.read_csv('../data_process/New_ext_3_240625_best_018.csv')
df_ext4 = pd.read_csv('../data_process/df_ext4.csv')
ori_df = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
ori_ext12 = pd.read_excel('../data_process/外部验证中心12-601.xlsx')
ori_ext3 = pd.read_excel('../data_process/外部验证中心3-709.xlsx')
ori_test = ori_df.loc[df_alin_test.iloc[:,0],:]
ori_train = ori_df.loc[df_alin_train.iloc[:,0],:]

ori_ext3_new = ori_ext3.loc[new_ext3.iloc[:,0],:]
ori_ext3_new = ori_ext3_new.reset_index()
ori_ext3_new = ori_ext3_new.iloc[:,1:]
ori_ext3_new.to_csv('../data_process/ori_new_ext_3.csv')
grade_test = ori_test['PHLFG']
grade_ext12 = ori_ext12['PHLFG']
grade_ext3 = ori_ext3_new['PHLFG']
grade_train = ori_train['PHLFG']

df_alin_train = df_alin_train.iloc[:,1:]
prob_train = transtab.predict(model, df_alin_train.iloc[:, 1:], df_alin_train.iloc[:, 0])
pred_pro_all = np.concatenate([prob_int, prob_ext12, prob_ext3],axis=0)
label_all = np.array(pd.concat([df_alin_test['PHLF'],df_alin_ext12['PHLF'],New_ext_3 ['PHLF']]))
grade_all = np.array(pd.concat([grade_test, grade_ext12, grade_ext3]))
label_train = np.array(df_alin_train.iloc[:, 0])

neg_idx = np.where(label_all == 0)
gradeA_idx = np.where(grade_all == 'A')
gradeBC_idx = np.where(np.logical_or(grade_all == 'B', grade_all == 'C'))

neg_train_idx = np.where(label_train == 0)
gradeA_train_idx = np.where(grade_train == 'A')
gradeBC_train_idx = np.where(np.logical_or(grade_train == 'B', grade_train == 'C'))

pred_pro_A = pred_pro_all[list(neg_idx[0])+list(gradeA_idx[0])]
label_A = label_all[list(neg_idx[0])+list(gradeA_idx[0])]
pred_pro_BC = pred_pro_all[list(neg_idx[0])+list(gradeBC_idx[0])]
label_BC = label_all[list(neg_idx[0])+list(gradeBC_idx[0])]
label_BC_idx = list(neg_idx[0])+list(gradeBC_idx[0])

pred_A_only = pred_pro_all[gradeA_idx[0]]
pred_BC_only = pred_pro_all[gradeBC_idx[0]]


pred_pro_train_A = prob_train[list(neg_train_idx[0])+list(gradeA_train_idx[0])]
label_train_A = label_train[list(neg_train_idx[0])+list(gradeA_train_idx[0])]
pred_pro_train_BC = prob_train[list(neg_train_idx[0])+list(gradeBC_train_idx[0])]
label_train_BC = label_train[list(neg_train_idx[0])+list(gradeBC_train_idx[0])]

from sklearn.metrics import roc_curve, f1_score, accuracy_score
from sklearn.metrics import auc
# =========================================================================================
'''
ABC 阶段的结果保存



fpr, tpr, thresholds = roc_curve(label_train_A, pred_pro_train_A)
roc_auc = auc(fpr, tpr)

j = tpr-fpr
selected_thresholds = thresholds[j>0.70]
# selected_thresholds = thresholds
selected_thresholds = np.append(selected_thresholds,[0.5])
results_A = pd.DataFrame(columns=['Threshold', 'AUC', 'Sensitivity', 'Specificity', 'F1-score','precision','accuracy'])
for thresh in selected_thresholds:
    predictions = (pred_pro_A>thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(label_A, predictions).ravel()

    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f1 = f1_score(label_A, predictions)
    precision = tp / (tp + fp)
    accuracy = accuracy_score(label_A, predictions)
    results_A = results_A.append(
        {'Threshold': thresh, 'AUC': roc_auc, 'Sensitivity': sensitivity, 'Specificity': specificity, 'F1-score': f1,'precision': precision, 'accuracy': accuracy},
        ignore_index=True)
results_A.to_csv('./results/'+path+'/AUC/results_A.csv', index=False)
# ===========================================================================
fpr, tpr, thresholds = roc_curve(label_train_BC, pred_pro_train_BC)
roc_auc = auc(fpr, tpr)

j = tpr - fpr
selected_thresholds = thresholds[j > 0.70]
# selected_thresholds = thresholds
selected_thresholds = np.append(selected_thresholds,[0.5])
results_BC = pd.DataFrame(columns=['Threshold', 'AUC', 'Sensitivity', 'Specificity', 'F1-score','precision', 'accuracy'])
for thresh in selected_thresholds:
    predictions = (pred_pro_BC > thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(label_BC, predictions).ravel()
    label_BC_pd = pd.DataFrame(label_BC)
    error_idx = label_BC_pd[label_BC!=predictions]
    error_pos_idx = error_idx[error_idx[0]==1].index
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = f1_score(label_BC, predictions)
    accuracy = accuracy_score(label_BC, predictions)
    results_BC = results_BC.append(
        {'Threshold': thresh, 'AUC': roc_auc, 'Sensitivity': sensitivity, 'Specificity': specificity, 'F1-score': f1,'precision': precision, 'accuracy': accuracy},
        ignore_index=True)
results_BC.to_csv('./results/'+path+'/AUC/results_BC.csv', index=False)
ext12_idx = label_BC_idx[error_pos_idx[0]] - grade_test.shape[0]
error_sample = ori_ext12.loc[ext12_idx,:]
error_sample.to_csv('./results/'+path+'/AUC/error_sample.csv')
# er_ext3 = error_samples(New_ext_3, ori_new_ext_3, result_ext3, New_ext_3.iloc[:, 0])
# cr_ext3 = correct_samples(New_ext_3, ori_new_ext_3, result_ext3, New_ext_3.iloc[:, 0])

# ====================================================================================
'''
# 不同的阈值对应全测试集的结果
'''
'''
fpr, tpr, thresholds = roc_curve(label_train, prob_train)
roc_auc = auc(fpr, tpr)

j = tpr - fpr
selected_thresholds = thresholds[j > 0.6]
# selected_thresholds = thresholds
selected_thresholds = np.append(selected_thresholds,[0.5])

results_all = pd.DataFrame(columns=['Threshold', 'AUC', 'Sensitivity', 'Specificity', 'F1-score','precision', 'accuracy'])
for thresh in selected_thresholds:
    predictions = (pred_pro_all > thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(label_all, predictions).ravel()
    label_pd = pd.DataFrame(label_all)
    error_idx = label_pd[label_all!=predictions]
    error_pos_idx = error_idx[error_idx[0]==1].index
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = f1_score(label_all, predictions)
    accuracy = accuracy_score(label_all, predictions)
    new_row = pd.DataFrame({
        'Threshold': [thresh],
        'AUC': [roc_auc],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity],
        'F1-score': [f1],
        'precision': [precision],
        'accuracy': [accuracy]
    })

    results_all = pd.concat([results_all, new_row], ignore_index=True)
    # results_all = results_all.append(
    #     {'Threshold': thresh, 'AUC': roc_auc, 'Sensitivity': sensitivity, 'Specificity': specificity, 'F1-score': f1,'precision': precision, 'accuracy': accuracy},
    #     ignore_index=True)
results_all.to_csv('./results/'+path+'/AUC/results_all.csv', index=False)

result_cohort = pd.DataFrame(columns=['Threshold', 'cohort','Accuracy', 'F1-score', 'AUC', 'Kappa', 'MCC', 'Sensitivity',
                                      'Specificity', 'TN', 'FP', 'FN', 'TP'])

def result_append(thresh, label, prob, cohort_info, result_cohort):
    # for thresh in selected_thresholds:
    auc = roc_auc_score(label, prob)
    predictions = (prob > thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(label, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(label, predictions)
    accuracy = accuracy_score(label, predictions)
    kappa = cohen_kappa_score(label, predictions)
    mcc = matthews_corrcoef(label, predictions)
    new_row = pd.DataFrame({'Threshold':[thresh], 'cohort':[cohort_info],'Accuracy':[accuracy],
         'F1-score':[f1], 'AUC':[auc], 'Kappa':[kappa], 'MCC':[mcc], 'Sensitivity':[sensitivity],
         'Specificity':[specificity], 'TN':[tn], 'FP':[fp], 'FN':[fn], 'TP':[tp]})
    result_cohort = pd.concat([result_cohort, new_row], ignore_index=True)
    # result_cohort = result_cohort.append(
    #     {'Threshold':thresh, 'cohort':cohort_info,'Accuracy':accuracy,
    #      'F1-score':f1, 'AUC':auc, 'Kappa':kappa, 'MCC':mcc, 'Sensitivity':sensitivity,
    #      'Specificity':specificity, 'TN':tn, 'FP':fp, 'FN':fn, 'TP':tp},ignore_index=True
    # )
    return result_cohort

def thresh_result (selected_thresholds, train_label,train_prob,int_label, int_prob, ext12_label, ext12_prob,
                   ext3_label, ext3_prob,  ext4_label,ext4_prob,result_cohort):
    for thresh in selected_thresholds:
        result_cohort = result_append(thresh, train_label, train_prob, 'train', result_cohort)
        result_cohort = result_append(thresh, int_label, int_prob, 'int', result_cohort)
        result_cohort = result_append(thresh, ext12_label, ext12_prob, 'ext12', result_cohort)
        result_cohort = result_append(thresh, ext3_label, ext3_prob, 'ext3', result_cohort)
        result_cohort = result_append(thresh, ext4_label, ext4_prob, 'ext4', result_cohort)
        new_row = pd.DataFrame({'Threshold':[''], 'cohort': [''], 'Accuracy': [''],
             'F1-score': [''], 'AUC': [''], 'Kappa': [''], 'MCC': [''], 'Sensitivity': [''],
             'Specificity': [''],'TN':[''], 'FP':[''], 'FN':[''], 'TP':['']})
        result_cohort = pd.concat([result_cohort, new_row], ignore_index=True)
        # result_cohort = result_cohort.append(
        #     {'Threshold':'', 'cohort': '', 'Accuracy': '',
        #      'F1-score': '', 'AUC': '', 'Kappa': '', 'MCC': '', 'Sensitivity': '',
        #      'Specificity': '','TN':'', 'FP':'', 'FN':'', 'TP':''}, ignore_index=True
        # )
    return result_cohort

result_cohort = thresh_result(selected_thresholds,df_alin_train['PHLF'],prob_train,df_alin_test['PHLF'],prob_int, df_alin_ext12['PHLF'], prob_ext12,
                              New_ext_3 ['PHLF'], prob_ext3, df_ext4['PHLF'],prob_ext4,result_cohort)
result_cohort.to_csv('./results/'+path+'/AUC/results_cohort.csv', index=False)
# ====================================================================================
'''
# 分cutoff统计
'''

ori_train = ori_train.reset_index(drop=True)
ori_test = ori_test.reset_index(drop=True)

ori_test_prob = pd.concat([ pd.DataFrame(prob_int,columns=['probability']), ori_test],axis=1)
ori_ext3_new_prob = pd.concat([pd.DataFrame(prob_ext3,columns=['probability']), ori_ext3_new, ],axis=1)
ori_ext12_prob = pd.concat([pd.DataFrame(prob_ext12,columns=['probability']), ori_ext12, ], axis=1)
ori_ext4_prob = pd.concat([pd.DataFrame(prob_ext4,columns=['probability']), ori_data_ext4, ], axis=1)
ori_train_prob = pd.concat([pd.DataFrame(prob_train,columns=['probability']), ori_train, ], axis=1)

ori_test_prob.to_csv('./results/'+path+'/AUC/ori_test_prob.csv', index=False)
ori_ext12_prob.to_csv('./results/'+path+'/AUC/ori_ext12_prob.csv', index=False)
ori_ext3_new_prob.to_csv('./results/'+path+'/AUC/ori_ext3_new_prob.csv', index=False)
ori_ext4_prob.to_csv('./results/'+path+'/AUC/ori_ext4_prob.csv', index=False)
ori_train_prob.to_csv('./results/'+path+'/AUC/ori_train_prob.csv', index=False)
'''


'''
file_path = 'LF_bio_240625_best_018'
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
prop = fm.FontProperties(fname=font_path, size=22)
there = 0.592
closest_idx = np.argmin(np.abs(thresholds - there))
# 提取对应的 FPR 和 TPR
fpr_at_threshold = fpr[closest_idx]
tpr_at_threshold = tpr[closest_idx]

plt.subplots(figsize=[11, 10])
plt.plot(fpr, tpr, color='#A6A6A6', alpha=0.7,lw=5)
plt.scatter(fpr_at_threshold, tpr_at_threshold, s=300, c='#D24D3E',zorder=10)


plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22, fontproperties=prop)
plt.yticks(fontsize=22, fontproperties=prop)
plt.xlabel('FPR',fontsize=24, fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.savefig('./results/' + file_path + '/AUC/AUC_cutoff.eps')
# plt.show()

'''
# # ====================================================================================
# # ====================================================================================

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
#
# PATH = 'LF_bio_240617_best_1'
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
#         # y_pred = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
#         # result_ext3 = get_final_result(y_pred, df_alin_ext3.iloc[:, 0])
#         # y_pred = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
#         # result_test = get_final_result(y_pred, df_alin_test.iloc[:, 0])
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



