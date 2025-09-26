from sympy.abc import kappa
import matplotlib.pyplot as plt
import transtab
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score,roc_curve
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
    # max_youden_index = -1
    # best_threshold = None
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
    con = confusion_matrix(label, y_preds)
    TN = con[0, 0]
    FP = con[0, 1]
    FN = con[1, 0]
    TP = con[1, 1]
    if (TP + FP) > 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 0  # 或者可以设置为 NaN，取决于你的需求

    if (TN + FN) > 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 0
    specificity = TN / (TN + FP)

    print(con)
    print('acc:', '{:.3f}'.format(acc), 'f1:', '{:.3f}'.format(f1), 'auc:', '{:.3f}'.format(auc),
          'kappa:', '{:.3f}'.format(kappa), 'MCC:', '{:.3f}'.format(mcc), 'recall:', '{:.3f}'.format(recall),
          'specificity:', '{:.3f}'.format(specificity), 'PPV:', '{:.3f}'.format(PPV), 'NPV:', '{:.3f}'.format(NPV))
    print('{:.3f}'.format(auc), '{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall),
          '{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV), '\n')
    # print(con)
    # print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
    #         'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc),
    #       'sensitivity:','{:.3f}'.format(recall), 'specificity:','{:.3f}'.format(specificirty) )
    return y_preds

def get_final_result_youden( preds,label,):
    max_youden_index = -1
    best_threshold = None

    # Compute ROC curve
    fpr, tpr, thresholds_roc = roc_curve(label, preds)

    # Calculate Youden Index for each threshold
    for i in range(len(thresholds_roc)):
        thresh = thresholds_roc[i]
        y_preds = (preds > thresh).astype(int)

        # Confusion matrix
        con = confusion_matrix(label, y_preds)
        TN = con[0, 0]
        FP = con[0, 1]
        FN = con[1, 0]
        TP = con[1, 1]

        # Sensitivity (Recall)
        if (TP + FN) > 0:
            sensitivity = TP / (TP + FN)
        else:
            sensitivity = 0

        # Specificity
        if (TN + FP) > 0:
            specificity = TN / (TN + FP)
        else:
            specificity = 0

        # Youden Index
        youden_index = sensitivity + specificity - 1

        # Check if this is the maximum Youden Index
        if youden_index > max_youden_index:
            max_youden_index = youden_index
            best_threshold = thresh

    # Output the results
    print(f"Best Threshold: {best_threshold}")
    print(f"Maximum Youden Index: {max_youden_index}")

    return best_threshold, max_youden_index

info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
df_alin_test = pd.read_csv('../data_process/df_int_test.csv')


df_PHLF_mimic = pd.read_csv('../data_process/df_PHLF_mimic.csv')
df_PHLF_ext6 = pd.read_csv('../data_process/df_PHLF_ext6.csv')
df_PHLF_ext7 = pd.read_csv('../data_process/df_PHLF_ext7_few_null.csv')

df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]

df_alin_mimic = df_PHLF_mimic.iloc[:,1:]
df_alin_ext6 = df_PHLF_ext6.iloc[:,1:]
df_alin_ext7 = df_PHLF_ext7.iloc[:,1:]

df_train, df_valid = train_test_split(df_alin_train, test_size=0.2,random_state=42)
train_set = [df_train.iloc[:,1:], df_train.iloc[:,0]]
valid_set = [df_valid.iloc[:,1:], df_valid.iloc[:,0]]

# =============================================================================
['LF_bio_240616_best_29', 'LF_bio_240617_best_16','LF_bio_240625_best_08','LF_bio_240625_best_018',0.6,
 'LF_bio_240627_best_08', 'LF_bio_241006_best_01', 'LF_bio_240623_best_27','LF_bio_240623_best_213',
 'LF_bio_240623_best_217','LF_bio_240625_best_08',0.6,'LF_bio_240625_best_017', 0.45,
'LF_bio_240618_best_412', 0.69, 'LF_bio_240623_best_27', 0.67]

# mimic threshold: 0.016
# path = 'LF_bio_240625_best_018'
path = 'LF_bio_240625_best_018'
# mimic: 0.0128: 0.236  6 -> 78
# ext7: 0.25, ext6: 0.46
# mimic: 0.0142

threshold = 0.20
model = transtab.build_classifier(cate_cols, num_cols, bin_cols)
model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+path)

# df_alin_mimic = df_alin_mimic.fillna(0)
# df_alin_ext6 = df_alin_ext6.fillna(0)
# df_alin_ext7 = df_alin_ext7.fillna(0)

prob_mimic = transtab.predict(model, df_alin_mimic.iloc[:, 1:], df_alin_mimic.iloc[:, 0])
result_mimic = get_final_result(threshold,prob_mimic, df_alin_mimic.iloc[:, 0])

prob_ext6 = transtab.predict(model, df_alin_ext6.iloc[:, 1:], df_alin_ext6.iloc[:, 0])
best_threshold_6, max_youden_index_6 = get_final_result_youden(prob_ext6, df_alin_ext6.iloc[:, 0])
result_ext6 = get_final_result(threshold, prob_ext6, df_alin_ext6.iloc[:, 0])

prob_ext7 = transtab.predict(model, df_alin_ext7.iloc[:, 1:], df_alin_ext7.iloc[:, 0])
best_threshold_7, max_youden_index_7 = get_final_result_youden(prob_ext7, df_alin_ext7.iloc[:, 0])
result_ext7 = get_final_result(threshold, prob_ext7, df_alin_ext7.iloc[:, 0])

Ext_6 = pd.read_excel('../data_process/EX6.xlsx')
Ext_7 = pd.read_excel('../data_process/EX7.xlsx')

Ext6_prob = pd.concat([pd.DataFrame(prob_ext6, columns=['probability']),Ext_6],axis=1)
Ext7_prob = pd.concat([pd.DataFrame(prob_ext7, columns=['probability']),Ext_7],axis=1)

Ext7_prob.to_csv('../data_process/EX7_prob.csv')
Ext6_prob.to_csv('../data_process/EX6_prob.csv')

print('========================================================================')
model_name = 'our'
# dict_result_obtain(threshold, prob_mimic, np.array(df_alin_mimic.iloc[:,0]),
#                    '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model_name+'_mimic_result.json')
# er_int = error_samples(df_alin_mimic, ori_int_test, result_mimic, df_alin_mimic.iloc[:, 0])
print('-------------------------------')
# dict_result_obtain(threshold, prob_ext6, np.array(df_alin_ext6.iloc[:,0]),
#                    '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model_name+'_ext6_result.json')
# er_ext12 = error_samples(df_alin_ext6, ori_data_ext12, result_ext12, df_alin_ext6.iloc[:, 0])
print('-------------------------------')
dict_result_obtain(threshold, prob_ext7, np.array(df_alin_ext7.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model_name+'_ext7_result.json')
# er_ext3 = error_samples(df_alin_ext7, ori_new_ext_3, result_ext3, New_ext_3.iloc[:, 0])
print('-------------------------------')





