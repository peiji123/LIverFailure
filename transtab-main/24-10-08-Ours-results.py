from sympy.abc import kappa

import transtab
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np


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
    return y_preds

info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
df_alin_test = pd.read_csv('../data_process/df_int_test.csv')
df_alin_ext12 = pd.read_csv('../data_process/df_ext12.csv')
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


df_train, df_valid = train_test_split(df_alin_train, test_size=0.2,random_state=42)
train_set = [df_train.iloc[:,1:], df_train.iloc[:,0]]
valid_set = [df_valid.iloc[:,1:], df_valid.iloc[:,0]]

# =============================================================================
['LF_bio_240616_best_29', 'LF_bio_240617_best_16','LF_bio_240625_best_08','LF_bio_240625_best_018',0.6,
 'LF_bio_240627_best_08', 'LF_bio_241006_best_01', 'LF_bio_240623_best_27','LF_bio_240623_best_213',
 'LF_bio_240623_best_217','LF_bio_240625_best_08',0.6,'LF_bio_240625_best_017', 0.45,
'LF_bio_240618_best_412', 0.69, 'LF_bio_240623_best_27', 0.65,'LF_bio_240625_best_03', 0.69,
 'LF_bio_240625_best_07', 0.35]

# threshold_list = [0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75]
# threshold_list = [0.4, 0.41,0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
# threshold_list = [0.6, 0.61,0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7]
# threshold_list = [0.5, 0.51,0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]
# threshold_list = [0.37]
# threshold_list = [0.65, 0.66,0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75]
# threshold_list = [0.55, 0.56,0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65]
threshold_list = [0.3, 0.31,0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4]
# threshold_list = [0.2, 0.21,0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
# threshold_list = [0.35, 0.36,0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45]
# threshold_list = [0.35, 0.351,0.352, 0.353, 0.354, 0.355, 0.356, 0.357, 0.358, 0.359, 0.36]
# threshold_list = [0.25, 0.251,0.252, 0.253, 0.254, 0.255, 0.256, 0.257, 0.258, 0.259, 0.26]
for threshold in threshold_list:
    path = 'LF_bio_240627_best_010'

    # threshold=0.75
    model = transtab.build_classifier(cate_cols, num_cols, bin_cols)
    model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+path)

    prob_int = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
    result_int = get_final_result(threshold,prob_int, df_alin_test.iloc[:, 0])

    prob_ext12 = transtab.predict(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0])
    result_ext12 = get_final_result(threshold, prob_ext12, df_alin_ext12.iloc[:, 0])

    prob_ext4 = transtab.predict(model, df_alin_ext4.iloc[:, 1:], df_alin_ext4.iloc[:, 0])
    result_ext4 = get_final_result(threshold, prob_ext4, df_alin_ext4.iloc[:, 0])

    '''
    读取保存的240625_best_018的New ext3.csv
    '''

    New_ext_3 = pd.read_csv('../data_process/New_ext_3_240625_best_018.csv')
    New_ext_3 = New_ext_3.iloc[:, 1:]
    ori_new_ext_3 = pd.read_csv('../data_process/ori_240625_best_018_new_ext_3.csv')
    ori_new_ext_3 = ori_new_ext_3.iloc[:, 1:]

    prob_ext3 = transtab.predict(model, New_ext_3.iloc[:, 1:], New_ext_3.iloc[:, 0])
    result_ext3 = get_final_result(threshold, prob_ext3, New_ext_3.iloc[:, 0])
    # print('===========================================================================')
    col_idx = json.load(open('../data_process/col_idx.json'))
    new_idx = {'Transfusion':'Transfusion_1', 'Preoperatively Gamma-glutamyl transferase':'E_GGT',
            'Preoperatively Total Bile Acids':'E_TBA', 'First postoperative Gamma-glutamyl transferase': 'D1_GGT',
            'First postoperative Total Bile Acids':'D1_TBA'}
    col_idx.update(new_idx)
    json.dump(col_idx, open('../data_process/col_idx.json', 'w'))
    '''
    分阶段储存结果
    '''
    print('===========================================================================')
    print(threshold)
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






