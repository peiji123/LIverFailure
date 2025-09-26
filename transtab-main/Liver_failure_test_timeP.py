import transtab
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN,BorderlineSMOTE
import torch
# print(torch.cuda.is_available(), torch.cuda.device_count())
import shap
from transtab.modeling_transtab import TransTabClassifier, TransTabFeatureExtractor, TransTabFeatureProcessor
from transformers import BertTokenizer, BertTokenizerFast, GPT2TokenizerFast, AutoTokenizer
from imblearn.under_sampling import RandomUnderSampler
import datetime
# ro, ru, smo, b_smo, ana
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

info = json.load(open('../data_process/info_0603.json'))

df_train_valid = pd.read_csv('../data_process/df_int_train.csv')
df_train_valid = df_train_valid.iloc[:,1:]
df_test = pd.read_csv('../data_process/df_int_test.csv')
df_test = df_test.iloc[:,1:]
df_test_ext12 = pd.read_csv('../data_process/df_ext12.csv')
df_test_ext12 = df_test_ext12.iloc[:,1:]
df_test_ext3 = pd.read_csv('../data_process/New_0702_ext_3.csv')
df_test_ext3 = df_test_ext3.iloc[:,1:]
df_test_ext3 = df_test_ext3.drop(['duration of hepatic pedicle clamping'], axis=1)
df_alin_ext4 = pd.read_csv('../data_process/df_ext4.csv')
# df_test_ext3 = df_test_ext3.dropna()
# df_train_valid = df_train_valid.loc[:, df_test_ext12.columns]
# df_test = df_test.loc[:, df_test_ext12.columns]

df_train, df_valid = train_test_split(df_train_valid, test_size=0.1)

int_pre_cols = [
'PHLF','Gender', 'Age', 'BMI', 'Hepatitis B Virus Surface Antigen', 'Hepatitis C Virus',
             'Fatty liver', 'Hypertension', 'Diabetes', 'Portal hypertension',
             'Cirrhosis', 'Liver Cancer',
    'Preoperatively Neutrophils',
       'Preoperatively Prothrombin Time International Normalized Ratio',
       'Preoperatively Potassium', 'Preoperatively Total Protein',
       'Preoperatively Alanine Aminotransferase', 'Preoperatively Hemoglobin',
       'Preoperatively Lymphocytes', 'Preoperatively Total Bilirubin',
       'Preoperatively Albumin', 'Preoperatively Creatinine',
       'Preoperatively White Blood Cell Count', 'Preoperatively Sodium',
       'Preoperatively Aspartate Aminotransferase',
       'Preoperatively Platelet Count','Tumor size','Tumor number','Ascites',
        'Alpha-fetoprotein','Indocyanine Green Retention at 15 Minutes',
                ]
int_preintra_cols = [
'PHLF',
'Gender', 'Age', 'BMI', 'Hepatitis B Virus Surface Antigen', 'Hepatitis C Virus',
             'Fatty liver', 'Hypertension', 'Diabetes', 'Portal hypertension',
             'Cirrhosis', 'Liver Cancer',
    'Preoperatively Neutrophils',
       'Preoperatively Prothrombin Time International Normalized Ratio',
       'Preoperatively Potassium', 'Preoperatively Total Protein',
       'Preoperatively Alanine Aminotransferase', 'Preoperatively Hemoglobin',
       'Preoperatively Lymphocytes', 'Preoperatively Total Bilirubin',
       'Preoperatively Albumin', 'Preoperatively Creatinine',
       'Preoperatively White Blood Cell Count', 'Preoperatively Sodium',
       'Preoperatively Aspartate Aminotransferase',
       'Preoperatively Platelet Count','Tumor size','Tumor number','Ascites',
        'Alpha-fetoprotein','Indocyanine Green Retention at 15 Minutes',
         'Methods', 'anatomic liver resection', 'extensive liver resection',
         'number of liver segmentectomies', 'duration of hepatic pedicle clamping',
         'Operation time', 'intraoperative bleeding',
         'intraoperative transfusion'
         ]

ext12_pre_cols = list(set(int_pre_cols)-set(['Indocyanine Green Retention at 15 Minutes']))
ext12_preintra_cols =list(set(int_preintra_cols)-set(['Indocyanine Green Retention at 15 Minutes','anatomic liver resection']))

# add_cold = list(set(df_test_ext3.columns)-set(df_test.columns))
ext3_pre_cols = ext12_pre_cols + ['Preoperatively Gamma-glutamyl transferase',  'Preoperatively Total Bile Acids']
ext3_preintra_cols = ext12_preintra_cols + ['Preoperatively Gamma-glutamyl transferase',  'Preoperatively Total Bile Acids']
ext3_preintra_cols = list(set(ext3_preintra_cols)-set(['duration of hepatic pedicle clamping']))


Pre_df_train = df_train[int_pre_cols]
preintra_df_train = df_train[int_preintra_cols]
Pre_df_valid = df_valid[int_pre_cols]
preintra_df_valid = df_valid[int_preintra_cols]
pre_df_test = df_test[int_pre_cols]
preintra_df_test = df_test[int_preintra_cols]
pre_df_ext12 = df_test_ext12[ext12_pre_cols]
preintra_df_ext12 = df_test_ext12[ext12_preintra_cols]
pre_df_ext3 = df_test_ext3[ext3_pre_cols]
preintra_df_ext3 = df_test_ext3[ext3_preintra_cols]
pre_df_ext4 = df_alin_ext4[ext12_pre_cols]
preintra_df_ext4 = df_alin_ext4[ext12_preintra_cols]

X_pre_train, y_pre_train = X_y_split(Pre_df_train, info['target'][0])
X_preintra_train, y_preintra_train = X_y_split(preintra_df_train, info['target'][0])
X_pre_valid, y_pre_valid = X_y_split(Pre_df_valid, info['target'][0])
X_preintra_valid, y_preintra_valid = X_y_split(preintra_df_valid, info['target'][0])
X_pre_test, y_pre_test = X_y_split(pre_df_test, info['target'][0])
X_preintra_test, y_preintra_test = X_y_split(preintra_df_test, info['target'][0])
X_pre_ext12, y_pre_ext12 = X_y_split(pre_df_ext12, info['target'][0])
X_preintra_ext12, y_preintra_ext12 = X_y_split(preintra_df_ext12, info['target'][0])
X_pre_ext3, y_pre_ext3 = X_y_split(pre_df_ext3, info['target'][0])
X_preintra_ext3, y_preintra_ext3 = X_y_split(preintra_df_ext3, info['target'][0])
X_pre_ext4, y_pre_ext4 = X_y_split(pre_df_ext4, info['target'][0])
X_preintra_ext4, y_preintra_ext4 = X_y_split(preintra_df_ext4, info['target'][0])

pre_trainset = [X_pre_train, y_pre_train]
preintra_trainset = [X_preintra_train, y_preintra_train]
pre_validset = [X_pre_valid, y_pre_valid]
preintra_validset = [X_preintra_valid, y_preintra_valid]
pre_testset = [X_pre_test, y_pre_test]
preintra_testset = [X_preintra_test, y_preintra_test]


# trainset = [X_train, y_train]
# valset = [X_valid, y_valid]
# testset = [X_test, y_test]
pre_cate_cols = list(set(info['cate_cols']) - set(['Methods']))
preintra_cat_cols = info['cate_cols']
num_cols = info['cont_cols']
pre_num_cols = list(set(num_cols).intersection(set(int_pre_cols)))
preintra_num_cols = list(set(num_cols).intersection(set(int_preintra_cols)))
bin_cols = info['bin_cols']
pre_bin_cols = list(set(bin_cols).intersection(set(int_pre_cols)))
preintra_bin_cols = list(set(bin_cols).intersection(set(int_preintra_cols)))
pre_training_arguments = {
    'num_epoch':200,
    'batch_size':256,
    'lr':2e-4,
    'patience':10,
    'imb_weight':1,
    'output_dir':'./checkpoint/pre_only_03'
}
preintra_training_arguments = {
    'num_epoch':200,
    'batch_size':256,
    'lr':2e-4,
    'output_dir':'./checkpoint/pre_intra_02'
}
pre_model = transtab.build_classifier(pre_cate_cols, pre_num_cols, pre_bin_cols, device='cuda:0')
pre_model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/pre_only_03')
# transtab.train(pre_model, pre_trainset, pre_validset, **pre_training_arguments)

pre_ypred = transtab.predict(pre_model, X_pre_test, y_pre_test)
pre_result = get_final_result(y_pre_test, pre_ypred)

pre_ext12_ypred = transtab.predict(pre_model, X_pre_ext12, y_pre_ext12)
pre_result_ext12 = get_final_result(y_pre_ext12, pre_ext12_ypred)

pre_ext3_ypred = transtab.predict(pre_model, X_pre_ext3, y_pre_ext3)
pre_result_ext3 = get_final_result(y_pre_ext3, pre_ext3_ypred)

pre_ext4_ypred = transtab.predict(pre_model, X_pre_ext4, y_pre_ext4)
pre_result_ext4 = get_final_result(y_pre_ext4, pre_ext4_ypred)

preintra_model = transtab.build_classifier(preintra_cat_cols, preintra_num_cols,
                                           preintra_bin_cols, device='cuda:0')
preintra_model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/pre_intra_01')

# transtab.train(preintra_model, preintra_trainset, preintra_validset, **preintra_training_arguments)
preintra_ypred = transtab.predict(preintra_model, X_preintra_test, y_preintra_test)
preintra_result = get_final_result(y_preintra_test, preintra_ypred)

preintra_ext12_yred = transtab.predict(preintra_model, X_preintra_ext12, y_preintra_ext12)
preintra_result_ext12 = get_final_result(y_preintra_ext12, preintra_ext12_yred)

preintra_ext3_ypred = transtab.predict(preintra_model, X_preintra_ext3, y_preintra_ext3)
preintra_result_ext3 = get_final_result(y_preintra_ext3, preintra_ext3_ypred)

preintra_ext4_ypred = transtab.predict(preintra_model, X_preintra_ext4, y_preintra_ext4)
preintra_result_ext4 = get_final_result(y_preintra_ext4, preintra_ext4_ypred)
# ========================================================================================

# ypred = transtab.predict(model, X_test_ext12, y_test_ext12)
# result = get_final_result(y_test_ext12, ypred)
#
# ypred = transtab.predict(model, X_test_ext3, y_test_ext3)
# result = get_final_result(y_test_ext3, ypred)


