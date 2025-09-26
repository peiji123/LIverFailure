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
df_test_ext3 = pd.read_csv('../data_process/df_ext3.csv')
df_test_ext3 = df_test_ext3.iloc[:,1:]
df_test_ext3 = df_test_ext3.drop(['duration of hepatic pedicle clamping'], axis=1)

df_test_ext3 = df_test_ext3.dropna()
# df_train_valid = df_train_valid.loc[:, df_test_ext12.columns]
# df_test = df_test.loc[:, df_test_ext12.columns]

df_train, df_valid = train_test_split(df_train_valid, test_size=0.1)

X_train, y_train = X_y_split(df_train, info['target'][0])
X_valid, y_valid = X_y_split(df_valid, info['target'][0])
X_test, y_test = X_y_split(df_test, info['target'][0])
X_test_ext12, y_test_ext12 = X_y_split(df_test_ext12, info['target'][0])
X_test_ext3, y_test_ext3 = X_y_split(df_test_ext3, info['target'][0])

trainset = [X_train, y_train]
valset = [X_valid, y_valid]
testset = [X_test, y_test]

cat_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

model = transtab.build_classifier(cat_cols, num_cols, bin_cols, device='cuda:0')

training_arguments = {
    'num_epoch':200,
    'batch_size':256,
    'lr':2e-4,
    'output_dir':'./checkpoint/LF_bio_240512_0'
}

transtab.train(model, trainset, valset, **training_arguments)

ypred = transtab.predict(model, X_test, y_test)
result = get_final_result(y_test, ypred)

ypred = transtab.predict(model, X_test_ext12, y_test_ext12)
result = get_final_result(y_test_ext12, ypred)

ypred = transtab.predict(model, X_test_ext3, y_test_ext3)
result = get_final_result(y_test_ext3, ypred)


