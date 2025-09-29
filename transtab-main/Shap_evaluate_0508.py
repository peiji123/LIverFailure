'''
0508编号的model shap value 获取原始代码的 
'''
import matplotlib.pyplot as plt
import datetime
import transtab
import pandas as pd
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

info = json.load(open('../data_process/info_0508.json'))

df_test2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_0508_test.csv')
df_test2 = df_test2.iloc[:,1:]
df_train2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_0508_train.csv')
df_train2 = df_train2.iloc[:,1:]

X_test2, y_test2 = X_y_split(df_test2, info['target'][0])
X_train2, y_train2 = X_y_split(df_train2, info['target'][0])

model = transtab.build_classifier(checkpoint='./checkpoint/LF_bio_240515_4',device='cuda:0')
# model = nn.DataParallel(model).to('cuda')

ypred = transtab.predict(model, X_test2, y_test2)

result = get_final_result(y_test2, ypred)

transtab.evaluate(ypred, y_test2, seed=123, metric='auc')
print(transtab.evaluate(ypred, y_test2, seed=123, metric='auc'))

col_idx = json.load(open('../data_process/col_idx.json'))
shap_values = np.load('./shap_values/'+str(inter)+'.npy')
new_cols = list(col_idx.values())
new_cols = new_cols[1:]

def shap_predict(data):
    return transtab.predict_fun(model, data, X_test2.columns)

x_train_clean = X_train2.dropna()
x_test_clean = X_test2.dropna()


# data = x_test_clean.iloc[:150,:]
data = x_test_clean.iloc[:100,:]

explainer = shap.KernelExplainer(model=shap_predict, data=data,)

# =================================================================
# nsamples=80
shap_values = explainer.shap_values(data, nsamples=80)
np.save('./shap_values/'+str(inter), shap_values)
print(shap_values.shape)
# =================================================================

data.columns = new_cols

summary_plot = shap.summary_plot(shap_values,data, show=False)

# fig = plt.gcf()
# fig.set_size_inches(16,16)
# shap.plots.heatmap(shap_values)
plt.savefig('./shap_values/240515_2.jpg')


