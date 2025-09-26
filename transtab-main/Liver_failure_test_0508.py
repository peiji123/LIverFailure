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
over_M = 'ro'


def oversampling(over_M, X_train, y_train):
    ov_samer = None
    if over_M == 'smo':
        ov_samer = SMOTE(random_state=42)
    elif over_M == 'b_smo':
        ov_samer = BorderlineSMOTE(random_state=42)
    elif over_M == 'ro':
        ov_samer = RandomOverSampler(random_state=42)
    elif over_M == 'ana':
        ov_samer = ADASYN(random_state=42)
    X_ov_data, y_ov_data = ov_samer.fit_resample(X_train, y_train)

    return X_ov_data, y_ov_data


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

df_train_valid = pd.read_csv('../data_process/TransTab_dataset/Time1_PHLF_0508_train.csv')
df_train_valid = df_train_valid.iloc[:,1:]
df_train, df_valid = train_test_split(df_train_valid, test_size=0.2)
df_test = pd.read_csv('../data_process/TransTab_dataset/Time1_PHLF_0508_test.csv')
df_test = df_test.iloc[:,1:]




X_train, y_train = X_y_split(df_train, info['target'][0])
X_valid, y_valid = X_y_split(df_valid, info['target'][0])
X_test, y_test = X_y_split(df_test, info['target'][0])


# X_train, y_train = oversampling(over_M, X_train, y_train)

trainset = [X_train, y_train]
valset = [X_valid, y_valid]
testset = [X_test, y_test]

df_train_valid2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_0508_train.csv')
df_train_valid2 = df_train_valid2.iloc[:,1:]
df_train2, df_valid2 = train_test_split(df_train_valid2, test_size=0.2)
df_test2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_0508_test.csv')
df_test2 = df_test2.iloc[:,1:]

X_train2, y_train2 = X_y_split(df_train2, info['target'][0])
X_valid2, y_valid2 = X_y_split(df_valid2, info['target'][0])
X_test2, y_test2 = X_y_split(df_test2, info['target'][0])

# X_train2, y_train2 = oversampling(over_M, X_train2, y_train2)

cat_cols = info['cate_cols']
# for col in ['S1', 'S2', 'S3', 'S4a', 'S4b', 'S5', 'S6', 'S7', 'S8']:
#     cat_cols.remove(col)
#
# cat_cols += ['Caudate lobe', 'Left lateral superior segment',
#        'Left lateral inferior segment', 'Left medial superior segment',
#        ' Left medial inferior segment', 'Right anterior inferior segment',
#        'Right posterior inferior segment', 'Right posterior superior segment',
#        'Right anterior superior segment']

# num_cols = info['bas_num'] + info['E_num'] + info['D1_num']
num_cols = info['cont_cols']

bin_cols = info['bin_cols']


trainset2 = [X_train2, y_train2]
valset2 = [X_valid2, y_valid2]
testset2 = [X_test2, y_test2]


# trainset = [ trainset, trainset]
# valset = [valset, valset]
# testset = [testset, testset]

trainset = [ trainset2, trainset]
valset = [valset2, valset]
testset = [testset2, testset]
# print("cuda:0" if torch.cuda.is_available() else "cpu")
model = transtab.build_classifier(cat_cols, num_cols, bin_cols, device='cuda:0')

training_arguments = {
    'num_epoch':200,
    'batch_size':256,
    'lr':2e-4,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':'./checkpoint/LF_bio_240524_12'
}

transtab.train(model, trainset, valset, **training_arguments)

x_test, y_test = testset[0]
print(x_test.shape)
ypred = transtab.predict(model, x_test, y_test)

result = get_final_result(y_test, ypred)

transtab.evaluate(ypred, y_test, seed=123, metric='auc')
print(transtab.evaluate(ypred, y_test, seed=123, metric='auc'))


# =====================================================================
# explainer = shap.Explainer(transtab.predict(model, x_test, y_test))
# tokenizer = BertTokenizerFast.from_pretrained('./transtab/tokenizer')
# x_train, _ = trainset[0]
# def shap_predict(data):
#     return transtab.predict_fun(model, data)
#
# x_train_clean = x_train.dropna()
# x_test_clean = x_test.dropna()
#
# explainer = shap.KernelExplainer(model=shap_predict, data=x_test_clean)
# shap_values = explainer.shap_values(x_test_clean.iloc[:100,:], nsamples=80)
# print(shap_values.shape)
# # shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0,:],x_test_clean.iloc[:1,:])
# shap.summary_plot(shap_values,x_test_clean.iloc[:100,:])

# explainer = shap.GradientExplainer(transtab.predict_fun(model, x_test_clean), model.input_encoder.feature_extractor(cat_cols, num_cols, bin_cols))
# shap_values = explainer(x_test_clean)
# explainer = shap.GradientExplainer()
# shap.plots.text(shap_values)