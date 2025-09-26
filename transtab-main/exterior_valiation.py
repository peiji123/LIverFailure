import transtab
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN,BorderlineSMOTE




model_path = './checkpoint/LF_bio_240531_06'
# '../data_process/exterior_info.json', '../data_process/info_0508.json'
# '../data_process/DropLiverseg_info.json'
info_path = '../data_process/exterior_info.json'
# '../data_process/TransTab_dataset/Time2_PHLF_ext_test.csv', '../data_process/TransTab_dataset/Time2_PHLF_0508_test.csv',
# '../data_process/TransTab_dataset/Time2_PHLF_DropLiver_0508_test.csv'
test_path = '../data_process/TransTab_dataset/Time2_PHLF_ext_test.csv'
# 外部验证2024.5.16.xlsx， PHLF508.xlsx
original_path = '../data_process/外部验证2024.5.16.xlsx'
error_path = '../data_process/外部验证2024.5.16.xlsx_error.csv'
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

# info = json.load(open('../data_process/exterior_info.json'))
# info = json.load(open('../data_process/info_0508.json'))
info = json.load(open(info_path))

# df_test2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_ext_test.csv')
# df_test2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_0508_test.csv')
df_test2 = pd.read_csv(test_path)

df_test2.index = df_test2.iloc[:,0]
df_test2 = df_test2.iloc[:,1:]

X_test2, y_test2 = X_y_split(df_test2, info['target'][0])


for col in info['bin_cols']:
    idx = ~X_test2[col].isnull()
    X_test2 = X_test2.loc[idx,:]
    y_test2 = y_test2.loc[idx]
X_test2[info['bin_cols']] = X_test2[info['bin_cols']].astype(int)
# X_test2 = X_test2.drop(['number of liver segmentectomies'],axis=1)

model = transtab.build_classifier(checkpoint=model_path,device='cuda:0')

ypred = transtab.predict(model, X_test2, y_test2)
result = get_final_result(y_test2, ypred)

y_pred = ypred
for idx, value in enumerate(ypred):
    # print(idx, value)
    if value > 0.5:
        y_pred[idx] = 1
    else:
        y_pred[idx] = 0

error_samples = X_test2.loc[y_pred != y_test2,:]
error_idx = error_samples.index
print(error_idx)

original_data = pd.read_excel(original_path)
original_error_data = original_data.loc[error_idx,:]
original_error_data.to_csv(error_path)


labels = ['Normal', 'Liver Failure']
cm = confusion_matrix(y_true=y_test2, y_pred=result)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels, )
disp.plot(cmap=plt.cm.Blues)
# disp.plot()
plt.show()
