import pandas as pd
import transtab
import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
from Result_process.metrics_process import metrics_with_youden
def get_final_result(thresholds, preds,label,):
    thresh = thresholds
    y_preds = (preds > thresh).astype(int)
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
    print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc),
          'sensitivity:','{:.3f}'.format(recall), 'specificity:','{:.3f}'.format(specificirty) )
    print('{:.3f}'.format(auc), '{:.3f}'.format(acc),
          '{:.3f}'.format(f1), '{:.3f}'.format(recall),'{:.3f}'.format(specificirty) )
    return y_preds


from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score

def get_final_result(thresholds, preds,label,):
    thresh = thresholds
    y_preds = (preds > thresh).astype(int)
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
    print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc),
          'sensitivity:','{:.3f}'.format(recall), 'specificity:','{:.3f}'.format(specificirty) )
    print('{:.3f}'.format(auc), '{:.3f}'.format(acc),
          '{:.3f}'.format(f1), '{:.3f}'.format(recall),'{:.3f}'.format(specificirty) )
    return y_preds


info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

path = 'LF_bio_240625_best_018'
threshold=0.6
model = transtab.build_classifier(cate_cols, num_cols, bin_cols)
model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+path)

df_alin_train = pd.read_csv('../data_process/df_int_train.csv', index_col=0).reset_index(drop=True)
df_alin_test = pd.read_csv('../data_process/df_int_test.csv', index_col=0).reset_index(drop=True)
df_alin_ext12 = pd.read_csv('../data_process/df_ext12.csv', index_col=0).reset_index(drop=True)
df_alin_ext12 = df_alin_ext12.drop([ 'intraoperative transfusion.1'], axis=1)
df_alin_ext3 = pd.read_csv('../data_process/New_ext_3_240625_best_018.csv', index_col=0).reset_index(drop=True)
df_alin_ext4 = pd.read_csv('../data_process/df_ext4.csv', index_col=0).reset_index(drop=True)
df_alin_ext4 = df_alin_ext4.drop([ 'anatomic liver resection'], axis=1)
df_PHLF_mimic = pd.read_csv('../data_process/df_PHLF_mimic_few_null.csv', index_col=0).reset_index(drop=True)
df_PHLF_ext6 = pd.read_csv('../data_process/df_PHLF_ext6.csv', index_col=0).reset_index(drop=True)
df_PHLF_ext7 = pd.read_csv('../data_process/df_PHLF_ext7.csv', index_col=0).reset_index(drop=True)

"""
内部测试：df_alin_test
外部测试
"""
df_sum_ext1 = pd.concat([df_alin_ext12, df_alin_ext3, df_alin_ext4])
# df_sum_ext1 = pd.concat([df_alin_ext12, df_alin_ext3, df_alin_ext4])
df_sum_ext2 = pd.concat([df_alin_ext12, df_alin_ext3, df_alin_ext4,df_PHLF_ext6, df_PHLF_ext7])
df_sum1 = pd.concat([df_alin_test, df_sum_ext1])
df_sum2 = pd.concat([df_alin_test, df_sum_ext2])

# df_sum_ext1 = df_sum2
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
        'Alpha-fetoprotein','Indocyanine Green Retention at 15 Minutes',  'Preoperatively Red Blood Cell Count',
    'Preoperatively Gamma-glutamyl transferase',  'Preoperatively Total Bile Acids',
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
         'intraoperative transfusion', 'Preoperatively Red Blood Cell Count',
    'Preoperatively Gamma-glutamyl transferase',  'Preoperatively Total Bile Acids',
         ]

int_intra_cols = list(set(int_preintra_cols)-set(int_pre_cols))

int_post_cols = list(set(list(df_sum1.columns))-set(int_preintra_cols))

# int_pre_PHLF = ['PHLF']+ int_pre_cols
int_intra_PHLF = ['PHLF']+ int_intra_cols
int_post_PHLF = ['PHLF']+ int_post_cols
int_intrapost_PHLF = ['PHLF'] + int_intra_cols + int_post_cols

int_pre_cols = [item for item in int_pre_cols if item != 'Indocyanine Green Retention at 15 Minutes']
int_preintra_cols = [item for item in int_preintra_cols if item != 'Indocyanine Green Retention at 15 Minutes']

df_sum_ext1_pre = df_sum_ext1.loc[:,int_pre_cols]
df_sum_ext1_intra = df_sum_ext1.loc[:,int_intra_PHLF]
df_sum_ext1_post = df_sum_ext1.loc[:,int_post_PHLF]
df_sum_ext1_preintra = df_sum_ext1.loc[:,int_preintra_cols]
df_sum_ext1_intrapost = df_sum_ext1.loc[:,int_intrapost_PHLF]

df_sum_ext1_pre = df_sum_ext1_pre.reset_index(drop=True)
df_sum_ext1_intra= df_sum_ext1_intra.reset_index(drop=True)
df_sum_ext1_post = df_sum_ext1_post.reset_index(drop=True)
df_sum_ext1_preintra = df_sum_ext1_preintra.reset_index(drop=True)
df_sum_ext1_intrapost = df_sum_ext1_intrapost.reset_index(drop=True)


print('====================================================================')
print('\n', 'Only preoperative variables')
prob_mimic = transtab.predict(model, df_sum_ext1_pre.iloc[:, 1:], df_sum_ext1_pre.iloc[:,0])
result_mimic = get_final_result(threshold,prob_mimic, df_sum_ext1_pre.iloc[:,0])
print('\n', 'Only intraoperative variables')
prob_intra = transtab.predict(model, df_sum_ext1_intra.iloc[:, 1:], df_sum_ext1_intra.iloc[:,0])
result_mimic = get_final_result(threshold,prob_intra, df_sum_ext1_intra.iloc[:,0])
print('\n', 'Only postoperative variables')
prob_post = transtab.predict(model, df_sum_ext1_post.iloc[:, 1:], df_sum_ext1_post.iloc[:,0])
result_mimic = get_final_result(threshold,prob_post, df_sum_ext1_post.iloc[:,0])
print('\n', 'The preoperative & intraoperative variables')
prob_preintra = transtab.predict(model, df_sum_ext1_preintra.iloc[:, 1:], df_sum_ext1_preintra.iloc[:,0])
result_mimic = get_final_result(threshold,prob_preintra, df_sum_ext1_preintra.iloc[:,0])
print('\n', 'The intraoperative & postoperative variables')
prob_intrapost = transtab.predict(model, df_sum_ext1_intrapost.iloc[:, 1:], df_sum_ext1_intrapost.iloc[:,0])
result_mimic = get_final_result(threshold,prob_intrapost, df_sum_ext1_intrapost.iloc[:,0])
print('\n','====================================================================')

df_sum1_pre = df_sum1.loc[:,int_pre_cols]
df_sum1_intra = df_sum1.loc[:,int_intra_PHLF]
df_sum1_post = df_sum1.loc[:,int_post_PHLF]
df_sum1_preintra = df_sum1.loc[:,int_preintra_cols]
df_sum1_intrapost = df_sum1.loc[:,int_intrapost_PHLF]

df_sum1_pre = df_sum1_pre.reset_index(drop=True)
df_sum1_intra= df_sum1_intra.reset_index(drop=True)
df_sum1_post = df_sum1_post.reset_index(drop=True)
df_sum1_preintra = df_sum1_preintra.reset_index(drop=True)
df_sum1_intrapost = df_sum1_intrapost.reset_index(drop=True)


print('====================================================================')
print('\n', 'Only preoperative variables')
prob_mimic = transtab.predict(model, df_sum1_pre.iloc[:, 1:], df_sum1_pre.iloc[:,0])
result_mimic = get_final_result(threshold,prob_mimic, df_sum1_pre.iloc[:,0])
print('\n', 'Only intraoperative variables')
prob_intra = transtab.predict(model, df_sum1_intra.iloc[:, 1:], df_sum1_intra.iloc[:,0])
result_mimic = get_final_result(threshold,prob_intra, df_sum1_intra.iloc[:,0])
print('\n', 'Only postoperative variables')
prob_post = transtab.predict(model, df_sum1_post.iloc[:, 1:], df_sum1_post.iloc[:,0])
result_mimic = get_final_result(threshold,prob_post, df_sum1_post.iloc[:,0])
print('\n', 'The preoperative & intraoperative variables')
prob_preintra = transtab.predict(model, df_sum1_preintra.iloc[:, 1:], df_sum1_preintra.iloc[:,0])
result_mimic = get_final_result(threshold,prob_preintra, df_sum1_preintra.iloc[:,0])
print('\n', 'The intraoperative & postoperative variables')
prob_intrapost = transtab.predict(model, df_sum1_intrapost.iloc[:, 1:], df_sum1_intrapost.iloc[:,0])
result_mimic = get_final_result(threshold,prob_intrapost, df_sum1_intrapost.iloc[:,0])
print('\n','====================================================================')

df_sum1_pre = df_sum2.loc[:,int_pre_cols]
df_sum1_intra = df_sum2.loc[:,int_intra_PHLF]
df_sum1_post = df_sum2.loc[:,int_post_PHLF]
df_sum1_preintra = df_sum2.loc[:,int_preintra_cols]
df_sum1_intrapost = df_sum2.loc[:,int_intrapost_PHLF]

df_sum1_pre = df_sum1_pre.reset_index(drop=True)
df_sum1_intra= df_sum1_intra.reset_index(drop=True)
df_sum1_post = df_sum1_post.reset_index(drop=True)
df_sum1_preintra = df_sum1_preintra.reset_index(drop=True)
df_sum1_intrapost = df_sum1_intrapost.reset_index(drop=True)


print('====================================================================')
print('\n', 'Only preoperative variables')
prob_pre = transtab.predict(model, df_sum1_pre.iloc[:, 1:], df_sum1_pre.iloc[:,0])
result_mimic = metrics_with_youden(df_sum1_pre.iloc[:,0],prob_pre)
print('\n', 'Only intraoperative variables')
prob_intra = transtab.predict(model, df_sum1_intra.iloc[:, 1:], df_sum1_intra.iloc[:,0])
result_mimic = metrics_with_youden(df_sum1_intra.iloc[:,0], prob_intra)
print('\n', 'Only postoperative variables')
prob_post = transtab.predict(model, df_sum1_post.iloc[:, 1:], df_sum1_post.iloc[:,0])
result_mimic = metrics_with_youden(df_sum1_post.iloc[:,0], prob_post)
print('\n', 'The preoperative & intraoperative variables')
prob_preintra = transtab.predict(model, df_sum1_preintra.iloc[:, 1:], df_sum1_preintra.iloc[:,0])
result_mimic = metrics_with_youden(df_sum1_preintra.iloc[:,0], prob_preintra)
print('\n', 'The intraoperative & postoperative variables')
prob_intrapost = transtab.predict(model, df_sum1_intrapost.iloc[:, 1:], df_sum1_intrapost.iloc[:,0])
result_mimic = metrics_with_youden(df_sum1_intrapost.iloc[:,0], prob_intrapost)
print('\n','====================================================================')