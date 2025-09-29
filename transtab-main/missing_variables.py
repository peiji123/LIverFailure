'''
计算缺失变量情况下 AUC的表现
'''
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
    print(con)
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
threshold=0.91
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
df_sum3 = pd.concat([df_PHLF_mimic, df_PHLF_ext6, df_PHLF_ext7])

com_2_sim = json.load(open('../data_process/col_idx.json'))
miss_ratio = 0
miss_ratio_list = [0, 0.1, 0.2, 0.3, 0.4]
for ratio in miss_ratio_list:
    df_miss = df_sum2.copy()
    n_missing = int(df_miss.shape[1]*ratio)
    # 4(40%: AUC-0.878), 6(40%: AUC-0.831), 15(40%: AUC-0.789), 29(40%: AUC-0.804), 30(40%: AUC-0.806), 39(40%: AUC-0.812)
    # 47(40%: AUC-0.871), 61(40%: AUC-0.877), 81(40%: AUC-0.856), 98(40%: AUC-0.725)
    missing_indices = np.random.default_rng(29).choice(df_miss.columns[1:], size=n_missing, replace=False)
    missing_var_names = [com_2_sim[str(idx)] for idx in missing_indices if str(idx) in com_2_sim]
    # print(missing_var_names)
    # print(len(missing_var_names))

    df_miss.drop(missing_indices, axis=1, inplace=True)
    df_miss = df_miss.reset_index(drop=True)
    print('The result of the variables with the missing ratio of '+str(ratio)+' is '+str(missing_var_names))
    prob_mimic = transtab.predict(model, df_miss.iloc[:, 1:], df_miss.iloc[:, 0])
    result_mimic = metrics_with_youden(df_miss.iloc[:, 0], prob_mimic)
    print('\n')
#
# prob_mimic = transtab.predict(model, df_alin_ext3.iloc[:, 7:], df_alin_ext3.iloc[:, 0])
# result_mimic = get_final_result(threshold,prob_mimic, df_alin_ext3.iloc[:, 0])

'''
============================================================================================
'''
top_shap_10 = ['D1_PT-INR', 'S_Number', 'HBVs Ag', 'Major', 'Cirrhosis',
       'Operation_time', 'E_PT-INR', 'Bleeding', 'Methods', 'Liver_Cancer']
top_shap_20 = ['D1_PT-INR', 'S_Number', 'HBVs Ag', 'Major', 'Cirrhosis',
       'Operation_time', 'E_PT-INR', 'Bleeding', 'Methods', 'Liver_Cancer',
       'Transfusion', 'Fatty_liver', 'Gender', 'Ascites', 'ICGR15', 'D1_ALT',
       'E_ALB', 'Age', 'D1_AST', 'BMI']
top_shap_30 = ['D1_PT-INR', 'S_Number', 'HBVs Ag', 'Major', 'Cirrhosis',
       'Operation_time', 'E_PT-INR', 'Bleeding', 'Methods', 'Liver_Cancer',
       'Transfusion', 'Fatty_liver', 'Gender', 'Ascites', 'ICGR15', 'D1_ALT',
       'E_ALB', 'Age', 'D1_AST', 'BMI', 'D1_TBIL', 'E_RBC', 'Diabetes',
       'E_ALT', 'E_PLT', 'E_AST', 'E_TP', 'D1_ALB', 'D1_RBC', 'E_LY']

sim_2_com = {v: k for k, v in com_2_sim.items()}
top_shap_10_com = [sim_2_com[sim_vari] for sim_vari in top_shap_10]
top_shap_20_com = [sim_2_com[sim_vari] for sim_vari in top_shap_20]
top_shap_30_com = [sim_2_com[sim_vari] for sim_vari in top_shap_30]

top_shap_10_com += ['PHLF']
top_shap_20_com += ['PHLF']
top_shap_30_com += ['PHLF']

# top_shap_9_shap = list(set(top_shap_10_com)-set(['Indocyanine Green Retention at 15 Minutes']))
top_shap_19_shap = [item for item in top_shap_20_com if item != 'Indocyanine Green Retention at 15 Minutes']
top_shap_29_shap = [item for item in top_shap_30_com if item != 'Indocyanine Green Retention at 15 Minutes']

data = df_sum1

df_sum_ext1_shap_10 = df_sum_ext1.loc[:, top_shap_10_com]
df_sum1_shap_10 = df_sum2.loc[:, top_shap_10_com]
df_sum_ext1_shap_20 = df_sum_ext1.loc[:, top_shap_19_shap]
df_sum1_shap_20 = df_sum2.loc[:, top_shap_20_com]
df_sum_ext1_shap_30 = df_sum_ext1.loc[:, top_shap_29_shap]
df_sum1_shap_30 = df_sum2.loc[:, top_shap_30_com]

df_sum_ext1_shap_10 = df_sum_ext1_shap_10.reset_index(drop=True)
df_sum_ext1_shap_20 = df_sum_ext1_shap_20.reset_index(drop=True)
df_sum_ext1_shap_30 = df_sum_ext1_shap_30.reset_index(drop=True)

df_sum1_shap_10 = df_sum1_shap_10.reset_index(drop=True)
df_sum1_shap_20 = df_sum1_shap_20.reset_index(drop=True)
df_sum1_shap_30 = df_sum1_shap_30.reset_index(drop=True)

# print("===================================================================================")
# print('\n','The results of the top 10 variables, which is defined by SHAP. Reporting in the External dataset')
# # print(top_shap_10_com)
# prob_mimic = transtab.predict(model, df_sum_ext1_shap_10.iloc[:, :-1], df_sum_ext1_shap_10.iloc[:, -1])
# result_mimic = get_final_result(threshold,prob_mimic, df_sum_ext1_shap_10.iloc[:, -1])
# print('\n','The results of the top 20 variables, which is defined by SHAP. Reporting in the External dataset')
# prob_mimic = transtab.predict(model, df_sum_ext1_shap_20.iloc[:, :-1], df_sum_ext1_shap_20.iloc[:, -1])
# result_mimic = get_final_result(threshold,prob_mimic, df_sum_ext1_shap_20.iloc[:, -1])
# print('\n','The results of the top 30 variables, which is defined by SHAP. Reporting in the External dataset')
# prob_mimic = transtab.predict(model, df_sum_ext1_shap_30.iloc[:, :-1], df_sum_ext1_shap_30.iloc[:, -1])
# result_mimic = get_final_result(threshold,prob_mimic, df_sum_ext1_shap_30.iloc[:, -1])

print("===================================================================================")
print('\n','The results of the top 10 variables, which is defined by SHAP. Reporting in the Internal & External dataset')
# print(top_shap_10_com)
prob_mimic = transtab.predict(model, df_sum1_shap_10.iloc[:, :-1], df_sum1_shap_10.iloc[:, -1])
result_mimic = metrics_with_youden(df_sum1_shap_10.iloc[:, -1], prob_mimic)
print('\n','The results of the top 20 variables, which is defined by SHAP. Reporting in the Internal & External dataset')
prob_mimic = transtab.predict(model, df_sum1_shap_20.iloc[:, :-1], df_sum1_shap_20.iloc[:, -1])
result_mimic = metrics_with_youden(df_sum1_shap_20.iloc[:, -1], prob_mimic)
print('\n','The results of the top 30 variables, which is defined by SHAP. Reporting in the Internal & External dataset')
prob_mimic = transtab.predict(model, df_sum1_shap_30.iloc[:, :-1], df_sum1_shap_30.iloc[:, -1])
result_mimic = metrics_with_youden(df_sum1_shap_30.iloc[:, -1], prob_mimic)
