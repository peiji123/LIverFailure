import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
from holoviews.plotting.bokeh.styles import font_size
from transtab.analysis_utils import confusion_matrix
import transtab
import pandas as pd
import json
import pickle
import shap
from sklearn.metrics import accuracy_score, roc_auc_score
# from Confusion_and_AUC import ori_data_ext3
from analysis_utils import get_final_result, X_y_split, error_samples, confusion_matrix_plots
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
prop = fm.FontProperties(fname=font_path, size=22)
# from attention import base_cols

info = json.load(open('../data_process/info_0603.json'))
df_train2 = pd.read_csv('../data_process/df_int_train.csv')
df_test = pd.read_csv('../data_process/df_int_test.csv')
df_test = df_test.set_index('Unnamed: 0')
col_idx = json.load(open('../data_process/col_idx.json'))
col_idx.pop('Transfusion')
df_ext12 = pd.read_csv('../data_process/df_ext12.csv')
df_ext3 = pd.read_csv('../data_process/New_0702_ext_3_alin.csv')
df_ext4 = pd.read_csv('../data_process/df_ext4.csv')
df_Dr_results = pd.read_excel('../data_process/Doctor_User_result(1).xlsx')
df_Dr_results2 = pd.read_excel('../data_process/doctor_result.xlsx')
dict_sim_2_com = {v: k for k, v in col_idx.items()}

df_ext3 = df_ext3.rename(columns=dict_sim_2_com)
df_ext3['Gender'] = df_ext3['Gender'].map({1:'male', 0:'female'})
df_ext3['Methods'] = df_ext3['Methods'].map({0: 'Laparotomy', 1: 'Laparoscopic surgery',
                                             2: 'Transfer to laparotomy'})

ori_data = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
ori_data_train = ori_data.loc[df_train2.iloc[:,0],:]
ori_data_test = ori_data.loc[df_test.index,:]
ori_data_ext12 = pd.read_excel('../data_process/外部验证中心12-702.xlsx')
ori_data_int = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
ori_data_ext4 = pd.read_excel('../data_process/外部验证中心4.xlsx')
ori_data_ext3 = pd.read_csv('../data_process/ori_0702_new_ext_3.csv')

df_train2 = df_train2.iloc[:,1:]
# df_test = df_test.iloc[:,1:]
df_ext12 = df_ext12.iloc[:,1:]
df_ext3 = df_ext3.iloc[:,1:]
df_ext4 = df_ext4.iloc[:,1:]

com_cols = df_train2.columns
sim_cols = {}
for col in com_cols:
    # if col in
    sim_cols[col] = col_idx[col]
df_ext12 = df_ext12.drop(['intraoperative transfusion.1'], axis=1)
df_ext3 = df_ext3.rename(columns={'Transfusion_1':'intraoperative transfusion'})

pd_all_test = pd.concat([df_test, df_ext12, df_ext3, df_ext4])
pd_all_test_ext = pd.concat([df_ext12, df_ext3, df_ext4])
# pd_all_test = pd_all_test.drop(['intraoperative transfusion.1'],axis=1)

ori_data_test = ori_data_test.rename(columns={'Bleeding_A':'Bleeding'})
ori_data_ext3 = ori_data_ext3.rename(columns={'Transfusion_1':'Transfusion'})
ori_data_ext4 = ori_data_ext4.rename(columns={'Transfusion_1':'Transfusion'})

ori_all_test = pd.concat([ori_data_test, ori_data_ext12, ori_data_ext3, ori_data_ext4])
ori_all_test_ext = pd.concat([ori_data_ext12, ori_data_ext3, ori_data_ext4])

pd_all_test = pd_all_test.reset_index(drop=True)
ori_all_test = ori_all_test.reset_index(drop=True)

pd_all_test_ext = pd_all_test_ext.reset_index(drop=True)
ori_all_test_ext = ori_all_test_ext.reset_index(drop=True)

# 2| acc: 0.900 f1: 0.688 auc: 0.936 kappa: 0.632 MCC: 0.662 recall: 0.917 specificity: 0.898[[85  7]
#  [ 0  8]]
# acc: 0.930 f1: 0.696 auc: 0.970 kappa: 0.660 MCC: 0.702 recall: 1.000 specificity: 0.924
# 5| acc: 0.900 f1: 0.722 auc: 0.957 kappa: 0.665 MCC: 0.690 recall: 0.929 specificity: 0.895
# 6| acc: 0.910 f1: 0.667 auc: 0.963 kappa: 0.621 MCC: 0.671 recall: 1.000 specificity: 0.901
#
file_path = 'LF_bio_240625_best_018'
model = transtab.build_classifier(checkpoint='./checkpoint/'+file_path,device='cuda:0')
'''
输出所有的测试集，同时加上 y_pred
'''
# X_all_test, y_all_test = X_y_split(pd_all_test, info['target'][0])
# all_pred = transtab.predict(model, X_all_test, y_all_test)
# y_all_pred = get_final_result(y_all_test, all_pred)
# ori_all_test_pred = pd.concat([ pd.DataFrame(all_pred, columns=['Probalility']),
#                                 ori_all_test], axis=1)



selected_test = pd_all_test.sample(n=100, random_state=15)
selected_ori_data = ori_all_test.sample(n=100, random_state=15)

# 3: 0.921, 4: 0.946, 5:0.939, 6:0.95,
random_seed = 3
selected_test_ext = pd_all_test_ext.sample(n=100, random_state=random_seed)
selected_ori_data_ext = ori_all_test_ext.sample(n=100, random_state=random_seed)

selected_test = selected_test.reset_index(drop=True)
selected_ori_data = selected_ori_data.reset_index(drop=True)

selected_test_ext = selected_test_ext.reset_index(drop=True)
selected_ori_data_ext = selected_ori_data_ext.reset_index(drop=True)

selected_test.to_csv('./results/' + file_path + '/User_Exm/selected_test.csv')
selected_ori_data.to_csv('./results/' + file_path + '/User_Exm/selected_ori_data.csv')



selected_test_ext.to_csv('./results/' + file_path + '/User_Exm/selected_test_ext.csv')
selected_ori_data_ext.to_csv('./results/' + file_path + '/User_Exm/selected_ori_data_ext.csv')


X_selected, y_selected  = X_y_split(selected_test_ext, info['target'][0])

ypred = transtab.predict(model, X_selected, y_selected)
y_pred = get_final_result(0.6, y_selected, ypred)

print('=================================================================================')


prob_selected = pd.concat([pd.DataFrame(ypred,columns=['Pred']),X_selected],axis=1)
prob_selected.to_csv('./results/' + file_path + '/User_Exm/probability_selected.csv')
predicted_selected = pd.concat([pd.DataFrame(y_pred,columns=['Pred']),X_selected],axis=1)
predicted_ori_selected = pd.concat([pd.DataFrame(y_pred,columns=['Pred']),selected_ori_data],axis=1)
predicted_selected.to_csv('./results/' + file_path + '/User_Exm/predicted_selected.csv')
predicted_ori_selected.to_csv('./results/' + file_path + '/User_Exm/predicted_ori_selected.csv')
prob_ori_selected = pd.concat([pd.DataFrame(ypred,columns=['Pred']),selected_ori_data],axis=1)
prob_ori_selected.to_csv('./results/' + file_path + '/User_Exm/probability_ori_selected.csv')


User_Exm_ext = pd.read_excel('../data_process/人机对照2025.2.14.xlsx')




df_7_WK = pd.read_excel('../data_process/doctor_use+WK.xlsx', header=None)
df_7_WK_new = pd.read_excel('../data_process/doctor_use+WK+加了肝衰预测指数.xlsx', header=None)


df_7_WK.columns = df_7_WK.iloc[1]
df_7_WK = df_7_WK.iloc[2:,:]
df_7_WK = df_7_WK.sort_values(by='排序', ascending=True)
df_7_WK.set_index('排序', inplace=True)

df_7_WK_new.columns = df_7_WK_new.iloc[1]
df_7_WK_new = df_7_WK_new.iloc[2:,:]
df_7_WK_new = df_7_WK_new.sort_values(by='排序', ascending=True)
df_7_WK_new.set_index('排序', inplace=True)

ypred_series = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
incorrect_samples = X_selected.loc[y_selected != ypred_series, :]
incor_idx = list(incorrect_samples.index)
incor_idx = [x for x in incor_idx if x < 100]
cor_idx = list(set(list(range(100))) - set(incor_idx))

incorrect_samples.fillna(0, inplace=True)


col_idx = json.load(open('../data_process/col_idx.json'))



# =============================================================================

'''
简化特征名称
'''
for col in X_selected.columns:
    if col in col_idx.keys():
        X_selected= X_selected.rename(columns={col:col_idx[col]})
        # print(col)

X_ori_selected = selected_ori_data[X_selected.columns]
X_selected['Gender'] = X_selected['Gender'].map({1:'male', 0:'female'})
X_selected['Methods'] = X_selected['Methods'].map({0: 'Laparotomy', 1: 'Laparoscopic surgery',
                                             2: 'Transfer to laparotomy'})
idx2cat = {'Gender': [0, 1], 'Methods': [0, 1, 2]}

# force_plot = shap.force_plot(explainer.expected_value, shap_values[0], x_data.iloc[0,:],show=True)
# =============================================================================
'''
更改名称
'''
form_cols = {'Gender':'Gender', 'Methods':'Approach',
             'Major':'Major liver resection',
             'ALR':'ALR', 'Portal_hypertension':'Portal hypertension',
             'HBVs Ag':'HBV','Hypertension':'Hypertension',
             'HCV':'HCV', 'Cirrhosis':'Cirrhosis', 'Liver_Cancer':'Liver cancer',
             'Fatty_liver': 'Fatty liver', 'Ascites':'Ascites', 'Diabetes':'Diabetes',
             'Age':'Age', 'AFP':'AFP', 'ICGR15':'ICGR15',
             'S_Number':'Number of liver segments',
             'Pringle':'Pringle maneuver','BMI':'BMI', 'Operation_time':'Operation time',
             'Tumor_size':'Tumor size', 'Bleeding':'Intra bleeding',
             'Tumor_number':'Tumor number', 'Transfusion':'Intra blood transfusion',
             'E_RBC':'Preop RBC', 'E_NE':'Preop NE', 'E_PT-INR':'Preop PT-INR',
             'E_K':'Preop K', 'E_TP':'Preop TP', 'E_ALT':'Preop ALT',
             'E_HGB':'Preop HGB', 'E_LY':'Preop LY', 'E_TBIL':'Preop TBIL',
             'E_ALB':'Preop ALB', 'E_CR':'Preop CR', 'E_WBC':'Preop WBC',
             'E_Na':'Preop Na', 'E_AST': 'Preop AST', 'E_PLT':'Preop PLT',
             'D1_ALT':'Postop 24h ALT', 'D1_ALB':'Postop 24h ALB',
             'D1_AST':'Postop 24h AST', 'D1_WBC':'Postop 24h WBC',
             'D1_NE':'Postop 24h NE', 'D1_TBIL':'Postop 24h TBIL','D1_K':'Postop 24h K',
             'D1_HGB':'Postop 24h HGB', 'D1_RBC':'Postop 24h RBC',
             'D1_TP':'Postop 24h TP', 'D1_Na':'Postop 24h Na', 'D1_CR':'Postop 24h CR',
             'D1_PLT':'Postop 24h PLT','D1_LY':'Postop 24h LY', 'D1_PT-INR':'Postop 24h PT-INR'}
with open ('../data_process/simple2form_cols.json', 'w') as f:
    json.dump(form_cols,f)

X_selected = X_selected.rename(columns=form_cols)

# =============================================================================
'''
force plotplt.close()
'''
# shap.initjs()
# for i in incor_idx:
#     shap.force_plot(explainer.expected_value, shap_values[i], np.array(X_ori_selected.iloc[i]),
#                     list(X_selected.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
#     plt.savefig('./results/' + file_path + '/User_Exm/patient_SHAP/force_error'+str(i)+'.eps')
#     plt.close()
# for i in cor_idx:
#     shap.force_plot(explainer.expected_value, shap_values[i], np.array(X_ori_selected.iloc[i]),
#                     list(X_selected.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
#     plt.savefig('./results/' + file_path + '/User_Exm/patient_SHAP/force_correct_full'+str(i)+'.eps')
#     plt.close()

# =============================================================================
'''
人机实验AUC曲线图
'''
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, auc, f1_score, cohen_kappa_score, matthews_corrcoef
def TPR_FPR (y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    print(confusion_matrix(y_true, y_pred))
    print(tpr, fpr)
    return tpr, fpr
def get_final_result_2(label,y_preds, phy):
    conf_mar = confusion_matrix(label, y_preds)
    acc = accuracy_score(y_true=label, y_pred=y_preds)
    sensitivity = recall_score(y_true=label, y_pred=y_preds)
    f1 = f1_score(y_true=label, y_pred=y_preds, )
    kappa = cohen_kappa_score(y1=label, y2=y_preds)
    mcc = matthews_corrcoef(y_true=label, y_pred=y_preds)
    AUC = roc_auc_score(label, y_preds)
    TN = conf_mar[0, 0]
    FP = conf_mar[0, 1]
    specificity = TN/(TN+FP)
    con = confusion_matrix(label, y_preds)
    print(con)
    print('auc','{:.3f}'.format(AUC),'acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),
          'sensitivity:','{:.3f}'.format(sensitivity), 'specificity:','{:.3f}'.format(specificity),
          'kappa:','{:.3f}'.format(kappa),'mcc:','{:.3f}'.format(mcc))
    # return {'Phy':[phy],'auc':['{:.3f}'.format(AUC)],'acc:':['{:.3f}'.format(acc)],'f1:':['{:.3f}'.format(f1)],
    #       'sensitivity:':['{:.3f}'.format(sensitivity)], 'specificity:':['{:.3f}'.format(specificirty)],
    #       'kappa:':['{:.3f}'.format(kappa)],'mcc:':['{:.3f}'.format(mcc)]}
    return [phy, '{:.3f}'.format(AUC), '{:.3f}'.format(acc), '{:.3f}'.format(f1),
            '{:.3f}'.format(sensitivity), '{:.3f}'.format(specificity),
            '{:.3f}'.format(kappa), '{:.3f}'.format(mcc)]
    # return [[phy],['{:.3f}'.format(AUC)],['{:.3f}'.format(acc)],['{:.3f}'.format(f1)],
    #       ['{:.3f}'.format(sensitivity)], ['{:.3f}'.format(specificirty)],
    #       ['{:.3f}'.format(kappa)],['{:.3f}'.format(mcc)]]
y_selected = np.array(y_selected)
# Dr_1 = pd.read_excel('../data_process/24-09-07-ewsy.xlsx')
# Dr_1 = np.array(Dr_1).reshape(-1)

Dr_1_LZH = np.array(User_Exm_ext.loc[:,'noAI_LZH'])
Dr_1_LZH_AI = np.array(User_Exm_ext.loc[:,'AI_LZH'])
Dr_2_TSH = np.array(User_Exm_ext.loc[:,'noAI_TSH'])
Dr_2_TSH_AI = np.array(User_Exm_ext.loc[:,'AI_TSH'])
Dr_3_LHB = np.array(User_Exm_ext.loc[:,'noAI_LHB'])
Dr_3_LHB_AI = np.array(User_Exm_ext.loc[:,'AI_LHB'])
Dr_4_LZX = np.array(User_Exm_ext.loc[:,'noAI_LZX'])
Dr_4_LZX_AI = np.array(User_Exm_ext.loc[:,'AI_LZX'])
Dr_5_HPX = np.array(User_Exm_ext.loc[:,'noAI_HPX'])
Dr_5_HPX_AI = np.array(User_Exm_ext.loc[:,'AI_HPX'])
Dr_6_YT = np.array(User_Exm_ext.loc[:, 'noAI_YT'])
Dr_6_YT_AI = np.array(User_Exm_ext.loc[:, 'AI_YT'])
Dr_7_XHR = np.array(User_Exm_ext.loc[:, 'noAI_XHR'])
Dr_7_XHR_AI = np.array(User_Exm_ext.loc[:, 'AI_XHR'])
Dr_8_LLY = np.array(User_Exm_ext.loc[:, 'noAI_LLY'])
Dr_8_LLY_AI = np.array(User_Exm_ext.loc[:, 'AI_LLY'])

tpr_1, fpr_1 = TPR_FPR(y_selected,Dr_1_LZH)
tpr_1_AI, fpr_1_AI = TPR_FPR(y_selected,Dr_1_LZH_AI)
tpr_2, fpr_2 = TPR_FPR(y_selected,Dr_2_TSH)
tpr_2_AI, fpr_2_AI = TPR_FPR(y_selected,Dr_2_TSH_AI)
tpr_3, fpr_3 = TPR_FPR(y_selected,Dr_3_LHB)
tpr_3_AI, fpr_3_AI = TPR_FPR(y_selected,Dr_3_LHB_AI)
tpr_4, fpr_4 = TPR_FPR(y_selected,Dr_4_LZX)
tpr_4_AI, fpr_4_AI = TPR_FPR(y_selected,Dr_4_LZX_AI)
tpr_5, fpr_5 = TPR_FPR(y_selected,Dr_5_HPX)
tpr_5_AI, fpr_5_AI = TPR_FPR(y_selected,Dr_5_HPX_AI)
tpr_6, fpr_6 = TPR_FPR(y_selected, Dr_6_YT)
tpr_6_AI, fpr_6_AI = TPR_FPR(y_selected, Dr_6_YT_AI)
tpr_7, fpr_7 = TPR_FPR(y_selected, Dr_7_XHR)
tpr_7_AI, fpr_7_AI = TPR_FPR(y_selected, Dr_7_XHR_AI)
tpr_8, fpr_8 = TPR_FPR(y_selected, Dr_8_LLY)
tpr_8_AI, fpr_8_AI = TPR_FPR(y_selected, Dr_8_LLY_AI)

print('================LZH================')
get_final_result_2(y_selected, Dr_1_LZH, 'PhyA LZH')
get_final_result_2(y_selected, Dr_1_LZH_AI, 'PhyA LZH with AI')
print('================TSH================')
get_final_result_2(y_selected, Dr_2_TSH, 'PhyB TSH')
get_final_result_2(y_selected, Dr_2_TSH_AI, 'PhyB TSH with AI')
print('================LHB================')
get_final_result_2(y_selected, Dr_3_LHB, 'PhyB LHB')
get_final_result_2(y_selected, Dr_3_LHB_AI, 'PhyB LHB with AI')
print('================LZX================')
get_final_result_2(y_selected, Dr_4_LZX, 'PhyC LZX')
get_final_result_2(y_selected, Dr_4_LZX_AI, 'PhyC LZX with AI')
print('================HPX================')
get_final_result_2(y_selected, Dr_5_HPX, 'PhyD HPX')
get_final_result_2(y_selected, Dr_5_HPX_AI, 'PhyD HPX with AI')
print('================YT================')
get_final_result_2(y_selected, Dr_6_YT, 'PhyE YT')
get_final_result_2(y_selected, Dr_6_YT_AI, 'PhyE YT with AI')
print('================XHR================')
get_final_result_2(y_selected, Dr_7_XHR, 'PhyF XHR')
get_final_result_2(y_selected, Dr_7_XHR_AI, 'PhyF XHR with AI')
print('================LLY================')
get_final_result_2(y_selected, Dr_8_LLY, 'PhyG LLY')
get_final_result_2(y_selected, Dr_8_LLY_AI, 'PhyG LLY with AI')

fpr_list = [fpr_1, fpr_2, fpr_5, fpr_6, fpr_7, ]
tpr_list = [tpr_1, tpr_2, tpr_5, tpr_6, tpr_7, ]
fpr_list_AI = [fpr_1_AI, fpr_2_AI, fpr_5_AI, fpr_6_AI, fpr_7_AI, ]
tpr_list_AI = [tpr_1_AI, tpr_2_AI, tpr_5_AI, tpr_6_AI, tpr_7_AI, ]
# label_list = ['Physician A', 'Physician B', 'Physician C', 'Physician D', 'Physician E']
label_list = ['Phy. A', 'Phy. B', 'Phy. C', 'Phy. D', 'Phy. E']

Dr_list = [Dr_1_LZH, Dr_2_TSH, Dr_5_HPX, Dr_6_YT, Dr_7_XHR]
Dr_list_AI= [Dr_1_LZH_AI, Dr_2_TSH_AI, Dr_5_HPX_AI, Dr_6_YT_AI, Dr_7_XHR_AI]

path='LF_bio_240625_best_018'
basic_path = './results/'+path+'/User_Exm/'
# for label, Dr in zip(label_list, Dr_list):
#     cmp = confusion_matrix_plots(y_selected, Dr,basic_path+label+'external.eps')
#     # print( label)
#     er_sams, ori_sams, _ = error_samples(selected_test, selected_ori_data, Dr, y_selected)
#
#
# cmp = confusion_matrix_plots(y_selected, y_pred, basic_path+'external_Ours.eps')

fpr, tpr, thresholds = roc_curve(y_selected, ypred)
auc_score = auc(fpr, tpr)

def calculate_auc_approx(fpr, tpr):
    fpr_points = [0, fpr, 1]
    tpr_points = [0, tpr, 1]
    auc_approx = auc(fpr_points, tpr_points)
    return auc_approx

Phy_auc = []
Phy_auc += [calculate_auc_approx(fpr_1, tpr_1)]
Phy_auc += [calculate_auc_approx(fpr_2, tpr_2)]
Phy_auc += [calculate_auc_approx(fpr_5, tpr_5)]
Phy_auc += [calculate_auc_approx(fpr_6, tpr_6)]
Phy_auc += [calculate_auc_approx(fpr_7, tpr_7)]
# Phy_auc += [calculate_auc_approx(fpr_8, tpr_8)]

Phy_auc_new = []
Phy_auc_new += [calculate_auc_approx(fpr_1_AI, tpr_1_AI)]
Phy_auc_new += [calculate_auc_approx(fpr_2_AI, tpr_2_AI)]
Phy_auc_new += [calculate_auc_approx(fpr_5_AI, tpr_5_AI)]
Phy_auc_new += [calculate_auc_approx(fpr_6_AI, tpr_6_AI)]
Phy_auc_new += [calculate_auc_approx(fpr_7_AI, tpr_7_AI)]
# Phy_auc_new += [calculate_auc_approx(fpr_8_AI, tpr_8_AI)]

marker_shape = ['o', 's', 'p', '^', 'D']
plt.subplots(figsize=[11, 10])
plt.plot(fpr, tpr, color='#D24D3E', alpha=0.7,lw=3, label='AUC = %0.3f' %auc_score)
# plt.scatter(fpr_list, tpr_list, s=100, c= 'black')
# plt.scatter(fpr_list_new, tpr_list_new, s=100, c= 'b')
# for i, (x, y) in enumerate(zip(fpr_list, tpr_list)):
#     plt.scatter(x, y, s=100, c='black', marker=marker_shape[i], label=f'{label_list[i]} AUC-approx = {Phy_auc[i]:.3f}')
# for i, (x, y) in enumerate(zip(fpr_list_new, tpr_list_new)):
#     plt.scatter(x, y, s=100, c='b', marker=marker_shape[i], label=f'{label_list[i]} AUC-approx = {Phy_auc_new[i]:.3f}')
for i in range(len(fpr_list)):
    if i==0:
        plt.scatter(fpr_list[i], tpr_list[i], s=100, c='black', marker=marker_shape[i],
                    label=f'{label_list[i]}  Original AUC = {Phy_auc[i]:.3f}')
        plt.scatter(fpr_list_AI[i], tpr_list_AI[i], s=100, c='b', marker=marker_shape[i],
                    label=f'{label_list[i]}  AI-assisted AUC = {Phy_auc_new[i]:.3f}'+'\u2191')
    else:
        plt.scatter(fpr_list[i], tpr_list[i], s=100, c='black', marker=marker_shape[i],
                    label=f'{label_list[i]}  Original AUC = {Phy_auc[i]:.3f}')
        plt.scatter(fpr_list_AI[i], tpr_list_AI[i], s=100, c='b', marker=marker_shape[i],
                    label=f'{label_list[i]}  AI-assisted AUC = {Phy_auc_new[i]:.3f}'+'\u2191')

# 添加标签
for x, y, label in zip(fpr_list, tpr_list, label_list):
# for x,y, label in zip(fpr_list, tpr_list, label_list):
    if label == 'Phy. A':
        plt.text(x, y-0.05, label, ha='center', fontsize=22, fontproperties=prop)
    elif label == 'Phy. B':
        plt.text(x, y-0.05, label, ha='center', fontsize=22, fontproperties=prop)
    elif label == 'Phy. C':
        plt.text(x, y+0.02, label,ha='center', fontsize=22, fontproperties=prop)
    elif label == 'Phy. D':
        plt.text(x, y+0.03, label, ha='center', fontsize=22, fontproperties=prop)
    elif label == 'Phy. E':
        plt.text(x+0.07, y-0.01, label, ha='center', fontsize=22, fontproperties=prop)
    # elif label == 'Phy. F':
    #     plt.text(x, y+0.02, label, ha='center', fontsize=22, fontproperties=prop)

for x,y, label in zip(fpr_list_AI, tpr_list_AI, label_list):
    if label == 'Phy. A':
        plt.text(x, y+0.02, label, ha='center', fontsize=22,color='b', fontproperties=prop)
    elif label == 'Phy. B':
        plt.text(x, y+0.02, label,ha='center', fontsize=22,color='b', fontproperties=prop)
    elif label == 'Phy. C':
        plt.text(x+0.07, y-0.01, label, ha='center', fontsize=22, color='b', fontproperties=prop)
    elif label == 'Phy. D':
        plt.text(x, y+0.02, label, ha='center', fontsize=22,color='b', fontproperties=prop)
    elif label == 'Phy. E':
        plt.text(x+0.07, y-0.01, label, ha='center', fontsize=22,color='b', fontproperties=prop)
    # else:
    #     plt.text(x+0.07, y-0.01, label, ha='center', fontsize=22, color='b',fontproperties=prop)

label_colors = ['black', 'black', 'b', 'black', 'b', 'black', 'b', 'black', 'b', 'black', 'b', 'black', 'b']
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22, fontproperties=prop)
plt.yticks(fontsize=22, fontproperties=prop)
plt.xlabel('FPR',fontsize=24, fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.legend(loc='lower right', fontsize=22, prop=prop, labelcolor=label_colors)
plt.savefig('./results/' + file_path + '/User_Exm/AUC_external.eps')
plt.savefig('./results/' + file_path + '/User_Exm/AUC_external.png')
plt.show()
plt.close()

'''
绘制cutoff的AUC图
'''


'''
'auc','{:.3f}'.format(AUC),'acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),
          'sensitivity:','{:.3f}'.format(sensitivity), 'specificity:','{:.3f}'.format(specificirty),
          'kappa:','{:.3f}'.format(kappa),'mcc:','{:.3f}'.format(mcc)
'''
'''-

'''
Dr_result = pd.DataFrame(columns=['Phy', 'auc', 'acc', 'f1','sensitivity','specificity', 'kappa','mcc', ])
none_row = pd.DataFrame({'Phy': [''], 'auc': [''], 'acc': [''], 'f1': [''],'sensitivity': [''],'specificity': [''], 'kappa': [''],'mcc': ['']})

def Dr_result_append(Dr_result, Dr_pred, Dr_pred_new, label, name, name_new, none_row):
    df_Dr_result = pd.DataFrame([get_final_result_2(label, Dr_pred, name)], columns=Dr_result.columns)
    df_DR_result_new = pd.DataFrame([get_final_result_2(label, Dr_pred_new, name_new)], columns=Dr_result.columns)
    Dr_result = pd.concat([Dr_result, df_Dr_result], ignore_index=True)
    Dr_result = pd.concat([Dr_result, df_DR_result_new], ignore_index=True)
    Dr_result = pd.concat([Dr_result, none_row], ignore_index=True)
    return Dr_result

Dr_result = Dr_result_append(Dr_result, Dr_1_LZH, Dr_1_LZH_AI, y_selected,
                             'PhyA LZH', 'PhyA LZH with AI',none_row)
Dr_result = Dr_result_append(Dr_result, Dr_2_TSH, Dr_2_TSH_AI, y_selected,
                             'PhyB TSH', 'PhyB TSH with AI',none_row)
Dr_result = Dr_result_append(Dr_result, Dr_5_HPX, Dr_5_HPX_AI, y_selected,
                             'PhyC HPX', 'PhyC HPX with AI',none_row)
Dr_result = Dr_result_append(Dr_result, Dr_6_YT, Dr_6_YT_AI, y_selected,
                             'PhyD YT', 'PhyD YT with AI',none_row)
Dr_result = Dr_result_append(Dr_result, Dr_7_XHR, Dr_7_XHR_AI, y_selected,
                             'PhyE XHR', 'PhyE XHR with AI',none_row)
Dr_result = Dr_result_append(Dr_result, Dr_8_LLY, Dr_8_LLY_AI, y_selected,
                             'PhyF LLY', 'PhyF LLY with AI',none_row)
Dr_result.to_csv('./results/' + file_path + '/User_Exm/Phy_AUC_external.csv')


# Dr_1_result = pd.DataFrame(get_final_result_2(y_selected, Dr_1_XHR, 'PhyA XHR'))
# Dr_1_result_new = get_final_result_2(y_selected, Dr_1_XHR_new, 'PhyA XHR with AI')
# Dr_result = pd.concat([Dr_result, Dr_1_result], ignore_index=True)
# Dr_result = pd.concat([Dr_result, Dr_1_result_new], ignore_index=True)
