import matplotlib.pyplot as plt
import numpy as np
import transtab
import pandas as pd
import json
import pickle
import shap
from analysis_utils import get_final_result, X_y_split, error_samples

info = json.load(open('../data_process/info_0603.json'))
df_train2 = pd.read_csv('../data_process/df_int_train.csv')
col_idx = json.load(open('../data_process/col_idx.json'))

df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
df_alin_test = pd.read_csv('../data_process/df_int_test.csv')
df_alin_ext12 = pd.read_csv('../data_process/df_ext12.csv')
df_alin_ext3 = pd.read_csv('../data_process/New_ext_3.csv')
df_valid = pd.read_csv('../data_process/df_valid.csv')

df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]
df_alin_ext12 = df_alin_ext12.iloc[:,1:]
df_alin_ext3 = df_alin_ext3.iloc[:,1:]

df_valid = df_valid.iloc[:,1:]

ori_data = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
ori_data_train = ori_data.loc[df_train2.iloc[:,0],:]

ori_valid = pd.read_csv('../data_process/valid.csv')
ori_valid = ori_valid.iloc[:,1:]

col_idx = json.load(open('../data_process/col_idx.json'))

df_valid = df_valid.rename(columns=col_idx)
df_valid.rename(columns={'Operation_time':'Operation_time_min', 'Bleeding':'Bleeding, ml'}, inplace=True)

ori_valid = ori_valid.loc[:,df_valid.columns]
X_ori_valid, y_ori_valid = X_y_split(ori_valid, info['target'][0])

df_train2 = df_train2.iloc[:,1:]
com_cols = df_train2.columns
sim_cols = {}
for col in com_cols:
    # if col in
    sim_cols[col] = col_idx[col]

num_samples = 500
file_path = 'LF_bio_240625_best_08'
X_train2, y_train2 = X_y_split(df_train2, info['target'][0])
X_valid, y_valid = X_y_split(df_valid, info['target'][0])

# X_train2 = pd.concat([X_train2, X_valid])
# y_train2 = pd.concat([y_train2, y_valid])

model = transtab.build_classifier(checkpoint='./checkpoint/'+file_path,device='cuda:0')
# model = nn.DataParallel(model).to('cuda')

# ypred = transtab.predict(model, X_train2, y_train2)
# y_pred = get_final_result(y_train2, ypred)
#
# prob_int = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
# result_int = get_final_result( df_alin_test.iloc[:, 0], prob_int,)
#
# prob_ext12 = transtab.predict(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0])
# result_ext12 = get_final_result( df_alin_ext12.iloc[:, 0], prob_ext12,)
#
# prob_ext3 = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
# result_ext3 = get_final_result(df_alin_ext3.iloc[:, 0], prob_ext3, )
#
# prob_val = transtab.predict(model, X_train2, y_train2)
# result_ext3 = get_final_result(y_train2, prob_val, )


# ypred_series = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
# incorrect_samples = X_train2.loc[y_train2 != ypred_series, :]
# incor_idx = list(incorrect_samples.index)
# incor_idx = [x for x in incor_idx if x < num_samples]
# cor_idx = list(set(list(range(num_samples))) - set(incor_idx))
#
# incorrect_samples.fillna(0, inplace=True)




def shap_predict(data):
    return transtab.predict_fun(model, data, X_valid.columns)

# train_clean = df_train2.dropna()
X_valid.fillna(0, inplace=True)

# x_train_clean, y_train_clean = X_y_split(df_train2, info['target'][0])
# x_data, y_data = x_train_clean.iloc[:num_samples,:], y_train_clean.iloc[:num_samples]
# x_data = pd.concat([x_data, X_valid])
# y_data = pd.concat([y_data, y_valid])

#
# ypred_clean = transtab.predict(model, x_train_clean, y_train_clean)
# result = get_final_result(y_train_clean, ypred_clean)

# er_sams,ori_sams,_ = error_samples(df_train2, ori_data_train,result,y_train_clean)
# x_er_sams, y_er_sams = X_y_split(er_sams, info['target'][0])
# x_er_sams_sim = x_er_sams.rename(columns=col_idx)
# x_er_sams_sim_sam = x_er_sams_sim.iloc[:5,:]

# ori_data_train

# ori_data_train = ori_data_train.rename(columns=sim_cols)
# sim_cols['intraoperative bleeding'] = 'Bleeding_A'
# ori_data_train = ori_data_train[sim_cols.values()]
# ori_data_train = ori_data_train.iloc[:num_samples,:]
# x_ori, y_ori = X_y_split(ori_data_train,info['target'][0])

# x_ori_nomiss = x_ori.dropna(how='any')
# idx_x_ori_nomiss = x_ori_nomiss.index
# x_ori = x_ori.fillna(0)
# cor_idx_full = [x for x in incor_idx if x in idx_x_ori_nomiss]
file_path = 'valid3'
explainer = shap.KernelExplainer(model=shap_predict, data=X_valid, device='cpu')
# =================================================================
'''
计算 shap valus
'''
shap_values = explainer.shap_values(X_valid, nsamples=200)
with open('./shap_values/'+file_path+'2.pkl','wb') as f:
    pickle.dump(shap_values, f)

# shap_values_df = explainer(x_data.iloc[0,:])
# with open('./shap_values/'+file_path+'samples.pkl','wb') as f:
#     pickle.dump(shap_values_df, f)

# =================================================================

with open('./shap_values/'+file_path+'2.pkl','rb') as f:
    shap_values = pickle.load(f)

# with open('./shap_values/'+file_path+'samples.pkl','rb') as f:
#     shap_values_df = pickle.load(f)
# =============================================================================
'''
简化特征名称
'''
for col in X_valid.columns:
    if col in col_idx.keys():
        X_valid= X_valid.rename(columns={col:col_idx[col]})

        # print(col)
# X_valid['Gender'] = X_valid['Gender'].map({'male': 0, 'female': 1})
# X_valid['Methods'] = X_valid['Methods'].map({'Laparoscopic surgery': 0, 'Transfer to laparotomy': 1,
#                                              'Laparotomy': 2})
# idx2cat = {'Gender': [0, 1], 'Methods': [0, 1, 2]}

# force_plot = shap.force_plot(explainer.expected_value, shap_values[0], x_data.iloc[0,:],show=True)
# =============================================================================
'''
Summary plot
'''
summary_plot = shap.summary_plot(shap_values,X_ori_valid, show=False,)
plt.savefig('./results/'+file_path+'/shap_summary.png')
plt.close()
# =============================================================================
'''
feature importance
'''
# shap.summary_plot(shap_values, x_data, plot_type="bar",max_display=x_data.shape[1],show=False)
# plt.savefig('./results/v'+file_path+'/feature_importance.png')
# plt.close()
# =============================================================================
'''
force plotplt.close()
'''
shap.initjs()
for i in X_valid.index:
    shap.force_plot(explainer.expected_value, shap_values[i], np.array(X_ori_valid.iloc[i]),
                    list(X_valid.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
    plt.savefig('./results/' + file_path + '/force_error'+str(i)+'.png')
    plt.close()
# for i in cor_idx_full:
#     shap.force_plot(explainer.expected_value, shap_values[i], np.array(x_ori.iloc[i]),
#                     list(x_ori.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
#     plt.savefig('./results/' + file_path + '/force_correct_full'+str(i)+'.png')
#     plt.close()
# for i in cor_idx:
#     shap.force_plot(explainer.expected_value, shap_values[i], np.array(x_ori.iloc[i]),
#                     list(x_ori.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
#     plt.savefig('./results/' + file_path + '/force_correct_miss'+str(i)+'.png')
#     plt.close()

# =============================================================================
'''
dependence plots
'''
# sim_E_num = []
# for i in info['E_num']:
#     sim_E_num.append(col_idx[i])
# sim_D1_num = []
# for i in info['D1_num']:
#     sim_D1_num.append(col_idx[i])
# dic_E_D1 = { 'E_PT-INR': 'D1_PT-INR', 'E_LY': 'D1_LY', 'E_AST': 'D1_AST', 'E_HGB': 'D1_HGB', 'E_PLT': 'D1_PLT',
#  'E_CR': 'D1_ALB', 'E_K': 'D1_PT-INR', 'E_Na': 'D1_Na', 'E_TBIL': 'D1_TBIL', 'E_ALT': 'D1_ALT',
#  'E_TP': 'D1_CR', 'E_NE': 'D1_NE', 'E_ALB': 'D1_ALB', 'E_WBC': 'D1_WBC', 'E_RBC': 'D1_RBC' }
#
# for E_col in dic_E_D1.keys():
#     D1_col = dic_E_D1[E_col]
#     idx_E = x_ori.columns.get_loc(E_col)
#     idx_D1 = x_ori.columns.get_loc(D1_col)
#     x_ori[D1_col] = x_ori[D1_col].replace(0, np.nan)
#     # shap.dependence_plot(ind=idx_E, shap_values=shap_values, features=x_ori, feature_names =x_ori.columns,
#     #                      display_features=x_ori,
#     #                      show=False, interaction_index=idx_D1)
#     shap.dependence_plot(idx_E, shap_values, x_ori.values, feature_names=x_ori.columns,
#                          show=False, interaction_index=idx_D1)
#     plt.savefig('./results/v' + file_path + '/dependence_' + E_col + '.png')
# =============================================================================
# '''
# 错误样本的 waterfall plots
# '''
# for row in range(shap_values_df.shape[0]):
#     shap.plots.waterfall(shap_values_df[row],show=False, feature_names = x_data.columns)
#     plt.savefig('./results/' + file_path + '/waterfall'+str(row)+'.png')
#     plt.close()
# =============================================================================

# x_ori.columns.get_loc[dic_E_D1.keys()[0]]

# # incorrest_shap_values = shap_values[:incorrect_samples.shape[0],:]
#
# for i in range(x_data.shape[0]):
#     shap.plots.force(
#         explainer.expected_value,
#         shap_values[i],
#         show=False# shap_values的第一个元素对应于第一个（也是唯一）样本  # 使用matplotlib绘制，以便于自定义和保存图表# 显示图表
#     )
#     plt.savefig('./results/' + file_path + '/force'+ str(i) + '.jpg')
# =============================================================================