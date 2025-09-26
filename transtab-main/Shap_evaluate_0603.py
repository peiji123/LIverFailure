import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

import transtab
import pandas as pd
import json

import pickle
import shap
from analysis_utils import get_final_result, X_y_split, error_samples
# from attention import base_cols

info = json.load(open('../data_process/info_0603.json'))
df_train2 = pd.read_csv('../data_process/df_int_train.csv')
col_idx = json.load(open('../data_process/col_idx.json'))

# df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
# df_alin_test = pd.read_csv('../data_process/df_int_test.csv')
# df_alin_ext12 = pd.read_csv('../data_process/df_ext12.csv')
# df_alin_ext3 = pd.read_csv('../data_process/New_ext_3.csv')
# df_valid = pd.read_csv('../data_process/df_valid.csv')

# df_alin_train = df_alin_train.iloc[:,1:]
# df_alin_test = df_alin_test.iloc[:,1:]
# df_alin_ext12 = df_alin_ext12.iloc[:,1:]
# df_alin_ext3 = df_alin_ext3.iloc[:,1:]
# df_valid = df_valid.iloc[:,1:]

ori_data = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
ori_data_train = ori_data.loc[df_train2.iloc[:,0],:]

# ori_valid = pd.read_excel('../data_process/Book1.xlsx').T
# ori_valid.columns = ori_valid.iloc[0]
# ori_valid = ori_valid[1:]

df_train2 = df_train2.iloc[:,1:]
com_cols = df_train2.columns
sim_cols = {}
for col in com_cols:
    # if col in
    sim_cols[col] = col_idx[col]

num_samples = 500
# LF_bio_240625_best_018
file_path = 'LF_bio_240625_best_018'
X_train2, y_train2 = X_y_split(df_train2, info['target'][0])
# X_valid, y_valid = X_y_split(df_valid, info['target'][0])

# X_train2 = pd.concat([X_train2, X_valid])
# y_train2 = pd.concat([y_train2, y_valid])
threshold=0.67
model = transtab.build_classifier(checkpoint='./checkpoint/'+file_path,device='cuda:0')
# model = nn.DataParallel(model).to('cuda')

ypred = transtab.predict(model, X_train2, y_train2)
y_pred = get_final_result(threshold, y_train2, ypred)

# prob_int = transtab.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
# result_int = get_final_result( df_alin_test.iloc[:, 0], prob_int,)

# prob_ext12 = transtab.predict(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0])
# result_ext12 = get_final_result( df_alin_ext12.iloc[:, 0], prob_ext12,)
#
# prob_ext3 = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
# result_ext3 = get_final_result(df_alin_ext3.iloc[:, 0], prob_ext3, )

prob_val = transtab.predict(model, X_train2, y_train2)
result_ext3 = get_final_result(threshold, y_train2, prob_val, )


ypred_series = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
incorrect_samples = X_train2.loc[y_train2 != ypred_series, :]
incor_idx = list(incorrect_samples.index)
incor_idx = [x for x in incor_idx if x < num_samples]
cor_idx = list(set(list(range(num_samples))) - set(incor_idx))

incorrect_samples.fillna(0, inplace=True)


col_idx = json.load(open('../data_process/col_idx.json'))

def shap_predict(data):
    return transtab.predict_fun(model, data, X_train2.columns)

# train_clean = df_train2.dropna()
df_train2.fillna(0, inplace=True)

x_train_clean, y_train_clean = X_y_split(df_train2, info['target'][0])
x_data, y_data = x_train_clean.iloc[:num_samples,:], y_train_clean.iloc[:num_samples]
# x_data, y_data = x_train_clean, y_train_clean
# x_data = pd.concat([x_data, X_valid])
# y_data = pd.concat([y_data, y_valid])


ypred_clean = transtab.predict(model, x_train_clean, y_train_clean)
result = get_final_result(threshold, y_train_clean, ypred_clean)

er_sams,ori_sams,_ = error_samples(df_train2, ori_data_train,result,y_train_clean)
x_er_sams, y_er_sams = X_y_split(er_sams, info['target'][0])
x_er_sams_sim = x_er_sams.rename(columns=col_idx)
x_er_sams_sim_sam = x_er_sams_sim.iloc[:5,:]

# ori_data_train

ori_data_train = ori_data_train.rename(columns=sim_cols)
sim_cols['intraoperative bleeding'] = 'Bleeding_A'
ori_data_train = ori_data_train[sim_cols.values()]
ori_data_train = ori_data_train.iloc[:num_samples,:]
x_ori, y_ori = X_y_split(ori_data_train,info['target'][0])

x_ori_nomiss = x_ori.dropna(how='any')
idx_x_ori_nomiss = x_ori_nomiss.index
x_ori_full = x_ori.fillna(0)
cor_idx_full = [x for x in incor_idx if x in idx_x_ori_nomiss]

explainer = shap.KernelExplainer(model=shap_predict, data=x_data, device='cpu')
# =================================================================
'''
计算 shap valus
'''
# shap_values = explainer.shap_values(x_data, nsamples=100)
# with open('./shap_values/'+file_path+'.pkl','wb') as f:
#     pickle.dump(shap_values, f)
#
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
for col in x_data.columns:
    if col in col_idx.keys():
        x_data= x_data.rename(columns={col:col_idx[col]})
        # print(col)
x_data['Gender'] = x_data['Gender'].map({'male': 0, 'female': 1})
x_data['Methods'] = x_data['Methods'].map({'Laparoscopic surgery': 0, 'Transfer to laparotomy': 1,
                                             'Laparotomy': 2})
idx2cat = {'Gender': [0, 1], 'Methods': [0, 1, 2]}

# force_plot = shap.force_plot(explainer.expected_value, shap_values[0], x_data.iloc[0,:],show=True)
# =============================================================================
'''
Summary plot
'''
form_cols = {'Gender':'Gender', 'Methods':'Methods',
             'Major':'Extensive liver resection',
             'ALR':'ALR', 'Portal_hypertension':'Portal hypertension',
             'HBVs Ag':'HBVs Ag','Hypertension':'Hypertension',
             'HCV':'HCV', 'Cirrhosis':'Cirrhosis', 'Liver_Cancer':'Liver cancer',
             'Fatty_liver': 'Fatty liver', 'Ascites':'Ascites', 'Diabetes':'Diabetes',
             'Age':'Age', 'AFP':'AFP', 'ICGR15':'ICGR15',
             'S_Number':'Number of liver segmentectomies',
             'Pringle':'Pringle','BMI':'BMI', 'Operation_time':'Operation time',
             'Tumor_size':'Tumor size', 'Bleeding':'Intraoperative bleeding',
             'Tumor_number':'Tumor number', 'Transfusion':'Intraoperative transfusion',
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
# x_data = x_data.rename(columns=form_cols)
# summary_plot = shap.summary_plot(shap_values,x_data, show=False,max_display=10)
# plt.savefig('./results/'+file_path+'/Global_explanation/shap_summary2.eps')
# plt.close()
#
# shap_value_pd = pd.DataFrame(shap_values, columns=x_data.columns)
# shap_value_pd.to_csv('./results/'+file_path+'/Global_explanation/shap_value_pd.csv')

# =============================================================================
'''
feature importance
'''
# shap.summary_plot(shap_values, x_data, plot_type="bar",max_display=x_data.shape[1],show=False)
# plt.savefig('./results/'+file_path+'/Global_explanation/feature_importance.eps')
# plt.close()
#
# abs_shap_value = np.abs(shap_values).mean(axis=0)
# abs_shap_value = pd.DataFrame(abs_shap_value.reshape(1,-1), columns=x_data.columns)
#
# abs_shap_value.to_csv('./results/'+file_path+'/Global_explanation/abs_shap_value.csv')

# =============================================================================
'''
分阶段 feature importance
'''
Preop_cols = ['Preop RBC', 'Preop NE', 'Preop PT-INR', 'Preop K', 'Preop TP', 'Preop ALT',
    'Preop HGB', 'Preop LY', 'Preop TBIL', 'Preop ALB', 'Preop CR', 'Preop WBC',
    'Preop Na', 'Preop AST',
    'Tumor size', 'Tumor number', 'ICGR15', 'AFP', 'Ascites'
]
intra_col = [
    'Methods', 'ALR', 'Extensive liver resection',
    'Number of liver segmentectomies', 'Pringle', 'Operation time',
    'Intraoperative bleeding', 'Intraoperative transfusion'
]
postop_cols = [
'Postop 24h ALT', 'Postop 24h ALB', 'Postop 24h AST', 'Postop 24h WBC',
    'Postop 24h NE', 'Postop 24h TBIL',
       'Postop 24h K', 'Postop 24h HGB', 'Postop 24h RBC', 'Postop 24h TP',
    'Postop 24h Na', 'Postop 24h CR', 'Postop 24h PLT',
       'Postop 24h LY', 'Postop 24h PT-INR'
]
bas_cols = [
    'Gender', 'Age', 'BMI', 'HBVs Ag',
       'Hypertension', 'HCV', 'Cirrhosis', 'Liver cancer', 'Fatty liver'
]


# bas_mean_shap_value = abs_shap_value[bas_cols].mean(axis=1)
# pre_mean_shap_value = abs_shap_value[Preop_cols].mean(axis=1)
# intra_mean_shap_value = abs_shap_value[intra_col].mean(axis=1)
# post_mean_shap_value = abs_shap_value[postop_cols].mean(axis=1)
# mean_shap_value = [float(bas_mean_shap_value), float(pre_mean_shap_value),
# #                    float(intra_mean_shap_value), float(post_mean_shap_value)]
# phase_name = ['Basic info', 'Preop factors', 'Intraop factors',
#               'Postop 24h factors',]
# plt.bar(phase_name, mean_shap_value)
# plt.show()

all_cols = x_data.columns.tolist()

# preop_data = x_data[Preop_cols]
# intra_data = x_data[intra_col]
# postop_data = x_data[postop_cols]
# bas_data = x_data[bas_cols]
#
# def col_idx(col_list):
#     cols_idx =  []
#     for col in col_list:
#         cols_idx.append(all_cols.index(col))
#     return cols_idx
#
# preop_idx = col_idx(Preop_cols)
# postop_idx = col_idx(postop_cols)
# intra_idx = col_idx(intra_col)
# bas_idx = col_idx(bas_cols)
#
# preop_shap_value = shap_values[:,preop_idx]
# post_shap_value = shap_values[:,postop_idx]
# intra_shap_value = shap_values[:,intra_idx]
# bas_shap_value = shap_values[:,bas_idx]
#
# preop_mean_shap_value = np.abs(preop_shap_value).mean(axis=0)
# post_mean_shap_value = np.abs(post_shap_value).mean(axis=0)
# intra_mean_shap_value = np.abs(intra_shap_value).mean(axis=0)
# bas_mean_shap_value = np.abs(bas_shap_value).mean(axis=0)
#
# preop_average = preop_mean_shap_value.mean()
# bas_average = bas_mean_shap_value.mean()
# intra_average = intra_mean_shap_value.mean()
# post_average = post_mean_shap_value.mean()


def top_five_mean_shap_value(mean_shap_value, column_name):
    mean_shap_value_pd = pd.DataFrame(mean_shap_value.reshape(1,-1), columns=column_name)
    series = mean_shap_value_pd.iloc[0]
    top_five = series.sort_values(ascending=False)[:5]
    # print(top_five)
    return top_five

def top_shap_value_variables(number, shap_values, column_names):
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    mean_shap_values_pd = pd.DataFrame([mean_shap_values], columns=column_names).iloc[0]
    top_variables = mean_shap_values_pd.sort_values(ascending=False).head(number)
    # print(top_five)
    return top_variables

# bas_top_five = top_five_mean_shap_value(bas_mean_shap_value, bas_cols)
# preop_top_five = top_five_mean_shap_value(preop_mean_shap_value, Preop_cols)
# intra_top_five = top_five_mean_shap_value(intra_mean_shap_value, intra_col)
# post_top_five = top_five_mean_shap_value(post_mean_shap_value, postop_cols)
all_top_10 = top_shap_value_variables(10, shap_values, all_cols)
print(all_top_10.index)
all_top_20 = top_shap_value_variables(20, shap_values, all_cols)
print(all_top_20.index)
all_top_30 = top_shap_value_variables(30, shap_values, all_cols)
print(all_top_30.index)

# bas_top_five['Basic average'] = bas_average
# preop_top_five['Preop average'] = preop_average
# intra_top_five['Intra average'] = intra_average
# post_top_five['Postop 24h average'] = post_average
#
# all_labels = (list(bas_top_five.index) + list(preop_top_five.index) +
#               list(intra_top_five.index)+ list(post_top_five.index))
# print(all_labels )
# top_data = [list(bas_top_five), list(preop_top_five), list(intra_top_five), list(post_top_five)]
# top_data_pd = pd.DataFrame(top_data,index=phase_name)
# top_data_pd.to_csv('./results/'+file_path+'/Global_explanation/top_shap_values.csv')

# num_features = len(top_data[0])
# num_grops = len(top_data)
# group_width = 0.8
# bar_width = 5 * group_width / num_features
# index = np.arange(0,num_grops*(num_features+1), num_features+1)
#
# fig, ax = plt.subplots(figsize=(20,10))
# for i, group_data in enumerate(top_data):
#     position = index[i] + np.arange(num_features)
#     bars = ax.bar(position , group_data,width=bar_width, label=phase_name[i])
#     for bar, label in zip(bars, all_labels[i*num_features:(i+1)*num_features]):
#         ax.text(bar.get_x()+bar.get_width()/2, -5, label, ha='center', va='top',rotation=90)
#
# ax.set_xticks(index+group_width*3)
# # ax.set_xticklabels(bas_cols+Preop_cols+intra_col+postop_cols)
# ax.set_xticklabels(phase_name, fontsize=26)
# ax.tick_params(axis='y', which='major', labelsize=26)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.show()

# N = 5

# top_preop_shap_value_idx = np.argsort(-preop_mean_shap_value)[:N]
# top_post_shap_value_idx = np.argsort(-post_mean_shap_value)[:N]
# top_intra_shap_value_idx = np.argsort(intra_mean_shap_value)[:N]
# top_bas_shap_value_idx = np.argsort(bas_mean_shap_value)[:N]

# top_preop_data = preop_data.loc[:,preop_top_five.index]
# top_post_data = postop_data.loc[:,post_top_five.index]
# top_intra_data = intra_data.loc[:,intra_top_five.index]
# top_bas_data = bas_data.loc[:,bas_top_five.index]

# top_preop_shap_value = preop_shap_value[:,top_preop_shap_value_idx]
# top_post_shap_value = post_shap_value[:,top_post_shap_value_idx]
# top_intra_shap_value = shap_values[:,top_intra_shap_value_idx]
# top_bas_shap_value = shap_values[:,top_bas_shap_value_idx]

# top_data = pd.concat([top_bas_data, top_preop_data, top_post_data, top_intra_data], axis=1)
# top_shap_value = np.concatenate([top_preop_shap_value, top_post_shap_value, top_intra_shap_value,top_bas_shap_value],axis=1)

# top_shap_value_cols = np.abs(top_shap_value).mean(axis=0)
# top_cols = list(reversed(top_data.columns))
# top_shap_value_cols = list(reversed(top_shap_value_cols))

# for i in range(5, len(top_cols), 6):
#     top_cols.insert(i, ' ')
#
# for i in range(5, len(top_shap_value_cols), 6):
#     top_shap_value_cols.insert(i, 0.)
#
# # shap.summary_plot(top_shap_value, top_data, plot_type="bar",show=True)
# plt.figure(figsize=(7,10))
# plt.subplots_adjust(left=0.2)
# plt.barh(top_cols, top_shap_value_cols, color='skyblue')
# plt.show()

# new_cols = []
# 创建一个新列表来存储带有间隔的值
# new_shap_value_cols = []
# 创建一个新列表来存储带有间隔的位置
# positions = []
# 添加原始的标签和值
# for i, col in enumerate(top_cols):
#     new_cols.append(col)
#     new_shap_value_cols.append(top_shap_value_cols[i])
#     positions.append(i)
#     # 在每五个元素之后插入一个间隔
#     if (i + 1) % 5 == 0 and i < len(top_cols) - 1:
#         new_cols.append(' ')
#         new_shap_value_cols.append(0.)
#         positions.append(i + 0.5)  # 添加一个中间位置作为间隔
# # 绘制柱状图
# plt.figure(figsize=(7, 10))
# plt.subplots_adjust(left=0.3)
# plt.barh(positions, new_shap_value_cols, color='skyblue')
# # 设置 y 轴的标签
# plt.yticks(positions, new_cols)
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.savefig('./results/'+file_path+'/Global_explanation/Top5.eps')
# plt.show()

# preop_summary_plot = shap.summary_plot(preop_shap_value, preop_data, show=True,max_display=5,plot_type="bar")
# plt.savefig('./results/'+file_path+'/Global_explanation/preop_summary.eps')
# plt.close()
# intra_summary_plot = shap.summary_plot(intra_shap_value, intra_data, show=True,max_display=5,plot_type="bar")
# =============================================================================
'''
前几个特征依赖图
'''


mean_shape_value = np.abs(shap_values).mean(axis=0)
top_num = 55
top_10_indices = np.argsort(mean_shape_value)[-top_num:][::-1]
top_10_values = shap_values[:,top_10_indices]
top_x_data = x_data.iloc[:,top_10_indices]
# explnation = explainer(x_data)
x_ori_full = x_ori_full.rename(columns=form_cols)


# 假设 base_value 是已知的，或者你可以从原始计算中获得
base_value = transtab.predict(model, x_data, y_data).mean() # 仅作为示例，实际应用中可能需要调整

# 创建SHAP解释对象
expl = shap.Explanation(values=shap_values,
base_values=base_value,
data=x_data,
feature_names=x_data.columns.tolist())
import matplotlib.font_manager as fm
for idx in top_10_indices:
    plt.figure(figsize=(10, 7))
    x_ori_reset_idx = x_ori.reset_index(drop=True)
    col_value = x_ori_reset_idx.iloc[:,idx]
    col_clean_value = col_value.dropna()
    clean_idx = col_clean_value.index
    clean_shap_value = shap_values[clean_idx,idx]

    value_shap_value = np.concatenate([np.array(col_clean_value).reshape(-1,1), clean_shap_value.reshape(-1,1)],axis=1)
    pd.DataFrame(value_shap_value).to_csv('./results/'+file_path+'/local_explanation/Top_'+x_data.iloc[:,idx].name+'.csv')
    # shap.dependence_plot(idx, shap_values, x_data.values,
    #                      feature_names=x_data.columns, show=True)
    # shap.plots.scatter(expl[:,idx], show=True)

    # plt.scatter(col_clean_value, clean_shap_value, color='#5B9BD5',alpha=0.7,zorder=2)
    pos_mark = clean_shap_value > 0
    neg_mark = clean_shap_value < 0

    plt.scatter(
        np.array(col_clean_value)[pos_mark],
        np.array(clean_shap_value)[pos_mark],
        color='#BF3354',
        zorder=2
    )
    plt.scatter(
        np.array(col_clean_value)[neg_mark],
        np.array(clean_shap_value)[neg_mark],
        color='#5B9BD5',
        zorder=2
    )
    # plt.scatter(col_clean_value, clean_shap_value, color='#5B9BD5', zorder=2)
    plt.axhline(y=0, color='#C00000',linestyle='--',linewidth=2, zorder=1)
    # plt.rcParams['font.family'] = 'Times New'
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    prop = fm.FontProperties(fname=font_path)

    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    plt.xlabel(x_data.iloc[:,idx].name, fontsize=18,fontproperties=prop)
    plt.ylabel(x_data.iloc[:,idx].name+' shap value', fontsize=18,fontproperties=prop)
    plt.xticks(fontsize=16,fontproperties=prop)
    plt.yticks(fontsize=16,fontproperties=prop)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Distribution of SHAP value on '+x_data.iloc[:,idx].name,fontproperties=prop,
              fontsize=22,pad=20, fontweight='bold')
    plt.savefig('./results/'+file_path+'/local_explanation/Top_'+x_data.iloc[:,idx].name+'.eps')
    # plt.show()


# =============================================================================
'''
force plotplt.close()
'''
# shap.initjs()
# for i in incor_idx:
#     shap.force_plot(explainer.expected_value, shap_values[i], np.array(x_ori_full.iloc[i]),
#                     list(x_ori_full.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
#     plt.savefig('./results/' + file_path + '/patient_explanation/force_error'+str(i)+'.svg')
#     plt.close()
# for i in cor_idx_full:
#     shap.force_plot(explainer.expected_value, shap_values[i], np.array(x_ori_full.iloc[i]),
#                     list(x_ori_full.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
#     plt.savefig('./results/' + file_path + '/patient_explanation/force_correct_full'+str(i)+'.svg')
#     plt.close()
# for i in cor_idx:
#     shap.force_plot(explainer.expected_value, shap_values[i], np.array(x_ori_full.iloc[i]),
#                     list(x_ori_full.columns),matplotlib=True,show=False, out_names='p(LF)', figsize=(20,5))
#     plt.savefig('./results/' + file_path + '/patient_explanation/force_correct_miss'+str(i)+'.svg')
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
#     plt.savefig('./results/' + file_path + '/dependence_' + E_col + '.eps')
# =============================================================================
# '''
# 错误样本的 waterfall plots
# '''
# for row in range(shap_values_df.shape[0]):
#     shap.plots.waterfall(shap_values_df[row],show=False, feature_names = x_data.columns)
#     plt.savefig('./results/' + file_path + '/waterfall'+str(row)+'.eps')
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