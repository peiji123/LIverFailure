import pandas as pd
import numpy as np
import json
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import transtab
from corals.threads import set_threads_for_external_libraries
from matplotlib.patches import Patch
from matplotlib.path import Path
import matplotlib.patches as mpatches
from corals.correlation.topk.default import cor_topk
set_threads_for_external_libraries(n_threads=1)
from chord import Chord
import seaborn as sns
from corals.correlation.full.default import cor_full
from adjustText import adjust_text
from torch import nn
def X_y_split(df, lal):
    X = df.drop([lal], axis=1)
    y = df[lal]
    return X, y
def find_module_by_type(module, module_type):
    if isinstance(module, module_type):
        return module
    for name, child in module.named_children():
        result = find_module_by_type(child, module_type)
        if result is not None:
            return result
    return None

info = json.load(open('../data_process/info_0702.json'))
col_idx = json.load(open('../data_process/col_idx.json'))
df_test2 = pd.read_csv('../data_process/df_0702_int_train.csv')


# X_test2 = X_test2.iloc[:,1:]

df_test2 = df_test2.iloc[:,1:]



cat_cols = info['cate_cols']
bin_cols = info['bin_cols']
cont_cols = info['cont_cols']
target = info['target']

# cate_list = []
# for col in cat_cols:
#     cate_list.append(X_test2[col].unique())
#
# gender= cate_list[0]
# method = cate_list[1]
# combined_cate = [[g, s] for g in gender for s in method]

# col_default = [1] * (len(bin_cols)+len(cont_cols))

# cont_mean = X_test2[cont_cols].mean()
# col_default = [1] * len(bin_cols) + list(cont_mean)

# new_rows = []
# for cat_row in combined_cate:
#     full_row = cat_row + col_default
#     new_rows.append(full_row)
# column_names = cat_cols + bin_cols + cont_cols
# df_new_rows = pd.DataFrame(new_rows, columns=column_names, index=range(X_test2.shape[0], X_test2.shape[0]+6))

# X_test2 = pd.concat([X_test2, df_new_rows], axis=0)



file_path = 'LF_bio_240625_best_017'

model = transtab.build_classifier(checkpoint='./checkpoint/'+file_path,device='cuda:0')

positive_sams = df_test2[df_test2[target[0]] == 1]
negative_sams = df_test2[df_test2[target[0]] == 0]

X_pos_test2, _ = X_y_split(positive_sams, info['target'][0])
X_neg_test2, _ = X_y_split(negative_sams, info['target'][0])
X_test2, y_test2 = X_y_split(df_test2, info['target'][0])

y_test2 = list(y_test2)
# y_test2 = y_test2 + [1]*6
y_test2 = pd.DataFrame(y_test2).iloc[:,0]

ypred = transtab.predict(model, X_test2, y_test2)

# output = model.input_encoder.feature_extractor(X_test2)
# outputs = model.input_encoder.feature_processor(**output)
# outputs = model.cls_token(**outputs)
# encoder_output = model.encoder(**outputs)
#
# encoder_output_np = encoder_output.detach().cpu().numpy()
# mean_encoded = np.mean(encoder_output_np, axis=2).T
# mean_encoded_noT = np.mean(encoder_output_np, axis=2)
#
# X_mean_encoded_noT = mean_encoded_noT[:,1:]
# X_pd_mean_encoded_noT = pd.DataFrame(X_mean_encoded_noT,columns = X_test2.columns)

def get_encoder(X, model):
    output = model.input_encoder.feature_extractor(X)
    outputs = model.input_encoder.feature_processor(**output)
    outputs = model.cls_token(**outputs)
    encoder_output = model.encoder(**outputs)

    encoder_output_np = encoder_output.detach().cpu().numpy()
    # encoder_output_pd = pd.DataFrame(encoder_output_np, columns=X_test2.columns)
    mean_encoded = np.mean(encoder_output_np, axis=2).T
    mean_encoded_noT = np.mean(encoder_output_np, axis=2)

    X_mean_encoded_noT = mean_encoded_noT[:, 1:]
    X_pd_mean_encoded_noT = pd.DataFrame(X_mean_encoded_noT, columns=X_test2.columns)

    return mean_encoded_noT, X_pd_mean_encoded_noT, encoder_output_np


# base_cols = ['Gender', 'Age', 'BMI', 'Hepatitis B Virus Surface Antigen', 'Hepatitis C Virus',
#              'Fatty liver', 'Hypertension', 'Diabetes', 'Portal hypertension',
#              'Cirrhosis', 'Liver Cancer']
# intra_cols = ['Methods', 'anatomic liver resection', 'extensive liver resection',
#               'number of liver segmentectomies', 'duration of hepatic pedicle clamping',
#               'Operation time', 'intraoperative bleeding',
#               'intraoperative transfusion']
# pre_cols = ['Preoperatively Hemoglobin', 'Preoperatively Total Bilirubin',
#             'Preoperatively Red Blood Cell Count', 'Preoperatively Neutrophils',
#             'Preoperatively Creatinine', 'Preoperatively Potassium',
#             'Preoperatively Total Protein', 'Preoperatively Platelet Count',
#             'Preoperatively Albumin', 'Preoperatively Lymphocytes',
#             'Preoperatively Alanine Aminotransferase', 'Preoperatively White Blood Cell Count',
#             'Preoperatively Prothrombin Time International Normalized Ratio',
#             'Preoperatively Aspartate Aminotransferase', 'Preoperatively Sodium', 'Ascites',
#             'Tumor size', 'Tumor number',
#             'Indocyanine Green Retention at 15 Minutes', 'Alpha-fetoprotein'
#             ]
# post_cols = [' First postoperative day Hemoglobin', ' First postoperative day Total Bilirubin',
#             ' First postoperative day Red Blood Cells', ' First postoperative day  Neutrophils',
#             ' First postoperative day Creatinine', ' First postoperative day Potassium',
#             ' First postoperative day Total Protein', ' First postoperative day Platelets',
#             ' First postoperative day Albumin', ' First postoperative day Lymphocytes',
#             ' First postoperative day Alanine Aminotransferase', ' First postoperative day  White Blood Cells',
#             ' First postoperative day Prothrombin Time international Normalized Ratio',
#             ' First postoperative day  Aspartate Aminotransferase', ' First postoperative day Sodium',]

# base_cols = ['Gender', 'Age', 'BMI', 'Hepatitis B Virus Surface Antigen', 'Hepatitis C Virus',
#              'Fatty liver', 'Hypertension', 'Diabetes', 'Portal hypertension', 'Ascites',
#              'Cirrhosis', 'Liver Cancer','Indocyanine Green Retention at 15 Minutes', 'Alpha-fetoprotein']
# intra_cols = ['Methods', 'anatomic liver resection', 'extensive liver resection',
#               'number of liver segmentectomies', 'duration of hepatic pedicle clamping',
#               'Operation time', 'Tumor size', 'Tumor number', 'intraoperative bleeding',
#               'intraoperative transfusion']
# pre_cols = ['Preoperatively Hemoglobin', 'Preoperatively Total Bilirubin',
#             'Preoperatively Red Blood Cell Count', 'Preoperatively Neutrophils',
#             'Preoperatively Creatinine', 'Preoperatively Potassium',
#             'Preoperatively Total Protein', 'Preoperatively Platelet Count',
#             'Preoperatively Albumin', 'Preoperatively Lymphocytes',
#             'Preoperatively Alanine Aminotransferase', 'Preoperatively White Blood Cell Count',
#             'Preoperatively Prothrombin Time International Normalized Ratio',
#             'Preoperatively Aspartate Aminotransferase', 'Preoperatively Sodium',
#             ]
# post_cols = [' First postoperative day Hemoglobin', ' First postoperative day Total Bilirubin',
#             ' First postoperative day Red Blood Cells', ' First postoperative day  Neutrophils',
#             ' First postoperative day Creatinine', ' First postoperative day Potassium',
#             ' First postoperative day Total Protein', ' First postoperative day Platelets',
#             ' First postoperative day Albumin', ' First postoperative day Lymphocytes',
#             ' First postoperative day Alanine Aminotransferase', ' First postoperative day  White Blood Cells',
#             ' First postoperative day Prothrombin Time international Normalized Ratio',
#             ' First postoperative day  Aspartate Aminotransferase', ' First postoperative day Sodium',]

base_cols = ['Hepatitis B Virus Surface Antigen', 'Hepatitis C Virus',
             'Fatty liver', 'Cirrhosis', 'Liver Cancer']
intra_cols = ['Methods', 'anatomic liver resection', 'extensive liver resection',
              'number of liver segmentectomies', 'duration of hepatic pedicle clamping',
              'Operation time', 'intraoperative bleeding',
              'intraoperative transfusion']
pre_cols = ['Preoperatively Hemoglobin', 'Preoperatively Total Bilirubin',
            'Preoperatively Creatinine', 'Preoperatively Potassium', 'Preoperatively Platelet Count',
            'Preoperatively Albumin',
            'Preoperatively Alanine Aminotransferase', 'Preoperatively White Blood Cell Count',
            'Preoperatively Prothrombin Time International Normalized Ratio',
            'Preoperatively Aspartate Aminotransferase', 'Preoperatively Sodium',
            'Indocyanine Green Retention at 15 Minutes','Ascites',  'Tumor size', 'Tumor number',
            ]
post_cols = [' First postoperative day Hemoglobin', ' First postoperative day Total Bilirubin',
            ' First postoperative day Creatinine', ' First postoperative day Potassium',
            ' First postoperative day Platelets',
            ' First postoperative day Albumin',
            ' First postoperative day Alanine Aminotransferase', ' First postoperative day  White Blood Cells',
            ' First postoperative day Prothrombin Time international Normalized Ratio',
            ' First postoperative day  Aspartate Aminotransferase', ' First postoperative day Sodium',]

# torch.save(encoder_output, './attention_map/encoder.pth')
# atten_weight = torch.load('./attention_map/240522.pth')
# mask = torch.load('./attention_map/mask.pth')
# mask_mean = mask.mean(axis=0).unsqueeze(1)
# atten_weight_mean = atten_weight.mean(axis=0)
# atten_weight_mean = atten_weight_mean/mask_mean
# atten_weight_mean_np = atten_weight_mean.detach().cpu().numpy()

# mask_1 = mask.unsqueeze(1)
# atten_weight = atten_weight*mask_1
# atten_weight_mean = atten_weight.mean(axis=0)

# ========================================================================
# for i in range(0, 177):
# # for i in range(-5, 0):
#     plt.figure(figsize=(15, 15))
#     plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
#     # cax = plt.imshow(atten_weight[i,:,:].detach().cpu().numpy(), cmap='YlGnBu')
#     # plt.colorbar(cax)
#     sns.heatmap(atten_weight[i,1:,1:].detach().cpu().numpy(), fmt=".2f", cmap='OrRd', xticklabels=sim_col_name_nolabel, yticklabels=sim_col_name_nolabel)
#     ax = plt.gca()
#     # ax.xaxis.set_ticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
#     ax.xaxis.set_ticklabels(ax.get_xticklabels(),  fontsize=10)
#     ax.yaxis.set_ticklabels(ax.get_yticklabels(),  fontsize=10)
#     plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/attention'+str(i)+'.png')
#     # plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/attention_map/'+file_path+'/attention'+str(i)+'.png')
#     plt.xticks(sim_col_name_nolabel, fontsize=12)
#     plt.yticks(sim_col_name_nolabel, fontsize=12)
#     plt.show()
# ========================================================================
# tsne_results = tsne_results[1:,:]

# color_ids = {
# # 'LF':['#C82423','o'],
# 'Gender':['#FFC04D','>'], # 黄色右三角形
# 'Methods':['#A5B55D','8'], # 绿色八边形
# 'Ascites':['#FFC04D', '*'], # 黄色五角星# 绿色八角形
# 'anatomic liver resection':['#A5B55D','o'], # 绿色圆形
# 'Diabetes':['#FFC04D','D'], # 黄色竖正方形
# 'Hepatitis C Virus':['#FFC04D','^'], # 黄色三角形
# 'Hypertension':['#FFC04D','h'], # 黄色竖六角形
# 'Fatty liver':['#FFC04D','<'], # 黄色左三角形
# 'Hepatitis B Virus Surface Antigen':['#FFC04D', 'X'], # 黄色乘号
# 'Cirrhosis':['#FFC04D','.'], # 黄色点
# 'Portal hypertension':['#FFC04D', 's'], # 黄色正方形
# 'extensive liver resection':['#A5B55D','P'], # 绿色加号
# 'Liver Cancer':['#FFC04D', 'p'], # 黄色五边形
# 'Age':['#FFC04D', 'd'], # 黄色菱形
# 'Alpha-fetoprotein':['#A5B55D', 's'], # 绿色正方形
# 'Indocyanine Green Retention at 15 Minutes':['#A5B55D', 'd'], # 绿色菱形
# 'number of liver segmentectomies':['#A5B55D','*'], # 绿色五角星
# 'duration of hepatic pedicle clamping':['#A5B55D', 'D'],# 绿色竖四边形
# 'BMI':['#FFC04D','H'], # 黄色六边形
# 'Operation time':['#A5B55D', 'p'], # 绿色五边形
# 'Tumor size':['#A5B55D','^'], # 绿色上三角形
# 'intraoperative bleeding':['#A5B55D','8'], # 绿色八边形
# 'Tumor number':['#A5B55D', 'o'], # 黄色圆形
# 'intraoperative transfusion':['#A5B55D', 'v'], # 绿色下三角形
# # 'Hemihepatic vascular exclusion':['#A5B55D', 'v'], # 绿色下三角形
# 'Preoperatively White Blood Cell Count':['#9AC9DB','>'], # 浅蓝右三角形
# # 'Preoperatively Prothrombin Time':['#9AC9DB','v'],
# 'Preoperatively Prothrombin Time International Normalized Ratio':['#9AC9DB','^'], # 浅蓝上三角
# 'Preoperatively Lymphocytes':['#9AC9DB','p'], # 浅蓝加号
# 'Preoperatively Creatinine':['#9AC9DB','s'], # 浅蓝正方形
# 'Preoperatively Hemoglobin':['#9AC9DB','8'], # 浅蓝八角形
# 'Preoperatively Platelet Count':['#9AC9DB','3'],# 浅蓝下三角
# 'Preoperatively Alanine Aminotransferase':['#9AC9DB','D'], # 浅蓝竖正方形
# 'Preoperatively Potassium':['#9AC9DB','H'], # 浅蓝六边形
# 'Preoperatively Neutrophils':['#9AC9DB','x'], # 浅蓝X
# 'Preoperatively Red Blood Cell Count':['#9AC9DB','.'], # 浅蓝点
# 'Preoperatively Albumin':['#9AC9DB','<'], # 浅蓝左三角形
# 'Preoperatively Aspartate Aminotransferase':['#9AC9DB','P'], # 浅蓝五边形
# 'Preoperatively Total Bilirubin':['#9AC9DB','*'], # 浅蓝五角星
# 'Preoperatively Total Protein':['#9AC9DB','h'], # 浅蓝竖六边形
# 'Preoperatively Sodium':['#9AC9DB','d'], # 浅蓝菱形
# ' First postoperative day Creatinine':['#2878B5','s'], # 蓝色正方形
# ' First postoperative day Total Bilirubin':['#2878B5','*'], # 蓝色五角星
# ' First postoperative day  Aspartate Aminotransferase':['#2878B5','P'], # 蓝色五边形
# ' First postoperative day Platelets':['#2878B5','3'], # 蓝色下三角
# ' First postoperative day Albumin':['#2878B5','<'], # 蓝色做三角形
# ' First postoperative day Total Protein':['#2878B5','h'], # 蓝色竖六边形
# ' First postoperative day  Neutrophils':['#2878B5','x'], # 蓝色X
# ' First postoperative day Prothrombin Time international Normalized Ratio':['#2878B5','^'], # 蓝色上三角
# ' First postoperative day  White Blood Cells':['#2878B5','>'], # 蓝色右三角形
# ' First postoperative day Sodium':['#2878B5','d'], # 蓝色菱形
# ' First postoperative day Lymphocytes':['#2878B5','p'], # 蓝色加号
# ' First postoperative day Red Blood Cells':['#2878B5','.'], # 蓝色点
# ' First postoperative day Hemoglobin':['#2878B5','8'], # 蓝色八角形
# ' First postoperative day Potassium':['#2878B5','H'], # 蓝色六边形
# ' First postoperative day Alanine Aminotransferase':['#2878B5','D'], # 蓝色竖正方形
# # ' First postoperative day Prothrombin Time':['#2878B5','v'],
# }

color_ids = {
# 'LF':['#C82423','o'],
'Gender':['#FFC04D','o'], # 黄色圆形
'Age':['#FFC04D', '^'], # 黄色上三角
'BMI':['#FFC04D','P'], # 黄色加号
'Hepatitis B Virus Surface Antigen':['#FFC04D', 'v'], # 黄色下三角
'Hepatitis C Virus':['#FFC04D','p'], # 黄色五边形
'Fatty liver':['#FFC04D','s'], # 黄色正方形
'Hypertension':['#FFC04D','H'], # 黄色平六边形
'Diabetes':['#FFC04D','d'], # 黄色菱形
'Portal hypertension':['#FFC04D', '>'], # 黄色右三角形
'Ascites':['#FFC04D', '8'], # 黄色八边形
'Cirrhosis':['#FFC04D','h'], # 黄色竖六边形
'Liver Cancer':['#FFC04D', 'D'], # 黄色竖正方形
'Indocyanine Green Retention at 15 Minutes':['#FFC04D', '<'], # 黄色左三角
'Alpha-fetoprotein':['#FFC04D', '*'], # 黄色星号
# ==================================================
'Preoperatively Hemoglobin':['#89bedc','8'], # 浅蓝八角形 9AC9DB
'Preoperatively Total Bilirubin':['#89bedc','*'], # 浅蓝五角星
'Preoperatively Red Blood Cell Count':['#89bedc','o'], # 浅蓝点
'Preoperatively Neutrophils':['#89bedc','X'], # 浅蓝X
'Preoperatively Creatinine':['#89bedc','s'], # 浅蓝正方形
'Preoperatively Potassium':['#89bedc','H'], # 浅蓝六边形
'Preoperatively Total Protein':['#89bedc','h'], # 浅蓝竖六边形
'Preoperatively Platelet Count':['#89bedc','v'],# 浅蓝下三角
'Preoperatively Albumin':['#89bedc','<'], # 浅蓝左三角形
'Preoperatively Lymphocytes':['#89bedc','p'], # 浅蓝加号
'Preoperatively Alanine Aminotransferase':['#89bedc','D'], # 浅蓝竖正方形
'Preoperatively White Blood Cell Count':['#89bedc','>'], # 浅蓝右三角形
'Preoperatively Prothrombin Time International Normalized Ratio':['#89bedc','^'], # 浅蓝上三角
'Preoperatively Aspartate Aminotransferase':['#89bedc','P'], # 浅蓝五边形
'Preoperatively Sodium':['#89bedc','d'], # 浅蓝菱形
# ==================================================
'Methods':['#A5B55D','o'], # 绿色圆形
'anatomic liver resection':['#A5B55D','^'], # 绿色上三角
'extensive liver resection':['#A5B55D','P'], # 绿色加号
'number of liver segmentectomies':['#A5B55D','v'], # 绿色下三角
'duration of hepatic pedicle clamping':['#A5B55D','p'], # 绿色五边形
'Operation time':['#A5B55D', 's'], # 绿色正方形
'Tumor size':['#A5B55D','H'], # 绿色平六边形
'Tumor number':['#A5B55D', 'd'], # 绿色菱形
'intraoperative bleeding':['#A5B55D','>'], # 绿色右三角形
'intraoperative transfusion':['#A5B55D','8'], # 绿色八边形
# ==================================================
' First postoperative day Hemoglobin':['#3989c1','8'], # 蓝色八角形 2878B5
' First postoperative day Total Bilirubin':['#3989c1','*'], # 蓝色五角星
' First postoperative day Red Blood Cells':['#3989c1','o'], # 蓝色点
' First postoperative day  Neutrophils':['#3989c1','X'], # 蓝色X
' First postoperative day Creatinine':['#3989c1','s'], # 蓝色正方形
' First postoperative day Potassium':['#3989c1','H'], # 蓝色六边形
' First postoperative day Total Protein':['#3989c1','h'], # 蓝色竖六边形
' First postoperative day Platelets':['#3989c1','v'], # 蓝色下三角
' First postoperative day Albumin':['#3989c1','<'], # 蓝色做三角形
' First postoperative day Lymphocytes':['#3989c1','p'], # 蓝色加号
' First postoperative day Alanine Aminotransferase':['#3989c1','D'], # 蓝色竖正方形
' First postoperative day  White Blood Cells':['#3989c1','>'], # 蓝色右三角形
' First postoperative day Prothrombin Time international Normalized Ratio':['#3989c1','^'], # 蓝色上三角
' First postoperative day  Aspartate Aminotransferase':['#3989c1','P'], # 蓝色五边形
' First postoperative day Sodium':['#3989c1','d'], # 蓝色菱形
# ' First postoperative day Prothrombin Time':['#2878B5','v'],
}
exam_color_dict = {
'Preoperatively Hemoglobin':['#89bedc','8'], # 浅蓝八角形 9AC9DB
'Preoperatively Total Bilirubin':['#89bedc','*'], # 浅蓝五角星
'Preoperatively Red Blood Cell Count':['#89bedc','o'], # 浅蓝点
'Preoperatively Neutrophils':['#89bedc','X'], # 浅蓝X
'Preoperatively Creatinine':['#89bedc','s'], # 浅蓝正方形
'Preoperatively Potassium':['#89bedc','H'], # 浅蓝六边形
'Preoperatively Total Protein':['#89bedc','h'], # 浅蓝竖六边形
'Preoperatively Platelet Count':['#89bedc','v'],# 浅蓝下三角
'Preoperatively Albumin':['#89bedc','<'], # 浅蓝左三角形
'Preoperatively Lymphocytes':['#89bedc','p'], # 浅蓝加号
'Preoperatively Alanine Aminotransferase':['#89bedc','D'], # 浅蓝竖正方形
'Preoperatively White Blood Cell Count':['#89bedc','>'], # 浅蓝右三角形
'Preoperatively Prothrombin Time International Normalized Ratio':['#89bedc','^'], # 浅蓝上三角
'Preoperatively Aspartate Aminotransferase':['#89bedc','P'], # 浅蓝五边形
'Preoperatively Sodium':['#89bedc','d'], # 浅蓝菱形'P'], # 浅蓝五边形
' First postoperative day Hemoglobin':['#3989c1','8'], # 蓝色八角形 2878B5
' First postoperative day Total Bilirubin':['#3989c1','*'], # 蓝色五角星
' First postoperative day Red Blood Cells':['#3989c1','o'], # 蓝色点
' First postoperative day  Neutrophils':['#3989c1','X'], # 蓝色X
' First postoperative day Creatinine':['#3989c1','s'], # 蓝色正方形
' First postoperative day Potassium':['#3989c1','H'], # 蓝色六边形
' First postoperative day Total Protein':['#3989c1','h'], # 蓝色竖六边形
' First postoperative day Platelets':['#3989c1','v'], # 蓝色下三角
' First postoperative day Albumin':['#3989c1','<'], # 蓝色做三角形
' First postoperative day Lymphocytes':['#3989c1','p'], # 蓝色加号
' First postoperative day Alanine Aminotransferase':['#3989c1','D'], # 蓝色竖正方形
' First postoperative day  White Blood Cells':['#3989c1','>'], # 蓝色右三角形
' First postoperative day Prothrombin Time international Normalized Ratio':['#3989c1','^'], # 蓝色上三角
' First postoperative day  Aspartate Aminotransferase':['#3989c1','P'], # 蓝色五边形
' First postoperative day Sodium':['#3989c1','d'], # 蓝色菱形
}
color_shape_list = list(color_ids.values())
colors = []
shapes = []
for item in color_shape_list:
    colors.append(item[0])  # 提取并添加颜色
    shapes.append(item[1])

color_name_list = list(color_ids.keys())
exam_color_name_list = list(exam_color_dict.keys())
sim_col_name = []
exam_sim_col_name = []
for col in color_name_list:
    sim_col_name.append(col_idx[col])
for col in exam_color_name_list:
    exam_sim_col_name.append(col_idx[col])

# sim_col_name_nolabel = sim_col_name[1:]

# x_coords = [point[0] for point in tsne_results]
# y_coords = [point[1] for point in tsne_results]

# ======================================================================================================
'''
获得嵌入表达
'''
mean_encoded_noT, X_pd_mean_encoded_noT, encoder_output_np = get_encoder(X_test2, model)
encoder_output_np = encoder_output_np[:,1:]
# ======================================================================================================

# X_test2 = X_test2[base_cols+intra_cols+pre_cols+post_cols]
# tsne = TSNE(n_components=2, perplexity=20, learning_rate=50, init='pca', random_state=42)
# tsne_results = tsne.fit_transform(encoder_output_np)
tsne_results = np.zeros([encoder_output_np.shape[0], encoder_output_np.shape[1], 2])
'''
t-sne 网格调参
'''
perplexity_list = [50]
learning_rate_list = [200, 500, 700, 1000]
n_iters_list = [500, 700, 1000]
'''
嵌入方案
    1. 对特征进行嵌入，即 [681, 1, 128] -> [681, 1, 2]
        操作说明： for i in range(encoder_output_np.shape[1]):  
                 tsne_results[:,i,] = tsne.fit_transform(encoder_output_np[:,i,])
                 feature
    2. 对病人进行嵌入， 即 [1, 54, 128] -> [1, 54, 2]
        操作说明： for i in range(encoder_output_np.shape[0]):
                 tsne_results[i] = tsne.fit_transform(encoder_output_np[i])
                 patient
'''
#
# for perplexity in perplexity_list:
#     for learning_rate in learning_rate_list:
#         for n_iters in n_iters_list:

# =====================================================================================================
ori_cols = list(X_test2.columns)
base_cols_dict = {item: ori_cols.index(item) for item in base_cols if item in ori_cols}
pre_cols_dict = {item: ori_cols.index(item) for item in pre_cols if item in ori_cols}
intra_cols_dict = {item: ori_cols.index(item) for item in intra_cols if item in ori_cols}
post_cols_dict = {item: ori_cols.index(item) for item in post_cols if item in ori_cols}
# =====================================================================================================
# pre_list = list(pre_cols_dict.values())
'''
pre & post operative feature embedding by T-SNE
'''
selected_idx_list = list(base_cols_dict.values())+list(pre_cols_dict.values())+list(intra_cols_dict.values())+list(post_cols_dict.values())
selected_name_list = list(base_cols_dict.keys())+list(pre_cols_dict.keys())+list(intra_cols_dict.keys())+list(post_cols_dict.keys())
selected_encoder = encoder_output_np[:,selected_idx_list,:]
reshape_np = np.transpose(selected_encoder, (1, 0 ,2))
reshape_np = reshape_np.reshape(-1, selected_encoder.shape[2])

# encoder_output_reshape = encoder_output_np.reshape(-1, encoder_output_np.shape[2])
tsne = TSNE(n_components=2, perplexity=50, learning_rate=200,  n_iter=500)
tsne_results = tsne.fit_transform(reshape_np)
# =====================================================================================================
'''
重新划分出各个阶段各个特征的tsne嵌入
'''
infor_name_dict = {
'Hepatitis B Virus Surface Antigen': "HBVs Ag",
 'Hepatitis C Virus': "HCV",
 'Fatty liver':'Fatty liver',
 'Cirrhosis':'Cirrhosis',
 'Liver Cancer':'Liver Cancer',
 'Preoperatively Hemoglobin':'Preop HGB',
 'Preoperatively Total Bilirubin':'Preop TBIL',
 'Preoperatively Creatinine': 'Preop CR',
 'Preoperatively Potassium':'Preop K',
 'Preoperatively Platelet Count':'Preop PLT',
 'Preoperatively Albumin':'Preop ALB',
 'Preoperatively Alanine Aminotransferase':'Preop ALT',
 'Preoperatively White Blood Cell Count':'Preop WBC',
 'Preoperatively Prothrombin Time International Normalized Ratio':'Preop PT-INR',
 'Preoperatively Aspartate Aminotransferase':'Preop AST',
 'Preoperatively Sodium':'Preop Na',
 'Indocyanine Green Retention at 15 Minutes':'ICGR15',
 'Ascites':'Ascites',
 'Tumor size': 'Tumor size',
 'Tumor number': 'Tumor number',
 'Methods': 'Methods',
 'anatomic liver resection':'ALR',
 'extensive liver resection':'extensive liver resection',
 'number of liver segmentectomies':'number of liver segmentectomies',
 'duration of hepatic pedicle clamping':'Pringle',
 'Operation time':'Operation time',
 'intraoperative bleeding':'intra bleeding',
 'intraoperative transfusion':'intra transfusion',
 ' First postoperative day Hemoglobin':'Postop 24h HGB',
 ' First postoperative day Total Bilirubin':'Postop 24h TBIL',
 ' First postoperative day Creatinine':'Postop 24h CR',
 ' First postoperative day Potassium':'Postop 24h K',
 ' First postoperative day Platelets':'Postop 24h PLT',
 ' First postoperative day Albumin':'Postop 24h ALB',
 ' First postoperative day Alanine Aminotransferase':'Postop 24h ALT',
 ' First postoperative day  White Blood Cells':'Postop 24h WBC',
 ' First postoperative day Prothrombin Time international Normalized Ratio':'Postop 24h PT-INR',
 ' First postoperative day  Aspartate Aminotransferase':'Postop 24h AST',
 ' First postoperative day Sodium':'Postop 24h K',
}

based_end = len(base_cols) * 681
preop_end = based_end + len(pre_cols) * 681
intra_end = preop_end + len(intra_cols) * 681
post_end = intra_end + len(post_cols) * 681

based_tsne_result = tsne_results[:based_end].reshape(len(base_cols),X_test2.shape[0], tsne_results.shape[1])
based_tsne_result = np.transpose(based_tsne_result, (1, 0 ,2))
preop_tsne_result = tsne_results[based_end:preop_end].reshape(len(pre_cols), X_test2.shape[0],tsne_results.shape[1])
preop_tsne_result = np.transpose(preop_tsne_result, (1, 0 ,2))
intra_tsne_result = tsne_results[preop_end: intra_end].reshape(len(intra_cols),X_test2.shape[0], tsne_results.shape[1])
intra_tsne_result = np.transpose(intra_tsne_result, (1, 0 ,2))
postop_tsne_result = tsne_results[intra_end:].reshape(len(post_cols),X_test2.shape[0], tsne_results.shape[1])
postop_tsne_result = np.transpose(postop_tsne_result, (1, 0 ,2))

# 000000
bas_color_list = ['#1F4E79', '#2E75B6', '#5B9BD5', '#9DC3E6', '#BDD7EE']
preop_color_list = ['#FFF647', '#FDE352', '#F9CA55', '#F5AC3B', '#E28860', '#E76F5B', '#E65858', '#E14773',
                    '#DC508C', '#D4509B', '#C15B9A', '#AD5BA1', '#9650A2', '#8150AE', '#5F54B0']
intra_color_list = ['#016F5D', '#016F5D', '#2C8F4B', '#4AA634', '#7EC97B', '#AEDD8E', '#E5F5AC', '#F7F7AB']
post_color_list = ['#FFFA91', '#FEEE97', '#FBDF99', '#F9CD89', '#EEB8A0', '#F1A99D', '#F09B9B', '#ED91AB',
                   '#EA96BA', '#E596C3', '#DA9DC2']
fig, ax = plt.subplots(figsize=(40, 20))
for i in range(len(bas_color_list)):
    ax.scatter(based_tsne_result[:,i,0], based_tsne_result[:,i,1], color=bas_color_list[i],
               label=infor_name_dict[list(base_cols_dict.keys())[i]])
for i in range(len(preop_color_list)):
    ax.scatter(preop_tsne_result[:,i,0], preop_tsne_result[:,i,1], color=preop_color_list[i],
               label=infor_name_dict[list(pre_cols_dict.keys())[i]])
for i in range(len(intra_color_list)):
    ax.scatter(intra_tsne_result[:,i,0], intra_tsne_result[:,i,1], color=intra_color_list[i],
               label=infor_name_dict[list(intra_cols_dict.keys())[i]])
for i in range(len(post_color_list)):
    ax.scatter(postop_tsne_result[:,i,0], postop_tsne_result[:,i,1], color=post_color_list[i],
               label=infor_name_dict[list(post_cols_dict.keys())[i]])
plt.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=26, ncol=2)
plt.subplots_adjust(right=0.65,left=0.03)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/' + file_path + '/tsne/selected_t-SNE.eps')
plt.show()




# fig, ax = plt.subplots(figsize=(30, 20))
# colors = plt.get_cmap('viridis')(np.linspace(0, 1, int(pre_encoder.shape[1]/2)))
# pre_color_list = ['#F19C9C', '#F5B88E', '#FFDC96', '#FFEAAB', '#EAEC92', '#B3DC93', '#99D097', '#93D8C6', '#A0D7DD',
#                   '#AAC9EF', '#B9B1E8', '#C7ADE3', '#DFACD5', '#D39BB6', '#D0A7A7']
# post_color_list = ['#E54B4B', '#ED7D31', '#FFBF3F', '#FFD966', '#D8DC38', '#74BF3B', '#46A941', '#3AB897', '#52B7C2',
#                    '#659DE1', '#8B82C4', '#996ACC', '#C468B2', '#AE487B', '#AA5E5E']

# name_list = list(base_cols_dict.keys())+list(pre_cols_dict.keys())+list(intra_cols_dict.keys())+list(post_cols_dict.keys())
# com2sim_info = json.load(open('../data_process/col_idx.json'))
# sim_col_list = []
# for col in name_list:
#     sim_col_list.append(com2sim_info[col])
# for i in range(selected_encoder.shape[1]):
#     idx = range(i * selected_encoder.shape[0], (i + 1) * 681)
#     if i <selected_encoder.shape[1]/2:
#         ax.scatter(tsne_results[idx,0], tsne_results[idx, 1], color=pre_color_list[i], label=sim_col_list[i], alpha=0.3,
#                    edgecolors=None, s=100)
#     else:
#         ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1], color=post_color_list[i-int(selected_encoder.shape[1]/2)],
#                    label=sim_col_list[i], alpha=0.3, edgecolors=None, s=100)
#
# plt.legend(fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/' + file_path + '/tsne/preoperative_post.eps')
# plt.show()
# =====================================================================================================
'''
获取pre and post 的嵌入
'''

#
# pre_tsne = tsne_results[:,list(pre_cols_dict.values()),:]
# post_tsne = tsne_results[:,list(post_cols_dict.values()),:]
# '''
# 绘制散点图
# '''
# colors = plt.get_cmap('viridis')(np.linspace(0, 1, pre_tsne.shape[1]))
# fig, ax = plt.subplots(figsize = (15,10))
#
# for feature_idx in range(pre_tsne.shape[1]):
#     pre_data = pre_tsne[:,feature_idx,:]
#     post_data = post_tsne[:,feature_idx,:]
#     ax.scatter(pre_data[:,0], pre_data[:,1], s=10, c=colors[feature_idx],  alpha=0.5 )
#     ax.scatter(post_data[:, 0], post_data[:, 1], s=10, c=colors[feature_idx], alpha=0.5)
#
# # plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/tsne/patient'+str(perplexity)+'lr'+str(learning_rate)+'iter'+str(n_iters)+'.eps')
# # plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/' + file_path + '/tsne/feature' + str(perplexity) + 'lr' + str(learning_rate) + 'iter' + str(n_iters) + '.eps')
# plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/' + file_path + '/tsne/preoperative.eps')
# plt.show()
# =====================================================================================================
'''
使用MDA进行降维度嵌入
'''
# from mda import *
# from sklearn.preprocessing import MinMaxScaler
# neighborNum = 5
#
# feature_encoder_np = encoder_output_np.transpose(1, 0, 2)
# y_pred = np.array([range(feature_encoder_np.shape[0])])
# ypred = ypred.reshape(ypred.shape[0], -1)
# # for i in range(encoder_output_np.shape[1]):
#
# # mm = MinMaxScaler()
# # y_pred = mm.fit_transform(y_pred)
# clusterIdx_pred = discoverManifold(ypred, neighborNum)
#
# for i in range(encoder_output_np.shape[1]):
#
#     yreg = mda(encoder_output_np[:,i,:], clusterIdx_pred)


# =====================================================================================================
'''
特征散点图
'''
# plt.figure(figsize=(15, 10))
# ax = plt.gca()
# texts = []
# points = []
# for x, y, color, shape, name in zip(tsne_results[:,0], tsne_results[:,1], colors, shapes, sim_col_name):
#     if name in exam_sim_col_name:
#         scatter = plt.scatter(x, y, c=color, marker=shape, alpha=0.7, s=300)
#     # scatter = plt.scatter(x, y, c=color, marker=shape, alpha=0.6, s=300)
#     # text = plt.annotate(name, xy=(x, y),ha='center', fontsize=16)
#     # texts.append(plt.annotate(name, xy=(x, y), xytext=(0, 20), textcoords='offset points', ha='center', fontsize=16))
#     # texts.append(plt.text(x, y, name, ha='center', fontsize=11,color='#767171'))
#     # plt.annotate(name, xy=(x, y+2),ha='center', va='top', fontsize=16, )
#     # text = plt.annotate(name, xy=(x, y), ha='center', va='top', fontsize=16, )
#     # points.append(scatter)
#     # texts.append(text)
# # adjust_text(texts, add_objects=points, force_points=10., expand_points=(20., 20.), force_text=10., force_object=10.,
# #             arrowprops=dict(arrowstyle='-', color='#767171'))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(axis='both', which='major', labelsize=16)
# # plt.scatter(tsne_results[:,0], tsne_results[:,1], c=colors,alpha=0.5,marker=shapes)
# # plt.show()
# # plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/attention_map/'+file_path+'/embedding.eps')
# plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/embedding_exam.eps')

# =====================================================================================================
'''
弦图以及Heatmap 的 correction values 以及 edges infor 
'''
mean_pos_encoded_noT, X_pd_mean_pos_encoded_noT, encoder_pos = get_encoder(X_pos_test2, model)
mean_neg_encoded_noT, X_pd_mean_neg_encoded_noT, encoder_neg = get_encoder(X_neg_test2, model)

col_list = base_cols+pre_cols+intra_cols+post_cols
sim_col_name

def get_edge(col_list,sim_cols, X_pd_mean_encoded_noT, mean_encoded_noT):
    X_pd_mean_encoded_noT = X_pd_mean_encoded_noT[col_list]
    cor_values = cor_full(mean_encoded_noT[:,1:])

    edges = []
    for i in range(len(cor_values)):
        for j in range(len(cor_values)):
            if i != j:  # 排除自环
                edges.append((sim_col_name[i], sim_col_name[j], cor_values[i, j]))
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'value'])
    cor_values = pd.DataFrame(cor_values, columns=sim_cols, index=sim_cols)
    return edges_df, cor_values

def get_edge_cos(cor_values, sim_cols):

    edges = []
    for i in range(len(cor_values)):
        for j in range(len(cor_values)):
            if i != j:  # 排除自环
                edges.append((sim_col_name[i], sim_col_name[j], cor_values[i, j]))
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'value'])
    cor_values = pd.DataFrame(cor_values, columns=sim_cols, index=sim_cols)
    return edges_df, cor_values

from sklearn.metrics.pairwise import cosine_similarity
correlation_matrix = np.zeros((encoder_output_np.shape[1], encoder_output_np.shape[1]))
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        feature_i = encoder_neg[:, i, :]
        feature_j = encoder_pos[:, j, :]
        similarity = cosine_similarity(feature_i, feature_j)
        correlation_matrix[i, j] = correlation_matrix[j, i] = np.mean(similarity)
        # if i != j:
        #     feature_i = encoder_neg[:, i, :]
        #     feature_j = encoder_pos[:, j, :]
        #
        #     correlation = np.corrcoef(feature_i, feature_j)[0, 1]
        #     correlation_matrix[i, j] = correlation
        # else:
        #     correlation_matrix[i, j] = 1

plt.figure(figsize=(12, 10))
ax = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', xticklabels=sim_col_name, yticklabels=sim_col_name)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
ax.invert_yaxis()
plt.savefig('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/' + file_path + '/connection/heat_map.eps')
plt.show()

# edges_df, cor_values = get_edge_cos(correlation_matrix, sim_col_name)
edges_pos_df, cor_pos_valuse = get_edge(col_list, sim_col_name,X_pd_mean_pos_encoded_noT, mean_pos_encoded_noT)
edges_neg_df, cor_neg_valuse = get_edge(col_list, sim_col_name,X_pd_mean_neg_encoded_noT, mean_neg_encoded_noT)


edges_df, cor_values = get_edge(col_list, sim_col_name,X_pd_mean_encoded_noT, mean_encoded_noT)
# edges_pos_df, cor_pos_valuse = get_edge(col_list, sim_col_name,X_pd_mean_pos_encoded_noT, mean_pos_encoded_noT)
# edges_neg_df, cor_neg_valuse = get_edge(col_list, sim_col_name,X_pd_mean_neg_encoded_noT, mean_neg_encoded_noT)
#
#
edges_df.to_csv('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/connection/all_edges.csv', index=False)
cor_values.to_csv('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/connection/all_cor_values.csv')
edges_pos_df.to_csv('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/connection/pos_edges.csv', index=False)
cor_pos_valuse.to_csv('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/connection/pos_cor_values.csv')
edges_neg_df.to_csv('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/connection/neg_edges.csv', index=False)
cor_neg_valuse.to_csv('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/connection/neg_cor_values.csv')

# =====================================================================================================
