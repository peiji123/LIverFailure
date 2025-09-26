import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
from transtab.analysis_utils import confusion_matrix_plots, error_samples, test_2_oridata, error_samples_name
import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
# fm.FontProperties(fname=font_path)
prop = fm.FontProperties(fname=font_path)
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
print(prop.get_name())
data_int = pd.read_csv('../data_process/df_int_test.csv')
ori_data_int = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
data_ext12 = pd.read_csv('../data_process/df_ext12.csv')
ori_data_ext12 = pd.read_excel('../data_process/外部验证中心12-601.xlsx')
data_ext3 = pd.read_csv('../data_process/New_ext_3_240625_best_018.csv')
ori_data_ext3 = pd.read_csv('../data_process/ori_240625_best_018_new_ext_3.csv')
ori_data_test = ori_data_int.loc[data_int.iloc[:,0],:]
ori_data_test.to_csv('../data_process/ori_test.csv')
data_ext4 = pd.read_csv('../data_process/df_ext4_alin.csv')
ori_data_ext4 = pd.read_excel('../data_process/外部验证中心4.xlsx')

path='LF_bio_240625_best_018'
model_list = ['lr', 'rf14', 'xgb', 'svc',  'mlp',  'saint', 'tabnet2', 'tabpfn2', 'transtab2', 'our']
data_list = ['int','int', 'ext12', 'ext3','ext4']

model_name_list = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Support Vector Machine', 'MLP', 'SAINT',
                   'TabNet', 'TabPFN', 'TransTab', 'Ours']
result_summary_list = {}
for data in data_list:
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.size'] = 42
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(40,30))

    models = {}
    for model in model_list:
        dic_path = './Real_Prediction/dict_' + model + '_' + data + '_result.json'
        dic = json.load(open(dic_path, 'r'))
        models[model] = dic
    result_summary_list[data] = models

    confusion_M = [confusion_matrix(models[model]['real'], models[model]['pred_pro']) for model in model_list]
    labels = ['non-PHLF', 'PHLF']
    for i, ax in enumerate(axes.flat):
        if i < len(confusion_M):
            # plt.rcParams['font.family'] = 'serif'
            # plt.rcParams['font.serif'] = ['Times New Roman']
            plt.rcParams['font.size'] = 42
            sns.heatmap(confusion_M[i], annot=True, cmap='Blues',fmt='d',ax=ax,
                        xticklabels=labels,yticklabels=labels,cbar=False)
            # ax.set_title(model_name_list[i],fontweight='bold', fontproperties=prop)
            ax.set_title(model_name_list[i], fontdict={'fontsize': 42,'fontweight': 'bold','fontname':prop.get_name()})


        else:
            ax.axis('off')
    plt.tight_layout()

    plt.savefig('./results/'+path+'/Confusion_matrix/'+data+'.eps')
# exter_rf_dict = {}
























# dict_all_auc = {}
# for ml, dt in itertools.product(model_list, data_list):
#     model = ml
#     data_name = dt
#
#     if data_name == 'int':
#         data = data_int
#         ori_data = ori_data_test
#     elif data_name == 'ext12':
#         data = data_ext12
#         ori_data = ori_data_ext12
#     elif data_name == 'ext3':
#         data = data_ext3
#         ori_data = ori_data_ext3
#     elif data_name == 'ext4':
#         data = data_ext4
#         ori_data = ori_data_ext4
#
#     dic_path = './Real_Prediction/dict_'+model+'_'+data_name+'_result.json'
#     # plot_path = './plots/confusion_matrix_'+model+'_'+data_name+'.png'
#     plot_path = './results/'+path+'/Confusion_matrix/'+model+'_'+data_name+'.png'
#
#     dic = json.load(open(dic_path, 'r'))
#     cmp = confusion_matrix_plots(dic['real'], dic['pred_pro'],plot_path)
#     _,_,er_list = error_samples_name(data, ori_data, dic['pred_pro'], data['PHLF'], model)
#
#     with open('./plots/error_type_list.txt','a') as f:
#         f.write(model + '_' + data_name + '\n')
#         f.write(str(er_list)+ '\n'+ '\n')
#
#     fpr, tpr, thresholds = roc_curve(dic['real'], dic['pred'])
#     roc_auc = auc(fpr, tpr)
#
#     dict_auc = {}
#     dict_auc['fpr_'+model+'_'+data_name] = list(fpr)
#     dict_auc['tpr_'+model+'_'+data_name] = list(tpr)
#     dict_auc['roc_auc_'+model+'_'+data_name] = roc_auc
#
#     dict_all_auc.update(dict_auc)
#     with open('./plots/roc_auc.json','w') as f:
#         json.dump(dict_all_auc,f)
#
# dict_auc = json.load(open('./plots/roc_auc.json'))
# lw = 3
#
# plt.figure(figsize=[13, 10])
# plt.plot(dict_auc['fpr_lr_int'], dict_auc['tpr_lr_int'], color='#FCDC89', alpha=0.7,
#          lw=lw, label='LR (AUC = %0.3f)' % dict_auc['roc_auc_lr_int'])
# plt.plot(dict_auc['fpr_rf_int'], dict_auc['tpr_rf_int'], color='#E3EA96',alpha=0.7,
#          lw=lw, label='RF (AUC = %0.3f)' % dict_auc['roc_auc_rf_int'])
# plt.plot(dict_auc['fpr_xgb_int'], dict_auc['tpr_xgb_int'], color='#AAD09D',alpha=0.7,
#          lw=lw, label='XGB (AUC = %0.3f)' % dict_auc['roc_auc_xgb_int'])
# plt.plot(dict_auc['fpr_svc_int'], dict_auc['tpr_svc_int'], color='#66BC98',alpha=0.7,
#          lw=lw, label='SVC (AUC = %0.3f)' % dict_auc['roc_auc_svc_int'])
# plt.plot(dict_auc['fpr_mlp_int'], dict_auc['tpr_mlp_int'], color='#73D2D7',alpha=0.7,
#          lw=lw, label='MLP (AUC = %0.3f)' % dict_auc['roc_auc_mlp_int'])
# plt.plot(dict_auc['fpr_saint_int'], dict_auc['tpr_saint_int'], color='#B2D2E8',alpha=0.7,
#          lw=lw, label='SAINT (AUC = %0.3f)' % dict_auc['roc_auc_saint_int'])
# plt.plot(dict_auc['fpr_tabnet2_int'], dict_auc['tpr_tabnet2_int'], color='#71B1D7',alpha=0.7,
#          lw=lw, label='TabNet (AUC = %0.3f)' % dict_auc['roc_auc_tabnet2_int'])
# plt.plot(dict_auc['fpr_tabpfn2_int'], dict_auc['tpr_tabpfn2_int'], color='#9491D7',alpha=0.7,
#          lw=lw, label='TabPFN (AUC = %0.3f)' % dict_auc['roc_auc_tabpfn2_int'])
# plt.plot(dict_auc['fpr_transtab2_int'], dict_auc['tpr_transtab2_int'], color='#7A6BB8',alpha=0.7,
#          lw=lw, label='TransTab (AUC = %0.3f)' % dict_auc['roc_auc_transtab2_int'])
# plt.plot(dict_auc['fpr_our_int'], dict_auc['tpr_our_int'], color='#D24D3E',alpha=1,
#          lw=lw, label='Ours (AUC = %0.3f)' % dict_auc['roc_auc_our_int'])
# # plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curves for cohort 1')
# plt.legend(loc="lower right")
# plt.savefig('./results/'+path+'/AUC/int.png')
# # plt.show()
#
# plt.figure(figsize=[13, 10])
# plt.plot(dict_auc['fpr_lr_ext12'], dict_auc['tpr_lr_ext12'], color='#FCDC89', alpha=0.7,
#          lw=lw, label='LR (AUC = %0.3f)' % dict_auc['roc_auc_lr_ext12'])
# plt.plot(dict_auc['fpr_rf_ext12'], dict_auc['tpr_rf_ext12'], color='#E3EA96',alpha=0.7,
#          lw=lw, label='RF (AUC = %0.3f)' % dict_auc['roc_auc_rf_ext12'])
# plt.plot(dict_auc['fpr_xgb_ext12'], dict_auc['tpr_xgb_ext12'], color='#AAD09D',alpha=0.7,
#          lw=lw, label='XGB (AUC = %0.3f)' % dict_auc['roc_auc_xgb_ext12'])
# plt.plot(dict_auc['fpr_svc_ext12'], dict_auc['tpr_svc_ext12'], color='#66BC98',alpha=0.7,
#          lw=lw, label='SVC (AUC = %0.3f)' % dict_auc['roc_auc_svc_ext12'])
# plt.plot(dict_auc['fpr_mlp_ext12'], dict_auc['tpr_mlp_ext12'], color='#73D2D7',alpha=0.7,
#          lw=lw, label='MLP (AUC = %0.3f)' % dict_auc['roc_auc_mlp_ext12'])
# plt.plot(dict_auc['fpr_saint_ext12'], dict_auc['tpr_saint_ext12'], color='#B2D2E8',alpha=0.7,
#          lw=lw, label='SAINT (AUC = %0.3f)' % dict_auc['roc_auc_saint_ext12'])
# plt.plot(dict_auc['fpr_tabnet2_ext12'], dict_auc['tpr_tabnet2_ext12'], color='#71B1D7',alpha=0.7,
#          lw=lw, label='TabNet (AUC = %0.3f)' % dict_auc['roc_auc_tabnet2_ext12'])
# plt.plot(dict_auc['fpr_tabpfn2_ext12'], dict_auc['tpr_tabpfn2_ext12'], color='#9491D7',alpha=0.7,
#          lw=lw, label='TabPFN (AUC = %0.3f)' % dict_auc['roc_auc_tabpfn2_ext12'])
# plt.plot(dict_auc['fpr_transtab2_ext12'], dict_auc['tpr_transtab2_ext12'], color='#7A6BB8',alpha=0.7,
#          lw=lw, label='TransTab (AUC = %0.3f)' % dict_auc['roc_auc_transtab2_ext12'])
# plt.plot(dict_auc['fpr_our_ext12'], dict_auc['tpr_our_ext12'], color='#D24D3E',alpha=1,
#          lw=lw, label='Ours (AUC = %0.3f)' % dict_auc['roc_auc_our_ext12'])
# # plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curves for cohort 2')
# plt.legend(loc="lower right")
# plt.savefig('./results/'+path+'/AUC/ext12.png')
#
# plt.figure(figsize=[13, 10])
# plt.plot(dict_auc['fpr_lr_ext3'], dict_auc['tpr_lr_ext3'], color='#FCDC89', alpha=0.7,
#          lw=lw, label='LR (AUC = %0.3f)' % dict_auc['roc_auc_lr_ext3'])
# plt.plot(dict_auc['fpr_rf_ext3'], dict_auc['tpr_rf_ext3'], color='#E3EA96',alpha=0.7,
#          lw=lw, label='RF (AUC = %0.3f)' % dict_auc['roc_auc_rf_ext3'])
# plt.plot(dict_auc['fpr_xgb_ext3'], dict_auc['tpr_xgb_ext3'], color='#AAD09D',alpha=0.7,
#          lw=lw, label='XGB (AUC = %0.3f)' % dict_auc['roc_auc_xgb_ext3'])
# plt.plot(dict_auc['fpr_svc_ext3'], dict_auc['tpr_svc_ext3'], color='#66BC98',alpha=0.7,
#          lw=lw, label='SVC (AUC = %0.3f)' % dict_auc['roc_auc_svc_ext3'])
# plt.plot(dict_auc['fpr_mlp_ext3'], dict_auc['tpr_mlp_ext3'], color='#73D2D7',alpha=0.7,
#          lw=lw, label='MLP (AUC = %0.3f)' % dict_auc['roc_auc_mlp_ext3'])
# plt.plot(dict_auc['fpr_saint_ext3'], dict_auc['tpr_saint_ext3'], color='#B2D2E8',alpha=0.7,
#          lw=lw, label='SAINT (AUC = %0.3f)' % dict_auc['roc_auc_saint_ext3'])
# plt.plot(dict_auc['fpr_tabnet2_ext3'], dict_auc['tpr_tabnet2_ext3'], color='#71B1D7',alpha=0.7,
#          lw=lw, label='TabNet (AUC = %0.3f)' % dict_auc['roc_auc_tabnet2_ext3'])
# plt.plot(dict_auc['fpr_tabpfn2_ext3'], dict_auc['tpr_tabpfn2_ext3'], color='#9491D7',alpha=0.7,
#          lw=lw, label='TabPFN (AUC = %0.3f)' % dict_auc['roc_auc_tabpfn2_ext3'])
# plt.plot(dict_auc['fpr_transtab2_ext3'], dict_auc['tpr_transtab2_ext3'], color='#7A6BB8',alpha=0.7,
#          lw=lw, label='TransTab (AUC = %0.3f)' % dict_auc['roc_auc_transtab2_ext3'])
# plt.plot(dict_auc['fpr_our_ext3'], dict_auc['tpr_our_ext3'], color='#D24D3E',alpha=1,
#          lw=lw, label='Ours (AUC = %0.3f)' % dict_auc['roc_auc_our_ext3'])
# # plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curves for cohort 3')
# plt.legend(loc="lower right")
# plt.savefig('./results/'+path+'/AUC/ext3.png')
#
# # =====================================================================
