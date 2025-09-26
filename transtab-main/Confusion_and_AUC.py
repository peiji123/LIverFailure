'''
这个代码是多模型对比的AUC曲线绘图代码
结果呈现在 Fig.2 的单个子图
'''

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
from transtab.analysis_utils import confusion_matrix_plots, error_samples, test_2_oridata, error_samples_name
import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.font_manager as fm
from sklearn.utils import resample
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
prop = fm.FontProperties(fname=font_path, size=22)

data_int = pd.read_csv('../data_process/df_int_test.csv')
ori_data_int = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
data_ext12 = pd.read_csv('../data_process/df_ext12.csv')
ori_data_ext12 = pd.read_excel('../data_process/外部验证中心12-601.xlsx')
# data_ext3 = pd.read_csv('../data_process/New_ext_3_240625_best_018.csv')
data_ext3 = pd.read_csv('../data_process/df_New_ext_alin_3_240625_best_018.csv')
# data_ext3_2 = pd.read_csv('../../../data_process/New_0702_ext_3_alin.csv')
ori_data_ext3 = pd.read_csv('../data_process/ori_240625_best_018_new_ext_3.csv')
data_ext3_2 = pd.read_csv('../data_process/New_0702_ext_3_alin.csv')
ori_data_test = ori_data_int.loc[data_int.iloc[:,0],:]
ori_data_test.to_csv('../data_process/ori_test.csv')
data_ext4 = pd.read_csv('../data_process/df_ext4_alin.csv')
ori_data_ext4 = pd.read_excel('../data_process/外部验证中心4.xlsx')

path='LF_bio_240625_best_018'
model_list = ['lr','rf14', 'xgb2','svc','lda','lgb3','mlp', 'saint2',  'tabnet3', 'tabpfn3', 'transtab3', 'our']
# model_list = ['our']
data_list = ['int', 'ext12', 'ext3','ext4']

dict_all_auc = {}
for ml, dt in itertools.product(model_list, data_list):
    model = ml
    data_name = dt

    if data_name == 'int':
        data = data_int
        ori_data = ori_data_test
    elif data_name == 'ext12':
        data = data_ext12
        ori_data = ori_data_ext12
    elif data_name == 'ext3':
        data = data_ext3
        ori_data = ori_data_ext3
    elif data_name == 'ext4':
        data = data_ext4
        ori_data = ori_data_ext4

    dic_path = './Real_Prediction/dict_'+model+'_'+data_name+'_result.json'
    # plot_path = './plots/confusion_matrix_'+model+'_'+data_name+'.png'
    plot_path = './results/'+path+'/Confusion_matrix/'+model+'_'+data_name+'.png'

    dic = json.load(open(dic_path, 'r'))
    # cmp = confusion_matrix_plots(dic['real'], dic['pred_pro'],plot_path)
    _,_,er_list = error_samples_name(data, ori_data, dic['pred_pro'], data['PHLF'], model)

    # with open('./plots/error_type_list.txt','a') as f:
    #     f.write(model + '_' + data_name + '\n')
    #     f.write(str(er_list)+ '\n'+ '\n')

    fpr, tpr, thresholds = roc_curve(dic['real'], dic['pred'])
    roc_auc = auc(fpr, tpr)
    # =================================================================\
    '''
    计算 95% CI
    '''
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_score = []
    bootstrapped_tprs = []
    dic['real'] = np.array(list(map(int, dic['real'])))
    dic['pred'] = np.array(dic['pred'])
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(dic['pred']), len(dic['pred']))
        if len(np.unique(dic['real'][indices])) < 2:
            continue
        score = roc_auc_score(dic['real'][indices], dic['pred'][indices])
        fpr_bs, tpr_bs,_ = roc_curve(dic['real'][indices], dic['pred'][indices])
        bootstrapped_score.append(score)
        bootstrapped_tprs.append(np.interp(np.linspace(0, 1, 100), fpr_bs, tpr_bs))

    sorted_scores = np.array(bootstrapped_score)
    bootstrapped_tprs = np.array(bootstrapped_tprs)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(len(sorted_scores)*0.025)]
    confidence_upper = sorted_scores[int(len(sorted_scores)*0.975)]
    tprs_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
    tprs_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)

    dict_auc = {}
    dict_auc['fpr_'+model+'_'+data_name] = list(fpr)
    dict_auc['tpr_'+model+'_'+data_name] = list(tpr)
    dict_auc['roc_auc_'+model+'_'+data_name] = roc_auc
    dict_auc['95%CI_'+model+'_'+data_name] = ('( {}, {})'.
                      format(round(confidence_lower,3),round(confidence_upper, 3)))
    dict_auc['tprs_'+model+'_'+data_name] = list([list(tprs_upper),list(tprs_lower)])
    dict_all_auc.update(dict_auc)
    with open('./plots/roc_auc_ori.json','w') as f:
        json.dump(dict_all_auc,f)

dict_auc = json.load(open('./plots/roc_auc_ori.json'))
dict_auc_2 = dict_all_auc
lw = 3


plt.figure(figsize=[11, 10])
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.plot(dict_auc['fpr_lr_int'], dict_auc['tpr_lr_int'], color='#FCDC89', alpha=0.7,
         lw=lw, label='LR: AUC = 0.908 (0.848, 0.958)')
plt.plot(dict_auc['fpr_rf14_int'], dict_auc['tpr_rf14_int'], color='#E3EA96',alpha=0.7,
         lw=lw, label='RF: AUC = %0.3f ' % dict_auc['roc_auc_rf14_int']+dict_auc['95%CI_rf14_int'])
plt.plot(dict_auc['fpr_xgb2_int'], dict_auc['tpr_xgb2_int'], color='#AAD09D',alpha=0.7,
         lw=lw, label='XGB: AUC = %0.3f ' % dict_auc['roc_auc_xgb2_int']+dict_auc['95%CI_xgb2_int'])
plt.plot(dict_auc['fpr_svc_int'], dict_auc['tpr_svc_int'], color='#66BC98',alpha=0.7,
         lw=lw, label='SVC: AUC = %0.3f ' % dict_auc['roc_auc_svc_int']+dict_auc['95%CI_svc_int'])

plt.plot(dict_auc_2['fpr_lda_int'], dict_auc_2['tpr_lda_int'], color='#C080C2',alpha=0.7,
         lw=lw, label='LDA: AUC = %0.3f ' % dict_auc_2['roc_auc_lda_int']+dict_auc_2['95%CI_lda_int'])
plt.plot(dict_auc_2['fpr_lgb3_int'], dict_auc_2['tpr_lgb3_int'], color='#A079BD',alpha=0.7,
         lw=lw, label='LGB: AUC = %0.3f ' % dict_auc_2['roc_auc_lgb3_int']+dict_auc_2['95%CI_lgb3_int'])

plt.plot(dict_auc['fpr_mlp_int'], dict_auc['tpr_mlp_int'], color='#73D2D7',alpha=0.7,
         lw=lw, label='MLP: AUC = %0.3f ' % dict_auc['roc_auc_mlp_int']+dict_auc['95%CI_mlp_int'])
plt.plot(dict_auc['fpr_saint2_int'], dict_auc['tpr_saint2_int'], color='#B2D2E8',alpha=0.7,
         lw=lw, label='SAINT: AUC = %0.3f ' % dict_auc['roc_auc_saint2_int']+dict_auc['95%CI_saint2_int'])
plt.plot(dict_auc['fpr_tabnet3_int'], dict_auc['tpr_tabnet3_int'], color='#71B1D7',alpha=0.7,
         lw=lw, label='TabNet: AUC = %0.3f ' % dict_auc['roc_auc_tabnet3_int']+dict_auc['95%CI_tabnet3_int'])
plt.plot(dict_auc['fpr_tabpfn3_int'], dict_auc['tpr_tabpfn3_int'], color='#9491D7',alpha=0.7,
         lw=lw, label='TabPFN: AUC = %0.3f ' % dict_auc['roc_auc_tabpfn3_int']+dict_auc['95%CI_tabpfn3_int'])
plt.plot(dict_auc['fpr_transtab3_int'], dict_auc['tpr_transtab3_int'], color='#7A6BB8',alpha=0.7,
         lw=lw, label='TransTab: AUC = %0.3f ' % dict_auc['roc_auc_transtab3_int']+dict_auc['95%CI_transtab3_int'])
plt.plot(dict_auc['fpr_our_int'], dict_auc['tpr_our_int'], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc['roc_auc_our_int']+dict_auc['95%CI_our_int'])

# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22,fontproperties=prop)
plt.yticks(fontsize=22,fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24,fontproperties=prop)
plt.title('ROC curves for internal validation', fontsize=26, pad=20,
          fontweight='bold',fontproperties=prop)
plt.legend(loc="lower right", prop=prop,fontsize=22)
# plt.legend(loc="lower right", bbox_to_anchor=(1.05, 0.),  # 调整图例位置
#            fontsize=28, prop=prop,
#            frameon=False)
ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
plt.savefig('./results/'+path+'/AUC/int_2.eps')
# plt.show()

plt.figure(figsize=[11, 10])
plt.plot(dict_auc['fpr_lr_ext12'], dict_auc['tpr_lr_ext12'], color='#FCDC89', alpha=0.7,
         lw=lw, label='LR: AUC = 0.758 (0.637, 0.861)')
plt.plot(dict_auc['fpr_rf14_ext12'], dict_auc['tpr_rf14_ext12'], color='#E3EA96',alpha=0.7,
         lw=lw, label='RF: AUC = %0.3f ' % dict_auc['roc_auc_rf14_ext12']+dict_auc['95%CI_rf14_ext12'])
plt.plot(dict_auc['fpr_xgb2_ext12'], dict_auc['tpr_xgb2_ext12'], color='#AAD09D',alpha=0.7,
         lw=lw, label='XGB: AUC = %0.3f ' % dict_auc['roc_auc_xgb2_ext12']+dict_auc['95%CI_xgb2_ext12'])
plt.plot(dict_auc['fpr_svc_ext12'], dict_auc['tpr_svc_ext12'], color='#66BC98',alpha=0.7,
         lw=lw, label='SVC: AUC = %0.3f ' % dict_auc['roc_auc_svc_ext12']+dict_auc['95%CI_svc_ext12'])

plt.plot(dict_auc_2['fpr_lda_ext12'], dict_auc_2['tpr_lda_ext12'], color='#C080C2',alpha=0.7,
         lw=lw, label='LDA: AUC = %0.3f ' % dict_auc_2['roc_auc_lda_ext12']+dict_auc_2['95%CI_lda_ext12'])
plt.plot(dict_auc_2['fpr_lgb3_ext12'], dict_auc_2['tpr_lgb3_ext12'], color='#A079BD',alpha=0.7,
         lw=lw, label='LGB: AUC = %0.3f ' % dict_auc_2['roc_auc_lgb3_ext12']+dict_auc_2['95%CI_lgb3_ext12'])

plt.plot(dict_auc['fpr_mlp_ext12'], dict_auc['tpr_mlp_ext12'], color='#73D2D7',alpha=0.7,
         lw=lw, label='MLP: AUC = %0.3f ' % dict_auc['roc_auc_mlp_ext12']+dict_auc['95%CI_mlp_ext12'])
plt.plot(dict_auc['fpr_saint2_ext12'], dict_auc['tpr_saint2_ext12'], color='#B2D2E8',alpha=0.7,
         lw=lw, label='SAINT: AUC = %0.3f ' % dict_auc['roc_auc_saint2_ext12']+dict_auc['95%CI_saint2_ext12'])
plt.plot(dict_auc['fpr_tabnet3_ext12'], dict_auc['tpr_tabnet3_ext12'], color='#71B1D7',alpha=0.7,
         lw=lw, label='TabNet: AUC = %0.3f ' % dict_auc['roc_auc_tabnet3_ext12']+dict_auc['95%CI_tabnet3_ext12'])
plt.plot(dict_auc['fpr_tabpfn3_ext12'], dict_auc['tpr_tabpfn3_ext12'], color='#9491D7',alpha=0.7,
         lw=lw, label='TabPFN: AUC = %0.3f ' % dict_auc['roc_auc_tabpfn3_ext12']+dict_auc['95%CI_tabpfn3_ext12'])
plt.plot(dict_auc['fpr_transtab3_ext12'], dict_auc['tpr_transtab3_ext12'], color='#7A6BB8',alpha=0.7,
         lw=lw, label='TransTab: AUC = %0.3f ' % dict_auc['roc_auc_transtab3_ext12']+dict_auc['95%CI_transtab3_ext12'])
plt.plot(dict_auc['fpr_our_ext12'], dict_auc['tpr_our_ext12'], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc['roc_auc_our_ext12']+dict_auc['95%CI_our_ext12'])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22,fontproperties=prop)
plt.yticks(fontsize=22,fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24,fontproperties=prop)
plt.title('ROC curves for external validation 1', fontsize=26, pad=20,
          fontweight='bold', fontproperties=prop)
plt.legend(loc="lower right",fontsize=22,prop=prop)
ax = plt.gca()
plt.savefig('./results/'+path+'/AUC/ext12_2.eps')

plt.figure(figsize=[11, 10])
plt.plot(dict_auc['fpr_lr_ext3'], dict_auc['tpr_lr_ext3'], color='#FCDC89', alpha=0.7,
         lw=lw, label='LR: AUC = 0.676 (0.483, 0.862)')
plt.plot(dict_auc['fpr_rf14_ext3'], dict_auc['tpr_rf14_ext3'], color='#E3EA96',alpha=0.7,
         lw=lw, label='RF: AUC = %0.3f ' % dict_auc['roc_auc_rf14_ext3']+dict_auc['95%CI_rf14_ext3'])
plt.plot(dict_auc['fpr_xgb2_ext3'], dict_auc['tpr_xgb2_ext3'], color='#AAD09D',alpha=0.7,
         lw=lw, label='XGB: AUC = %0.3f ' % dict_auc['roc_auc_xgb2_ext3']+dict_auc['95%CI_xgb2_ext3'])
plt.plot(dict_auc['fpr_svc_ext3'], dict_auc['tpr_svc_ext3'], color='#66BC98',alpha=0.7,
         lw=lw, label='SVC: AUC = %0.3f ' % dict_auc['roc_auc_svc_ext3']+dict_auc['95%CI_svc_ext3'])

plt.plot(dict_auc_2['fpr_lda_ext3'], dict_auc_2['tpr_lda_ext3'], color='#C080C2',alpha=0.7,
         lw=lw, label='LDA: AUC = %0.3f ' % dict_auc_2['roc_auc_lda_ext3']+dict_auc_2['95%CI_lda_ext3'])
plt.plot(dict_auc_2['fpr_lgb3_ext3'], dict_auc_2['tpr_lgb3_ext3'], color='#A079BD',alpha=0.7,
         lw=lw, label='LGB: AUC = %0.3f ' % dict_auc_2['roc_auc_lgb3_ext3']+dict_auc_2['95%CI_lgb3_ext3'])

plt.plot(dict_auc['fpr_mlp_ext3'], dict_auc['tpr_mlp_ext3'], color='#73D2D7',alpha=0.7,
         lw=lw, label='MLP: AUC = %0.3f ' % dict_auc['roc_auc_mlp_ext3']+dict_auc['95%CI_mlp_ext3'])
plt.plot(dict_auc['fpr_saint2_ext3'], dict_auc['tpr_saint2_ext3'], color='#B2D2E8',alpha=0.7,
         lw=lw, label='SAINT: AUC = %0.3f ' % dict_auc['roc_auc_saint2_ext3']+dict_auc['95%CI_saint2_ext3'])
plt.plot(dict_auc['fpr_tabnet3_ext3'], dict_auc['tpr_tabnet3_ext3'], color='#71B1D7',alpha=0.7,
         lw=lw, label='TabNet: AUC = %0.3f ' % dict_auc['roc_auc_tabnet3_ext3']+dict_auc['95%CI_tabnet3_ext3'])
plt.plot(dict_auc['fpr_tabpfn3_ext3'], dict_auc['tpr_tabpfn3_ext3'], color='#9491D7',alpha=0.7,
         lw=lw, label='TabPFN: AUC = %0.3f ' % dict_auc['roc_auc_tabpfn3_ext3']+dict_auc['95%CI_tabpfn3_ext3'])
plt.plot(dict_auc['fpr_transtab3_ext3'], dict_auc['tpr_transtab3_ext3'], color='#7A6BB8',alpha=0.7,
         lw=lw, label='TransTab: AUC = %0.3f ' % dict_auc['roc_auc_transtab3_ext3']+dict_auc['95%CI_transtab3_ext3'])
plt.plot(dict_auc['fpr_our_ext3'], dict_auc['tpr_our_ext3'], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc['roc_auc_our_ext3']+dict_auc['95%CI_our_ext3'])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22,fontproperties=prop)
plt.yticks(fontsize=22,fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.title('ROC curves for external validation 2', fontsize=26, pad=20,
          fontweight='bold', fontproperties=prop)
plt.legend(loc="lower right",fontsize=22, prop=prop)
ax = plt.gca()
plt.savefig('./results/'+path+'/AUC/ext3_2.eps')

plt.figure(figsize=[11, 10])
plt.plot(dict_auc['fpr_lr_ext4'], dict_auc['tpr_lr_ext4'], color='#FCDC89', alpha=0.7,
         lw=lw, label='LR: AUC = 0.868 (0.734, 0.971)')
plt.plot(dict_auc['fpr_rf14_ext4'], dict_auc['tpr_rf14_ext4'], color='#E3EA96',alpha=0.7,
         lw=lw, label='RF: AUC = %0.3f ' % dict_auc['roc_auc_rf14_ext4']+dict_auc['95%CI_rf14_ext4'])
plt.plot(dict_auc['fpr_xgb2_ext4'], dict_auc['tpr_xgb2_ext4'], color='#AAD09D',alpha=0.7,
         lw=lw, label='XGB: AUC = %0.3f ' % dict_auc['roc_auc_xgb2_ext4']+dict_auc['95%CI_xgb2_ext4'])
plt.plot(dict_auc['fpr_svc_ext4'], dict_auc['tpr_svc_ext4'], color='#66BC98',alpha=0.7,
         lw=lw, label='SVC: AUC = %0.3f ' % dict_auc['roc_auc_svc_ext4']+dict_auc['95%CI_svc_ext4'])

plt.plot(dict_auc_2['fpr_lda_ext4'], dict_auc_2['tpr_lda_ext4'], color='#C080C2',alpha=0.7,
         lw=lw, label='LDA: AUC = %0.3f ' % dict_auc_2['roc_auc_lda_ext4']+dict_auc_2['95%CI_lda_ext4'])
plt.plot(dict_auc_2['fpr_lgb3_ext4'], dict_auc_2['tpr_lgb3_ext4'], color='#A079BD',alpha=0.7,
         lw=lw, label='LGB: AUC = %0.3f ' % dict_auc_2['roc_auc_lgb3_ext4']+dict_auc_2['95%CI_lgb3_ext4'])

plt.plot(dict_auc['fpr_mlp_ext4'], dict_auc['tpr_mlp_ext4'], color='#73D2D7',alpha=0.7,
         lw=lw, label='MLP: AUC = %0.3f ' % dict_auc['roc_auc_mlp_ext4']+dict_auc['95%CI_mlp_ext4'])
plt.plot(dict_auc['fpr_saint2_ext4'], dict_auc['tpr_saint2_ext4'], color='#B2D2E8',alpha=0.7,
         lw=lw, label='SAINT: AUC = %0.3f ' % dict_auc['roc_auc_saint2_ext4']+dict_auc['95%CI_saint2_ext4'])
plt.plot(dict_auc['fpr_tabnet3_ext4'], dict_auc['tpr_tabnet3_ext4'], color='#71B1D7',alpha=0.7,
         lw=lw, label='TabNet: AUC = %0.3f ' % dict_auc['roc_auc_tabnet3_ext4']+dict_auc['95%CI_tabnet3_ext4'])
plt.plot(dict_auc['fpr_tabpfn3_ext4'], dict_auc['tpr_tabpfn3_ext4'], color='#9491D7',alpha=0.7,
         lw=lw, label='TabPFN: AUC = %0.3f ' % dict_auc['roc_auc_tabpfn3_ext4']+dict_auc['95%CI_tabpfn3_ext4'])
plt.plot(dict_auc['fpr_transtab3_ext4'], dict_auc['tpr_transtab3_ext4'], color='#7A6BB8',alpha=0.7,
         lw=lw, label='TransTab: AUC = %0.3f ' % dict_auc['roc_auc_transtab3_ext4']+dict_auc['95%CI_transtab3_ext4'])
plt.plot(dict_auc['fpr_our_ext4'], dict_auc['tpr_our_ext4'], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc['roc_auc_our_ext4']+dict_auc['95%CI_our_ext4'])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22, fontproperties=prop)
plt.yticks(fontsize=22, fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.title('ROC curves for external validation 3', fontsize=26, pad=20,
          fontweight='bold', fontproperties=prop)
plt.legend(loc="lower right",fontsize=22, prop=prop)
ax = plt.gca()
plt.savefig('./results/'+path+'/AUC/ext4_2.eps')

# =====================================================================


def get_final_result3(prob, pred, label, name, result_cohort):
    auc = roc_auc_score(label, prob)
    # predictions = (prob > thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(label, pred)
    accuracy = accuracy_score(label, pred)
    PPV = tp / (tp + fp) if (tp + fp) > 0 else 0
    NPV = tn / (tn + fn) if (tn + fn) > 0 else 0
    new_row = pd.DataFrame({'model': [name],'AUC': [auc],  'Accuracy': [accuracy],
                            'F1-score': [f1],
                            'Sensitivity': [sensitivity],
                            'Specificity': [specificity], 'PPV': [PPV], 'NPV': [NPV]})
    result_cohort = pd.concat([result_cohort, new_row], ignore_index=True)
    return result_cohort

# result_cohort_int = pd.DataFrame(columns=['model','AUC','Accuracy', 'F1-score', 'Sensitivity',
#                                       'Specificity',  'PPV', 'NPV'])
# result_cohort_ext12 = pd.DataFrame(columns=['model','AUC','Accuracy', 'F1-score', 'Sensitivity',
#                                       'Specificity',  'PPV', 'NPV'])
# result_cohort_ext3 = pd.DataFrame(columns=['model','AUC','Accuracy', 'F1-score', 'Sensitivity',
#                                       'Specificity',  'PPV', 'NPV'])
# result_cohort_ext4 = pd.DataFrame(columns=['model','AUC','Accuracy', 'F1-score', 'Sensitivity',
#                                       'Specificity',  'PPV', 'NPV'])
# for model in model_list:
#     dic_path_int = './Real_Prediction/dict_' + model + '_int_result.json'
#     dic_path_ext12 = './Real_Prediction/dict_' + model + '_ext12_result.json'
#     dic_path_ext3 = './Real_Prediction/dict_' + model + '_ext3_result.json'
#     dic_path_ext4 = './Real_Prediction/dict_' + model + '_ext4_result.json'
#     dic_int = json.load(open(dic_path_int, 'r'))
#     dic_ext12 = json.load(open(dic_path_ext12, 'r'))
#     dic_ext3 = json.load(open(dic_path_ext3, 'r'))
#     dic_ext4 = json.load(open(dic_path_ext4, 'r'))
#     result_cohort_int = get_final_result3(dic_int['pred'], dic_int['pred_pro'], dic_int['real'],
#                                           model, result_cohort_int)
#     result_cohort_ext12 = get_final_result3(dic_ext12['pred'], dic_ext12['pred_pro'], dic_ext12['real'],
#                                           model, result_cohort_ext12)
#     result_cohort_ext3 = get_final_result3(dic_ext3['pred'], dic_ext3['pred_pro'], dic_ext3['real'],
#                                            model,result_cohort_ext3)
#     result_cohort_ext4 = get_final_result3(dic_ext4['pred'], dic_ext4['pred_pro'],dic_ext4['real'],
#                                            model,result_cohort_ext4)
# result_cohort_int.to_csv('./results/'+path+'/AUC/result_model_int.csv', index=False)
# result_cohort_ext12.to_csv('./results/'+path+'/AUC/result_model_ext12.csv', index=False)
# result_cohort_ext3.to_csv('./results/'+path+'/AUC/result_model_ext3.csv', index=False)
# result_cohort_ext4.to_csv('./results/'+path+'/AUC/result_model_ext4.csv', index=False)
#
