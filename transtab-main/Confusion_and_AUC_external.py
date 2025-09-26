'''
这部分代码就是 Confusion_and_AUC.py 的复制，用于major之后额外三个中心数据的AUC曲线绘图
Fig.2 的子图 f,g,i
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

# data_int = pd.read_csv('../data_process/df_int_test.csv')
# ori_data_int = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
# data_ext12 = pd.read_csv('../data_process/df_ext12.csv')
# ori_data_ext12 = pd.read_excel('../data_process/外部验证中心12-601.xlsx')
# data_ext3 = pd.read_csv('../data_process/New_ext_3_240625_best_018.csv')
# ori_data_ext3 = pd.read_csv('../data_process/ori_240625_best_018_new_ext_3.csv')
# ori_data_test = ori_data_int.loc[data_int.iloc[:,0],:]
# ori_data_test.to_csv('../data_process/ori_test.csv')
# data_ext4 = pd.read_csv('../data_process/df_ext4_alin.csv')
# ori_data_ext4 = pd.read_excel('../data_process/外部验证中心4.xlsx')

df_PHLF_mimic = pd.read_csv('../data_process/df_PHLF_mimic.csv')
df_PHLF_ext6 = pd.read_csv('../data_process/df_PHLF_ext6.csv')
df_PHLF_ext7 = pd.read_csv('../data_process/df_PHLF_ext7_few_null.csv')
ori_data_ext6 = pd.read_excel('../data_process/EX6.xlsx')
ori_data_ext7 = pd.read_excel('../data_process/EX7.xlsx')
ori_data_mimic = pd.read_excel('../data_process/mimic.xlsx')

path='LF_bio_240625_best_018'
model_list = ['lr4', 'rf19', 'xgb4', 'svc1', 'lda', 'lgb3', 'mlp5', 'saint3', 'tabnet2', 'tabpfn4','transtab2', 'our']
# model_list = ['our']
data_list = ['mimic', 'ext6','ext7']

dict_all_auc = {}
for ml, dt in itertools.product(model_list, data_list):
    model = ml
    data_name = dt

    if data_name == 'mimic':
        data = df_PHLF_mimic
        ori_data = ori_data_mimic
    elif data_name == 'ext6':
        data = df_PHLF_ext6
        ori_data = ori_data_ext6
    elif data_name == 'ext7':
        data = df_PHLF_ext7
        ori_data = ori_data_ext7

    dic_path = './Real_Prediction/dict_'+model+'_'+data_name+'_result.json'
    # plot_path = './plots/confusion_matrix_'+model+'_'+data_name+'.png'
    plot_path = './results/'+path+'/Confusion_matrix/'+model+'_'+data_name+'.png'

    dic = json.load(open(dic_path, 'r'))
    cmp = confusion_matrix_plots(dic['real'], dic['pred_pro'],plot_path)
    _,_,er_list = error_samples_name(data, ori_data, dic['pred_pro'], data['PHLF'], model)

    with open('./plots/error_type_list.txt','a') as f:
        f.write(model + '_' + data_name + '\n')
        f.write(str(er_list)+ '\n'+ '\n')

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
    with open('./plots/roc_auc.json','w') as f:
        json.dump(dict_all_auc,f)


def all_dict_obtain(model, data_name, dict, dict_all_auc, file_name):
    fpr, tpr, thresholds = roc_curve(dict['real'], dict['pred'])
    roc_auc = auc(fpr, tpr)
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_score = []
    bootstrapped_tprs = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(dict['pred']), len(dict['pred']))
        if len(np.unique(dict['real'][indices])) < 2:
            continue
        score = roc_auc_score(dict['real'][indices], dict['pred'][indices])
        fpr_bs, tpr_bs,_ = roc_curve(dict['real'][indices], dict['pred'][indices])
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
    with open('./plots/'+file_name+'.json','w') as f:
        json.dump(dict_all_auc,f)
    return dict_all_auc

result_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_int_result.json'))
result_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext12_result.json'))
result_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext3_result.json'))
result_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext4_result.json'))
result_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_mimic_result.json'))
result_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext6_result.json'))
result_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext7_result.json'))

dict_alldataset_auc = {}
dic_our = {}
dic_our['real'] = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                   result_ext4_dict['real'] + result_ext6_dict['real'] + result_ext7_dict['real'])
dic_our['real'] = np.array(list(map(int, dic_our['real'])))
dic_our['pred'] = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                   result_ext4_dict['pred'] + result_ext6_dict['pred'] + result_ext7_dict['pred'])
dic_our['pred'] = np.array(dic_our['pred'])

dict_alldataset_auc = all_dict_obtain('our', 'all', dic_our, dict_alldataset_auc,'roc_auc_AllDataset')

dict_all_extdataset_auc = {}
dic_our_ext = {}
dic_our_ext['real'] = (result_ext12_dict['real'] + result_ext3_dict['real'] +
                   result_ext4_dict['real'] + result_ext6_dict['real'] + result_ext7_dict['real'])
dic_our_ext['real'] = np.array(list(map(int, dic_our_ext['real'])))
dic_our_ext['pred'] = (result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                   result_ext4_dict['pred'] + result_ext6_dict['pred'] + result_ext7_dict['pred'])
dic_our_ext['pred'] = np.array(dic_our_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('our', 'all_ext', dic_our_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')

result_lr_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_int_result.json'))
result_lr_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext12_result.json'))
result_lr_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext3_result.json'))
result_lr_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext4_result.json'))
result_lr_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_mimic_result.json'))
result_lr_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext6_result.json'))
result_lr_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext7_result.json'))

dic_lr = {}
dic_lr['real'] = (result_lr_int_dict['real'] + result_lr_ext12_dict['real'] + result_lr_ext3_dict['real'] +
                   result_lr_ext4_dict['real'] + result_lr_ext6_dict['real'] + result_lr_ext7_dict['real'])
dic_lr['real'] = np.array(list(map(int, dic_lr['real'])))
dic_lr['pred'] = (result_lr_int_dict['pred'] + result_lr_ext12_dict['pred'] + result_lr_ext3_dict['pred'] +
                   result_lr_ext4_dict['pred'] + result_lr_ext6_dict['pred'] + result_lr_ext7_dict['pred'])
dic_lr['pred'] = np.array(dic_lr['pred'])

dic_lr_ext = {}
dic_lr_ext['real'] = (result_lr_ext12_dict['real'] + result_lr_ext3_dict['real'] +
                   result_lr_ext4_dict['real'] + result_lr_ext6_dict['real'] + result_lr_ext7_dict['real'])
dic_lr_ext['real'] = np.array(list(map(int, dic_lr_ext['real'])))
dic_lr_ext['pred'] = (result_lr_ext12_dict['pred'] + result_lr_ext3_dict['pred'] +
                   result_lr_ext4_dict['pred'] + result_lr_ext6_dict['pred'] + result_lr_ext7_dict['pred'])
dic_lr_ext['pred'] = np.array(dic_lr_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('lr', 'all_ext', dic_lr_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')

dict_alldataset_auc = all_dict_obtain('lr', 'all', dic_lr, dict_alldataset_auc,'roc_auc_AllDataset')



result_rf_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf14_int_result.json'))
result_rf_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf14_ext12_result.json'))
result_rf_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf14_ext3_result.json'))
result_rf_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf14_ext4_result.json'))
result_rf_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf19_mimic_result.json'))
result_rf_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf19_ext6_result.json'))
result_rf_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf19_ext7_result.json'))

dic_rf = {}
dic_rf['real'] = (result_rf_int_dict['real'] + result_rf_ext12_dict['real'] + result_rf_ext3_dict['real'] +
                   result_rf_ext4_dict['real'] + result_rf_ext6_dict['real'] + result_rf_ext7_dict['real'])
dic_rf['real'] = np.array(list(map(int, dic_rf['real'])))
dic_rf['pred'] = (result_rf_int_dict['pred'] + result_rf_ext12_dict['pred'] + result_rf_ext3_dict['pred'] +
                   result_rf_ext4_dict['pred'] + result_rf_ext6_dict['pred'] + result_rf_ext7_dict['pred'])
dic_rf['pred'] = np.array(dic_rf['pred'])

dict_alldataset_auc = all_dict_obtain('rf', 'all', dic_rf, dict_alldataset_auc,'roc_auc_AllDataset')

dic_rf_ext = {}
dic_rf_ext['real'] = (result_rf_ext12_dict['real'] + result_rf_ext3_dict['real'] +
                   result_rf_ext4_dict['real'] + result_rf_ext6_dict['real'] + result_rf_ext7_dict['real'])
dic_rf_ext['real'] = np.array(list(map(int, dic_rf_ext['real'])))
dic_rf_ext['pred'] = (result_rf_ext12_dict['pred'] + result_rf_ext3_dict['pred'] +
                   result_rf_ext4_dict['pred'] + result_rf_ext6_dict['pred'] + result_rf_ext7_dict['pred'])
dic_rf_ext['pred'] = np.array(dic_rf_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('rf', 'all_ext', dic_rf_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')

result_xgb_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb2_int_result.json'))
result_xgb_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb2_ext12_result.json'))
result_xgb_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb2_ext3_result.json'))
result_xgb_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb2_ext4_result.json'))
result_xgb_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb4_mimic_result.json'))
result_xgb_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb4_ext6_result.json'))
result_xgb_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb4_ext7_result.json'))

dic_xgb = {}
dic_xgb['real'] = (result_xgb_int_dict['real'] + result_xgb_ext12_dict['real'] + result_xgb_ext3_dict['real'] +
                   result_xgb_ext4_dict['real'] + result_xgb_ext6_dict['real'] + result_xgb_ext7_dict['real'])
dic_xgb['real'] = np.array(list(map(int, dic_xgb['real'])))
dic_xgb['pred'] = (result_xgb_int_dict['pred'] + result_xgb_ext12_dict['pred'] + result_xgb_ext3_dict['pred'] +
                   result_xgb_ext4_dict['pred'] + result_xgb_ext6_dict['pred'] + result_xgb_ext7_dict['pred'])
dic_xgb['pred'] = np.array(dic_xgb['pred'])

dict_alldataset_auc = all_dict_obtain('xgb', 'all', dic_xgb, dict_alldataset_auc,'roc_auc_AllDataset')

dic_xgb_ext = {}
dic_xgb_ext['real'] = (result_xgb_ext12_dict['real'] + result_xgb_ext3_dict['real'] +
                   result_xgb_ext4_dict['real'] + result_xgb_ext6_dict['real'] + result_xgb_ext7_dict['real'])
dic_xgb_ext['real'] = np.array(list(map(int, dic_xgb_ext['real'])))
dic_xgb_ext['pred'] = (result_xgb_ext12_dict['pred'] + result_xgb_ext3_dict['pred'] +
                   result_xgb_ext4_dict['pred'] + result_xgb_ext6_dict['pred'] + result_xgb_ext7_dict['pred'])
dic_xgb_ext['pred'] = np.array(dic_xgb_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('xgb', 'all_ext', dic_xgb_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')

result_svc_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_int_result.json'))
result_svc_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext12_result.json'))
result_svc_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext3_result.json'))
result_svc_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext4_result.json'))
result_svc_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_mimic_result.json'))
result_svc_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext6_result.json'))
result_svc_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext7_result.json'))

dic_svc = {}
dic_svc['real'] = (result_svc_int_dict['real'] + result_svc_ext12_dict['real'] + result_svc_ext3_dict['real'] +
                   result_svc_ext4_dict['real'] + result_svc_ext6_dict['real'] + result_svc_ext7_dict['real'])
dic_svc['real'] = np.array(list(map(int, dic_svc['real'])))
dic_svc['pred'] = (result_svc_int_dict['pred'] + result_svc_ext12_dict['pred'] + result_svc_ext3_dict['pred'] +
                   result_svc_ext4_dict['pred'] + result_svc_ext6_dict['pred'] + result_svc_ext7_dict['pred'])
dic_svc['pred'] = np.array(dic_svc['pred'])

dict_alldataset_auc = all_dict_obtain('svc', 'all', dic_svc, dict_alldataset_auc,'roc_auc_AllDataset')

dic_svc_ext = {}
dic_svc_ext['real'] = (result_svc_ext12_dict['real'] + result_svc_ext3_dict['real'] +
                   result_svc_ext4_dict['real'] + result_svc_ext6_dict['real'] + result_svc_ext7_dict['real'])
dic_svc_ext['real'] = np.array(list(map(int, dic_svc_ext['real'])))
dic_svc_ext['pred'] = (result_svc_ext12_dict['pred'] + result_svc_ext3_dict['pred'] +
                   result_svc_ext4_dict['pred'] + result_svc_ext6_dict['pred'] + result_svc_ext7_dict['pred'])
dic_svc_ext['pred'] = np.array(dic_svc_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('svc', 'all_ext', dic_svc_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')

result_lda_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_int_result.json'))
result_lda_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext12_result.json'))
result_lda_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext3_result.json'))
result_lda_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext4_result.json'))
result_lda_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_mimic_result.json'))
result_lda_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext6_result.json'))
result_lda_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext7_result.json'))

dic_lda = {}
dic_lda['real'] = (result_lda_int_dict['real'] + result_lda_ext12_dict['real'] + result_lda_ext3_dict['real'] +
                   result_lda_ext4_dict['real'] + result_lda_ext6_dict['real'] + result_lda_ext7_dict['real'])
dic_lda['real'] = np.array(list(map(int, dic_lda['real'])))
dic_lda['pred'] = (result_lda_int_dict['pred'] + result_lda_ext12_dict['pred'] + result_lda_ext3_dict['pred'] +
                   result_lda_ext4_dict['pred'] + result_lda_ext6_dict['pred'] + result_lda_ext7_dict['pred'])
dic_lda['pred'] = np.array(dic_lda['pred'])

dict_alldataset_auc = all_dict_obtain('lda', 'all', dic_lda, dict_alldataset_auc,'roc_auc_AllDataset')

dic_lda_ext = {}
dic_lda_ext['real'] = (result_lda_ext12_dict['real'] + result_lda_ext3_dict['real'] +
                   result_lda_ext4_dict['real'] + result_lda_ext6_dict['real'] + result_lda_ext7_dict['real'])
dic_lda_ext['real'] = np.array(list(map(int, dic_lda_ext['real'])))
dic_lda_ext['pred'] = (result_lda_ext12_dict['pred'] + result_lda_ext3_dict['pred'] +
                   result_lda_ext4_dict['pred'] + result_lda_ext6_dict['pred'] + result_lda_ext7_dict['pred'])
dic_lda_ext['pred'] = np.array(dic_lda_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('lda', 'all_ext', dic_lda_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')


result_lgb_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_int_result.json'))
result_lgb_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext12_result.json'))
result_lgb_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext3_result.json'))
result_lgb_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext4_result.json'))
result_lgb_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_mimic_result.json'))
result_lgb_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext6_result.json'))
result_lgb_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext7_result.json'))

dic_lgb = {}
dic_lgb['real'] = (result_lgb_int_dict['real'] + result_lgb_ext12_dict['real'] + result_lgb_ext3_dict['real'] +
                   result_lgb_ext4_dict['real'] + result_lgb_ext6_dict['real'] + result_lgb_ext7_dict['real'])
dic_lgb['real'] = np.array(list(map(int, dic_lgb['real'])))
dic_lgb['pred'] = (result_lgb_int_dict['pred'] + result_lgb_ext12_dict['pred'] + result_lgb_ext3_dict['pred'] +
                   result_lgb_ext4_dict['pred'] + result_lgb_ext6_dict['pred'] + result_lgb_ext7_dict['pred'])
dic_lgb['pred'] = np.array(dic_lgb['pred'])

dict_alldataset_auc = all_dict_obtain('lgb', 'all', dic_lgb, dict_alldataset_auc,'roc_auc_AllDataset')

dic_lgb_ext = {}
dic_lgb_ext['real'] = (result_lgb_ext12_dict['real'] + result_lgb_ext3_dict['real'] +
                   result_lgb_ext4_dict['real'] + result_lgb_ext6_dict['real'] + result_lgb_ext7_dict['real'])
dic_lgb_ext['real'] = np.array(list(map(int, dic_lgb_ext['real'])))
dic_lgb_ext['pred'] = (result_lgb_ext12_dict['pred'] + result_lgb_ext3_dict['pred'] +
                   result_lgb_ext4_dict['pred'] + result_lgb_ext6_dict['pred'] + result_lgb_ext7_dict['pred'])
dic_lgb_ext['pred'] = np.array(dic_lgb_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('lgb', 'all_ext', dic_lgb_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')


result_mlp_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp_int_result.json'))
result_mlp_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp_ext12_result.json'))
result_mlp_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp_ext3_result.json'))
result_mlp_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp_ext4_result.json'))
result_mlp_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp5_mimic_result.json'))
result_mlp_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp5_ext6_result.json'))
result_mlp_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp5_ext7_result.json'))

dic_mlp = {}
dic_mlp['real'] = (result_mlp_int_dict['real'] + result_mlp_ext12_dict['real'] + result_mlp_ext3_dict['real'] +
                   result_mlp_ext4_dict['real'] + result_mlp_ext6_dict['real'] + result_mlp_ext7_dict['real'])
dic_mlp['real'] = np.array(list(map(int, dic_mlp['real'])))
dic_mlp['pred'] = (result_mlp_int_dict['pred'] + result_mlp_ext12_dict['pred'] + result_mlp_ext3_dict['pred'] +
                   result_mlp_ext4_dict['pred'] + result_mlp_ext6_dict['pred'] + result_mlp_ext7_dict['pred'])
dic_mlp['pred'] = np.array(dic_mlp['pred'])

dict_alldataset_auc = all_dict_obtain('mlp', 'all', dic_mlp, dict_alldataset_auc,'roc_auc_AllDataset')

dic_mlp_ext = {}
dic_mlp_ext['real'] = (result_mlp_ext12_dict['real'] + result_mlp_ext3_dict['real'] +
                   result_mlp_ext4_dict['real'] + result_mlp_ext6_dict['real'] + result_mlp_ext7_dict['real'])
dic_mlp_ext['real'] = np.array(list(map(int, dic_mlp_ext['real'])))
dic_mlp_ext['pred'] = (result_mlp_ext12_dict['pred'] + result_mlp_ext3_dict['pred'] +
                   result_mlp_ext4_dict['pred'] + result_mlp_ext6_dict['pred'] + result_mlp_ext7_dict['pred'])
dic_mlp_ext['pred'] = np.array(dic_mlp_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('mlp', 'all_ext', dic_mlp_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')

result_saint_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint2_int_result.json'))
result_saint_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint2_ext12_result.json'))
result_saint_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint2_ext3_result.json'))
result_saint_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint2_ext4_result.json'))
result_saint_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint3_mimic_result.json'))
result_saint_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint3_ext6_result.json'))
result_saint_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint3_ext7_result.json'))

dic_saint = {}
dic_saint['real'] = (result_saint_int_dict['real'] + result_saint_ext12_dict['real'] + result_saint_ext3_dict['real'] +
                   result_saint_ext4_dict['real'] + result_saint_ext6_dict['real'] + result_saint_ext7_dict['real'])
dic_saint['real'] = np.array(list(map(int, dic_saint['real'])))
dic_saint['pred'] = (result_saint_int_dict['pred'] + result_saint_ext12_dict['pred'] + result_saint_ext3_dict['pred'] +
                   result_saint_ext4_dict['pred'] + result_saint_ext6_dict['pred'] + result_saint_ext7_dict['pred'])
dic_saint['pred'] = np.array(dic_saint['pred'])

dict_alldataset_auc = all_dict_obtain('saint', 'all', dic_saint, dict_alldataset_auc,'roc_auc_AllDataset')


dic_saint_ext = {}
dic_saint_ext['real'] = (result_saint_ext12_dict['real'] + result_saint_ext3_dict['real'] +
                   result_saint_ext4_dict['real'] + result_saint_ext6_dict['real'] + result_saint_ext7_dict['real'])
dic_saint_ext['real'] = np.array(list(map(int, dic_saint_ext['real'])))
dic_saint_ext['pred'] = (result_saint_ext12_dict['pred'] + result_saint_ext3_dict['pred'] +
                   result_saint_ext4_dict['pred'] + result_saint_ext6_dict['pred'] + result_saint_ext7_dict['pred'])
dic_saint_ext['pred'] = np.array(dic_saint_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('saint', 'all_ext', dic_saint_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')

result_tabnet_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_int_result.json'))
result_tabnet_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext12_result.json'))
result_tabnet_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext3_result.json'))
result_tabnet_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext4_result.json'))
result_tabnet_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_mimic_result.json'))
result_tabnet_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext6_result.json'))
result_tabnet_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext7_result.json'))

dic_tabnet = {}
dic_tabnet['real'] = (result_tabnet_int_dict['real'] + result_tabnet_ext12_dict['real'] + result_tabnet_ext3_dict['real'] +
                   result_tabnet_ext4_dict['real'] + result_tabnet_ext6_dict['real'] + result_tabnet_ext7_dict['real'])
dic_tabnet['real'] = np.array(list(map(int, dic_tabnet['real'])))
dic_tabnet['pred'] = (result_tabnet_int_dict['pred'] + result_tabnet_ext12_dict['pred'] + result_tabnet_ext3_dict['pred'] +
                   result_tabnet_ext4_dict['pred'] + result_tabnet_ext6_dict['pred'] + result_tabnet_ext7_dict['pred'])
dic_tabnet['pred'] = np.array(dic_tabnet['pred'])

dict_alldataset_auc = all_dict_obtain('tabnet', 'all', dic_tabnet, dict_alldataset_auc,'roc_auc_AllDataset')

dic_tabnet_ext = {}
dic_tabnet_ext['real'] = (result_tabnet_ext12_dict['real'] + result_tabnet_ext3_dict['real'] +
                   result_tabnet_ext4_dict['real'] + result_tabnet_ext6_dict['real'] + result_tabnet_ext7_dict['real'])
dic_tabnet_ext['real'] = np.array(list(map(int, dic_tabnet_ext['real'])))
dic_tabnet_ext['pred'] = (result_tabnet_ext12_dict['pred'] + result_tabnet_ext3_dict['pred'] +
                   result_tabnet_ext4_dict['pred'] + result_tabnet_ext6_dict['pred'] + result_tabnet_ext7_dict['pred'])
dic_tabnet_ext['pred'] = np.array(dic_tabnet_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('tabnet', 'all_ext', dic_tabnet_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')


result_tabpfn_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn3_int_result.json'))
result_tabpfn_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn3_ext12_result.json'))
result_tabpfn_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn3_ext3_result.json'))
result_tabpfn_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn3_ext4_result.json'))
result_tabpfn_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn4_mimic_result.json'))
result_tabpfn_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn4_ext6_result.json'))
result_tabpfn_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn4_ext7_result.json'))

dic_tabpfn = {}
dic_tabpfn['real'] = (result_tabpfn_int_dict['real'] + result_tabpfn_ext12_dict['real'] + result_tabpfn_ext3_dict['real'] +
                   result_tabpfn_ext4_dict['real'] + result_tabpfn_ext6_dict['real'] + result_tabpfn_ext7_dict['real'])
dic_tabpfn['real'] = np.array(list(map(int, dic_tabpfn['real'])))
dic_tabpfn['pred'] = (result_tabpfn_int_dict['pred'] + result_tabpfn_ext12_dict['pred'] + result_tabpfn_ext3_dict['pred'] +
                   result_tabpfn_ext4_dict['pred'] + result_tabpfn_ext6_dict['pred'] + result_tabpfn_ext7_dict['pred'])
dic_tabpfn['pred'] = np.array(dic_tabpfn['pred'])

dict_alldataset_auc = all_dict_obtain('tabpfn', 'all', dic_tabpfn, dict_alldataset_auc,'roc_auc_AllDataset')


dic_tabpfn_ext = {}
dic_tabpfn_ext['real'] = (result_tabpfn_ext12_dict['real'] + result_tabpfn_ext3_dict['real'] +
                   result_tabpfn_ext4_dict['real'] + result_tabpfn_ext6_dict['real'] + result_tabpfn_ext7_dict['real'])
dic_tabpfn_ext['real'] = np.array(list(map(int, dic_tabpfn_ext['real'])))
dic_tabpfn_ext['pred'] = (result_tabpfn_ext12_dict['pred'] + result_tabpfn_ext3_dict['pred'] +
                   result_tabpfn_ext4_dict['pred'] + result_tabpfn_ext6_dict['pred'] + result_tabpfn_ext7_dict['pred'])
dic_tabpfn_ext['pred'] = np.array(dic_tabpfn_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('tabpfn', 'all_ext', dic_tabpfn_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')


result_transtab_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab3_int_result.json'))
result_transtab_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab3_ext12_result.json'))
result_transtab_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab3_ext3_result.json'))
result_transtab_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab3_ext4_result.json'))
result_transtab_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab2_mimic_result.json'))
result_transtab_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab2_ext6_result.json'))
result_transtab_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab2_ext7_result.json'))

dic_transtab = {}
dic_transtab['real'] = (result_transtab_int_dict['real'] + result_transtab_ext12_dict['real'] + result_transtab_ext3_dict['real'] +
                   result_transtab_ext4_dict['real'] + result_transtab_ext6_dict['real'] + result_transtab_ext7_dict['real'])
dic_transtab['real'] = np.array(list(map(int, dic_transtab['real'])))
dic_transtab['pred'] = (result_transtab_int_dict['pred'] + result_transtab_ext12_dict['pred'] + result_transtab_ext3_dict['pred'] +
                   result_transtab_ext4_dict['pred'] + result_transtab_ext6_dict['pred'] + result_transtab_ext7_dict['pred'])
dic_transtab['pred'] = np.array(dic_transtab['pred'])

dict_alldataset_auc = all_dict_obtain('transtab', 'all', dic_transtab, dict_alldataset_auc,'roc_auc_AllDataset')

dic_transtab_ext = {}
dic_transtab_ext['real'] = (result_transtab_ext12_dict['real'] + result_transtab_ext3_dict['real'] +
                   result_transtab_ext4_dict['real'] + result_transtab_ext6_dict['real'] + result_transtab_ext7_dict['real'])
dic_transtab_ext['real'] = np.array(list(map(int, dic_transtab_ext['real'])))
dic_transtab_ext['pred'] = (result_transtab_ext12_dict['pred'] + result_transtab_ext3_dict['pred'] +
                   result_transtab_ext4_dict['pred'] + result_transtab_ext6_dict['pred'] + result_transtab_ext7_dict['pred'])
dic_transtab_ext['pred'] = np.array(dic_transtab_ext['pred'])

dict_all_extdataset_auc = all_dict_obtain('transtab', 'all_ext', dic_transtab_ext, dict_all_extdataset_auc,'roc_auc_All_extDataset')

dict_all_auc = {}
model = 'our'
data_name = 'all'
fpr, tpr, thresholds = roc_curve(dic_our['real'], dic_our['pred'])
roc_auc = auc(fpr, tpr)

dict_all_auc_ext = {}
model = 'our'
data_name_ext = 'all_ext'
fpr_ext, tpr_ext, thresholds_ext = roc_curve(dic_our_ext['real'], dic_our_ext['pred'])
roc_auc_ext = auc(fpr, tpr)


dict_auc = json.load(open('./plots/roc_auc.json'))
dict_auc_alldataset = json.load(open('./plots/roc_auc_AllDataset.json'))

dict_auc_ext = json.load(open('./plots/roc_auc.json'))
dict_auc_all_extdataset = json.load(open('./plots/roc_auc_All_extDataset.json'))
lw = 3

# =====================================================================
dataset = 'ext6'
model_list = ['lr4', 'rf19', 'xgb4', 'svc1', 'lda', 'lgb3', 'mlp5', 'saint3', 'tabnet2', 'tabpfn4','transtab2', 'our']

plt.figure(figsize=[11, 10])
plt.plot(dict_auc['fpr_'+model_list[0]+'_'+dataset], dict_auc['tpr_'+model_list[0]+'_'+dataset],
         color='#FCDC89', alpha=0.7, lw=lw,
         label='LR: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[0]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[0]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[1]+'_'+dataset], dict_auc['tpr_'+model_list[1]+'_'+dataset],
         color='#E3EA96',alpha=0.7, lw=lw,
         label='RF: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[1]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[1]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[2]+'_'+dataset], dict_auc['tpr_'+model_list[2]+'_'+dataset],
         color='#AAD09D',alpha=0.7, lw=lw,
         label='XGB: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[2]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[2]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[3]+'_'+dataset], dict_auc['tpr_'+model_list[3]+'_'+dataset],
         color='#66BC98',alpha=0.7, lw=lw,
         label='SVC: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[3]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[3]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[4]+'_'+dataset], dict_auc['tpr_'+model_list[4]+'_'+dataset],
         color='#C080C2',alpha=0.7, lw=lw,
         label='LDA: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[4]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[4]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[5]+'_'+dataset], dict_auc['tpr_'+model_list[5]+'_'+dataset],
         color='#A079BD',alpha=0.7, lw=lw,
         label='GBM: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[5]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[5]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[6]+'_'+dataset], dict_auc['tpr_'+model_list[6]+'_'+dataset],
         color='#73D2D7',alpha=0.7, lw=lw,
         label='MLP: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[6]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[6]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[7]+'_'+dataset], dict_auc['tpr_'+model_list[7]+'_'+dataset],
         color='#B2D2E8',alpha=0.7, lw=lw,
         label='SAINT: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[7]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[7]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[8]+'_'+dataset], dict_auc['tpr_'+model_list[8]+'_'+dataset],
         color='#71B1D7',alpha=0.7, lw=lw,
         label='TabNet: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[8]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[8]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[9]+'_'+dataset], dict_auc['tpr_'+model_list[9]+'_'+dataset],
         color='#9491D7',alpha=0.7, lw=lw,
         label='TabPFN: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[9]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[9]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[10]+'_'+dataset], dict_auc['tpr_'+model_list[10]+'_'+dataset],
         color='#7A6BB8',alpha=0.7,lw=lw,
         label='TransTab: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[10]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[10]+'_'+dataset])

plt.plot(dict_auc['fpr_our_'+dataset], dict_auc['tpr_our_'+dataset], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc['roc_auc_our_'+dataset]+dict_auc['95%CI_our_'+dataset])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22, fontproperties=prop)
plt.yticks(fontsize=22, fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.title('ROC curves for external validation 6', fontsize=26, pad=20,
          fontweight='bold', fontproperties=prop)
plt.legend(loc="lower right",fontsize=22, prop=prop)
ax = plt.gca()
# plt.savefig('./results/'+path+'/AUC/ext4.eps')

# =====================================================================
dataset = 'ext7'
model_list = ['lr4', 'rf19', 'xgb4', 'svc1', 'lda', 'lgb3', 'mlp5', 'saint3', 'tabnet2', 'tabpfn4','transtab2', 'our']

plt.figure(figsize=[11, 10])
plt.plot(dict_auc['fpr_'+model_list[0]+'_'+dataset], dict_auc['tpr_'+model_list[0]+'_'+dataset],
         color='#FCDC89', alpha=0.7, lw=lw,
         label='LR: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[0]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[0]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[1]+'_'+dataset], dict_auc['tpr_'+model_list[1]+'_'+dataset],
         color='#E3EA96',alpha=0.7, lw=lw,
         label='RF: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[1]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[1]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[2]+'_'+dataset], dict_auc['tpr_'+model_list[2]+'_'+dataset],
         color='#AAD09D',alpha=0.7, lw=lw,
         label='XGB: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[2]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[2]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[3]+'_'+dataset], dict_auc['tpr_'+model_list[3]+'_'+dataset],
         color='#66BC98',alpha=0.7, lw=lw,
         label='SVC: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[3]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[3]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[4]+'_'+dataset], dict_auc['tpr_'+model_list[4]+'_'+dataset],
         color='#C080C2',alpha=0.7, lw=lw,
         label='LDA: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[4]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[4]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[5]+'_'+dataset], dict_auc['tpr_'+model_list[5]+'_'+dataset],
         color='#A079BD',alpha=0.7, lw=lw,
         label='GBM: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[5]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[5]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[6]+'_'+dataset], dict_auc['tpr_'+model_list[6]+'_'+dataset],
         color='#73D2D7',alpha=0.7, lw=lw,
         label='MLP: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[6]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[6]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[7]+'_'+dataset], dict_auc['tpr_'+model_list[7]+'_'+dataset],
         color='#B2D2E8',alpha=0.7, lw=lw,
         label='SAINT: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[7]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[7]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[8]+'_'+dataset], dict_auc['tpr_'+model_list[8]+'_'+dataset],
         color='#71B1D7',alpha=0.7, lw=lw,
         label='TabNet: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[8]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[8]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[9]+'_'+dataset], dict_auc['tpr_'+model_list[9]+'_'+dataset],
         color='#9491D7',alpha=0.7, lw=lw,
         label='TabPFN: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[9]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[9]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[10]+'_'+dataset], dict_auc['tpr_'+model_list[10]+'_'+dataset],
         color='#7A6BB8',alpha=0.7,lw=lw,
         label='TransTab: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[10]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[10]+'_'+dataset])

plt.plot(dict_auc['fpr_our_'+dataset], dict_auc['tpr_our_'+dataset], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc['roc_auc_our_'+dataset]+dict_auc['95%CI_our_'+dataset])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22, fontproperties=prop)
plt.yticks(fontsize=22, fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.title('ROC curves for external validation 7', fontsize=26, pad=20,
          fontweight='bold', fontproperties=prop)
plt.legend(loc="lower right",fontsize=22, prop=prop)
ax = plt.gca()
# plt.savefig('./results/'+path+'/AUC/ext5.eps')

# =====================================================================
dataset = 'mimic'
model_list = ['lr4', 'rf19', 'xgb4', 'svc1', 'lda', 'lgb3', 'mlp5', 'saint3', 'tabnet2', 'tabpfn4','transtab2', 'our']

plt.figure(figsize=[11, 10])
plt.plot(dict_auc['fpr_'+model_list[0]+'_'+dataset], dict_auc['tpr_'+model_list[0]+'_'+dataset],
         color='#FCDC89', alpha=0.7, lw=lw,
         label='LR: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[0]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[0]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[1]+'_'+dataset], dict_auc['tpr_'+model_list[1]+'_'+dataset],
         color='#E3EA96',alpha=0.7, lw=lw,
         label='RF: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[1]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[1]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[2]+'_'+dataset], dict_auc['tpr_'+model_list[2]+'_'+dataset],
         color='#AAD09D',alpha=0.7, lw=lw,
         label='XGB: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[2]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[2]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[3]+'_'+dataset], dict_auc['tpr_'+model_list[3]+'_'+dataset],
         color='#66BC98',alpha=0.7, lw=lw,
         label='SVC: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[3]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[3]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[4]+'_'+dataset], dict_auc['tpr_'+model_list[4]+'_'+dataset],
         color='#C080C2',alpha=0.7, lw=lw,
         label='LDA: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[4]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[4]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[5]+'_'+dataset], dict_auc['tpr_'+model_list[5]+'_'+dataset],
         color='#A079BD',alpha=0.7, lw=lw,
         label='GBM: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[5]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[5]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[6]+'_'+dataset], dict_auc['tpr_'+model_list[6]+'_'+dataset],
         color='#73D2D7',alpha=0.7, lw=lw,
         label='MLP: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[6]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[6]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[7]+'_'+dataset], dict_auc['tpr_'+model_list[7]+'_'+dataset],
         color='#B2D2E8',alpha=0.7, lw=lw,
         label='SAINT: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[7]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[7]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[8]+'_'+dataset], dict_auc['tpr_'+model_list[8]+'_'+dataset],
         color='#71B1D7',alpha=0.7, lw=lw,
         label='TabNet: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[8]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[8]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[9]+'_'+dataset], dict_auc['tpr_'+model_list[9]+'_'+dataset],
         color='#9491D7',alpha=0.7, lw=lw,
         label='TabPFN: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[9]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[9]+'_'+dataset])

plt.plot(dict_auc['fpr_'+model_list[10]+'_'+dataset], dict_auc['tpr_'+model_list[10]+'_'+dataset],
         color='#7A6BB8',alpha=0.7,lw=lw,
         label='TransTab: AUC = %0.3f ' % dict_auc['roc_auc_'+model_list[10]+'_'+dataset]+
               dict_auc['95%CI_'+model_list[10]+'_'+dataset])

plt.plot(dict_auc['fpr_our_'+dataset], dict_auc['tpr_our_'+dataset], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc['roc_auc_our_'+dataset]+dict_auc['95%CI_our_'+dataset])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22, fontproperties=prop)
plt.yticks(fontsize=22, fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.title('ROC curves for mimic', fontsize=26, pad=20,
          fontweight='bold', fontproperties=prop)
plt.legend(loc="lower right",fontsize=22, prop=prop)
ax = plt.gca()
# plt.savefig('./results/'+path+'/AUC/mimic.eps')

dataset = 'all'
model_list = ['lr', 'rf', 'xgb', 'svc', 'lda', 'lgb', 'mlp', 'saint', 'tabnet', 'tabpfn','transtab', 'our']

plt.figure(figsize=[11, 10])
plt.plot(dict_auc_alldataset['fpr_'+model_list[0]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[0]+'_'+dataset],
         color='#FCDC89', alpha=0.7, lw=lw,
         label='LR: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[0]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[0]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[1]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[1]+'_'+dataset],
         color='#E3EA96',alpha=0.7, lw=lw,
         label='RF: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[1]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[1]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[2]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[2]+'_'+dataset],
         color='#AAD09D',alpha=0.7, lw=lw,
         label='XGB: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[2]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[2]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[3]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[3]+'_'+dataset],
         color='#66BC98',alpha=0.7, lw=lw,
         label='SVC: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[3]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[3]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[4]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[4]+'_'+dataset],
         color='#C080C2',alpha=0.7, lw=lw,
         label='LDA: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[4]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[4]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[5]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[5]+'_'+dataset],
         color='#A079BD',alpha=0.7, lw=lw,
         label='GBM: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[5]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[5]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[6]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[6]+'_'+dataset],
         color='#73D2D7',alpha=0.7, lw=lw,
         label='MLP: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[6]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[6]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[7]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[7]+'_'+dataset],
         color='#B2D2E8',alpha=0.7, lw=lw,
         label='SAINT: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[7]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[7]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[8]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[8]+'_'+dataset],
         color='#71B1D7',alpha=0.7, lw=lw,
         label='TabNet: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[8]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[8]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[9]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[9]+'_'+dataset],
         color='#9491D7',alpha=0.7, lw=lw,
         label='TabPFN: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[9]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[9]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_'+model_list[10]+'_'+dataset], dict_auc_alldataset['tpr_'+model_list[10]+'_'+dataset],
         color='#7A6BB8',alpha=0.7,lw=lw,
         label='TransTab: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_'+model_list[10]+'_'+dataset]+
               dict_auc_alldataset['95%CI_'+model_list[10]+'_'+dataset])

plt.plot(dict_auc_alldataset['fpr_our_'+dataset], dict_auc_alldataset['tpr_our_'+dataset], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc_alldataset['roc_auc_our_'+dataset]+dict_auc_alldataset['95%CI_our_'+dataset])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22, fontproperties=prop)
plt.yticks(fontsize=22, fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.title('ROC curves for all Chinese validation dataset', fontsize=26, pad=20,
          fontweight='bold', fontproperties=prop)
plt.legend(loc="lower right",fontsize=22, prop=prop)
ax = plt.gca()
# plt.savefig('./results/'+path+'/AUC/all_dataset.eps')

dataset = 'all_ext'
model_list = ['lr', 'rf', 'xgb', 'svc', 'lda', 'lgb', 'mlp', 'saint', 'tabnet', 'tabpfn','transtab', 'our']

plt.figure(figsize=[11, 10])
plt.plot(dict_auc_all_extdataset['fpr_'+model_list[0]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[0]+'_'+dataset],
         color='#FCDC89', alpha=0.7, lw=lw,
         label='LR: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[0]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[0]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[1]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[1]+'_'+dataset],
         color='#E3EA96',alpha=0.7, lw=lw,
         label='RF: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[1]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[1]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[2]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[2]+'_'+dataset],
         color='#AAD09D',alpha=0.7, lw=lw,
         label='XGB: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[2]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[2]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[3]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[3]+'_'+dataset],
         color='#66BC98',alpha=0.7, lw=lw,
         label='SVC: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[3]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[3]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[4]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[4]+'_'+dataset],
         color='#C080C2',alpha=0.7, lw=lw,
         label='LDA: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[4]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[4]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[5]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[5]+'_'+dataset],
         color='#A079BD',alpha=0.7, lw=lw,
         label='GBM: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[5]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[5]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[6]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[6]+'_'+dataset],
         color='#73D2D7',alpha=0.7, lw=lw,
         label='MLP: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[6]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[6]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[7]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[7]+'_'+dataset],
         color='#B2D2E8',alpha=0.7, lw=lw,
         label='SAINT: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[7]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[7]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[8]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[8]+'_'+dataset],
         color='#71B1D7',alpha=0.7, lw=lw,
         label='TabNet: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[8]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[8]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[9]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[9]+'_'+dataset],
         color='#9491D7',alpha=0.7, lw=lw,
         label='TabPFN: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[9]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[9]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_'+model_list[10]+'_'+dataset], dict_auc_all_extdataset['tpr_'+model_list[10]+'_'+dataset],
         color='#7A6BB8',alpha=0.7,lw=lw,
         label='TransTab: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_'+model_list[10]+'_'+dataset]+
               dict_auc_all_extdataset['95%CI_'+model_list[10]+'_'+dataset])

plt.plot(dict_auc_all_extdataset['fpr_our_'+dataset], dict_auc_all_extdataset['tpr_our_'+dataset], color='#D24D3E',alpha=1,
         lw=lw, label='Ours: AUC = %0.3f ' % dict_auc_all_extdataset['roc_auc_our_'+dataset]+dict_auc_all_extdataset['95%CI_our_'+dataset])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=22, fontproperties=prop)
plt.yticks(fontsize=22, fontproperties=prop)
plt.xlabel('FPR',fontsize=24,fontproperties=prop)
plt.ylabel('TPR',fontsize=24, fontproperties=prop)
plt.title('ROC curves for all Chinese external validation dataset', fontsize=26, pad=20,
          fontweight='bold', fontproperties=prop)
plt.legend(loc="lower right",fontsize=22, prop=prop)
ax = plt.gca()
plt.savefig('./results/'+path+'/AUC/all_ext_dataset.eps')


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

result_cohort_int = pd.DataFrame(columns=['model','AUC','Accuracy', 'F1-score', 'Sensitivity',
                                      'Specificity',  'PPV', 'NPV'])
result_cohort_ext12 = pd.DataFrame(columns=['model','AUC','Accuracy', 'F1-score', 'Sensitivity',
                                      'Specificity',  'PPV', 'NPV'])
result_cohort_ext3 = pd.DataFrame(columns=['model','AUC','Accuracy', 'F1-score', 'Sensitivity',
                                      'Specificity',  'PPV', 'NPV'])
result_cohort_ext4 = pd.DataFrame(columns=['model','AUC','Accuracy', 'F1-score', 'Sensitivity',
                                      'Specificity',  'PPV', 'NPV'])








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
