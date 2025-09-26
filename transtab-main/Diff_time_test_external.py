import pandas as pd
import transtab
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
prop = fm.FontProperties(fname=font_path, size=22)
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
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
int_pre_cols_ext = [
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
int_preintra_cols_ext = [
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
         ]

"""
内部测试：df_alin_test
外部测试
"""
df_sum_ext1 = pd.concat([df_alin_ext12, df_alin_ext3, df_alin_ext4])
# df_sum_ext1 = pd.concat([df_alin_ext12, df_alin_ext3, df_alin_ext4])
df_sum_ext2 = pd.concat([df_alin_ext12, df_alin_ext3, df_alin_ext4,df_PHLF_ext6, df_PHLF_ext7])
df_sum1 = pd.concat([df_alin_test, df_sum_ext1])
df_sum2 = pd.concat([df_alin_test, df_sum_ext2])


int_intra_cols = list(set(int_preintra_cols)-set(int_pre_cols))

int_post_cols = list(set(list(df_sum1.columns))-set(int_preintra_cols))
int_post_cols_ext = list(set(list(df_sum1.columns))-set(int_preintra_cols+['First postoperative Total Bile Acids',
                              'First postoperative Gamma-glutamyl transferase']))

# int_pre_PHLF = ['PHLF']+ int_pre_cols
int_intra_PHLF = ['PHLF']+ int_intra_cols
int_post_PHLF = ['PHLF']+ int_post_cols
int_post_cols_ext_PHLF = ['PHLF'] + int_post_cols_ext
int_post_PHLF_ext = ['PHLF']+ int_post_cols_ext
int_intrapost_PHLF = ['PHLF'] + int_intra_cols + int_post_cols

int_pre_cols = [item for item in int_pre_cols if item != 'Indocyanine Green Retention at 15 Minutes']
int_preintra_cols = [item for item in int_preintra_cols if item != 'Indocyanine Green Retention at 15 Minutes']

df_sum_ext1_pre = df_sum2.loc[:,int_pre_cols]
df_sum_ext1_intra = df_sum2.loc[:,int_intra_PHLF]
df_sum_ext1_post = df_sum2.loc[:,int_post_PHLF]
df_sum_ext1_preintra = df_sum2.loc[:,int_preintra_cols]
df_sum_ext1_intrapost = df_sum2.loc[:,int_intrapost_PHLF]

df_sum_ext1_pre = df_sum_ext1_pre.reset_index(drop=True)
df_sum_ext1_intra= df_sum_ext1_intra.reset_index(drop=True)
df_sum_ext1_post = df_sum_ext1_post.reset_index(drop=True)
df_sum_ext1_preintra = df_sum_ext1_preintra.reset_index(drop=True)
df_sum_ext1_intrapost = df_sum_ext1_intrapost.reset_index(drop=True)

df_sum2 = df_sum2.reset_index(drop=True)

df_ext6_pre = df_PHLF_ext6.loc[:,int_pre_cols_ext]
df_ext6_intra  = df_PHLF_ext6.loc[:,int_intra_PHLF]
df_ext6_post = df_PHLF_ext6.loc[:,int_post_PHLF_ext]
df_ext6_preintra = df_PHLF_ext6.loc[:,int_preintra_cols_ext]
df_ext6_intrapost = df_PHLF_ext6.loc[:,int_post_cols_ext_PHLF]

df_ext7_pre = df_PHLF_ext7.loc[:,int_pre_cols_ext]
df_ext7_intra  = df_PHLF_ext7.loc[:,int_intra_PHLF]
df_ext7_post = df_PHLF_ext7.loc[:,int_post_PHLF_ext]
df_ext7_preintra = df_PHLF_ext7.loc[:,int_preintra_cols_ext]
df_ext7_intrapost = df_PHLF_ext7.loc[:,int_post_cols_ext_PHLF]

df_mimic_pre = df_PHLF_mimic.loc[:,int_pre_cols_ext]
df_mimic_intra  = df_PHLF_mimic.loc[:,int_intra_PHLF]
df_mimic_post = df_PHLF_mimic.loc[:,int_post_PHLF_ext]
df_mimic_preintra = df_PHLF_mimic.loc[:,int_preintra_cols_ext]
df_mimic_intrapost = df_PHLF_mimic.loc[:,int_post_cols_ext_PHLF]

print('=============================all=======================================')
print('\n', 'Only preoperative variables')
prob_pre = transtab.predict(model, df_sum_ext1_pre.iloc[:, 1:], df_sum_ext1_pre.iloc[:,0])
result_pre = get_final_result(threshold,prob_pre, df_sum_ext1_pre.iloc[:,0])
print('\n', 'Only intraoperative variables')
prob_intra = transtab.predict(model, df_sum_ext1_intra.iloc[:, 1:], df_sum_ext1_intra.iloc[:,0])
result_intra = get_final_result(threshold,prob_intra, df_sum_ext1_intra.iloc[:,0])
print('\n', 'Only postoperative variables')
prob_post = transtab.predict(model, df_sum_ext1_post.iloc[:, 1:], df_sum_ext1_post.iloc[:,0])
result_mpost = get_final_result(threshold,prob_post, df_sum_ext1_post.iloc[:,0])
print('\n', 'The preoperative & intraoperative variables')
prob_preintra = transtab.predict(model, df_sum_ext1_preintra.iloc[:, 1:], df_sum_ext1_preintra.iloc[:,0])
result_preintra = get_final_result(threshold,prob_preintra, df_sum_ext1_preintra.iloc[:,0])
print('\n', 'The intraoperative & postoperative variables')
prob_intrapost = transtab.predict(model, df_sum_ext1_intrapost.iloc[:, 1:], df_sum_ext1_intrapost.iloc[:,0])
result_intrapost = get_final_result(threshold,prob_intrapost, df_sum_ext1_intrapost.iloc[:,0])
print('\n', 'All variabels')
prob_all = transtab.predict(model, df_sum2.iloc[:, 1:], df_sum2.iloc[:,0].astype(int))
result_all = get_final_result(threshold,prob_all, df_sum2.iloc[:,0])
print('\n','====================================================================')


def AUC_CI_95_calculate(dic):
    fpr, tpr, thresholds = roc_curve(dic['real'], dic['pred'])
    roc_auc = auc(fpr, tpr)
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_score = []
    bootstrapped_tprs = []

    # Convert real and pred to numpy arrays
    dic['real'] = np.array(list(map(int, dic['real'])))
    dic['pred'] = np.array(dic['pred'])

    # Define a fixed grid for interpolation
    mean_fpr = np.linspace(0, 1, 100)  # Fixed grid for FPR

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(dic['pred']), len(dic['pred']))
        if len(np.unique(dic['real'][indices])) < 2:
            continue
        score = roc_auc_score(dic['real'][indices], dic['pred'][indices])
        fpr_bs, tpr_bs, _ = roc_curve(dic['real'][indices], dic['pred'][indices])
        bootstrapped_score.append(score)
        bootstrapped_tprs.append(np.interp(mean_fpr, fpr_bs, tpr_bs))  # Interpolate TPR on fixed grid

    sorted_scores = np.array(bootstrapped_score)
    bootstrapped_tprs = np.array(bootstrapped_tprs)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(len(sorted_scores) * 0.025)]
    confidence_upper = sorted_scores[int(len(sorted_scores) * 0.975)]
    tprs_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
    tprs_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)

    # Save interpolated FPR instead of original FPR
    dic['fpr'] = list(mean_fpr)  # Use the fixed grid for FPR
    dic['tpr'] = list(np.interp(mean_fpr, fpr, tpr))  # Interpolate original TPR on fixed grid
    dic['roc_auc'] = roc_auc
    dic['95%CI'] = ('( {}, {})'.format(round(confidence_lower, 3), round(confidence_upper, 3)))
    dic['tprs_lower'] = tprs_lower
    dic['tprs_upper'] = tprs_upper
    return dic

def create_dic(real, pred):
    dic = {}
    dic['real'] = real
    dic['pred'] = pred
    dic = AUC_CI_95_calculate(dic)
    return dic


def roc_auc_with_95CI(dic, variable_name, path_name):
    lw = 5
    plt.figure(figsize=[11, 10])
    plt.plot(dic['fpr'], dic['tpr'], color='#2b7bba', alpha=0.7, lw=lw,
             label='AUC = %0.3f ' % dic['roc_auc'] + dic['95%CI'])
    # plt.fill_between(dic['fpr'], dic['tprs_lower'], dic['tprs_upper'],
    #                  color='#E2EDF7')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=22, fontproperties=prop)
    plt.yticks(fontsize=22, fontproperties=prop)
    plt.xlabel('FPR', fontsize=24, fontproperties=prop)
    plt.ylabel('TPR', fontsize=24, fontproperties=prop)
    # plt.title('ROC curves for ' + variable_name, fontsize=26, pad=20,
    #           fontweight='bold', fontproperties=prop)
    # plt.legend(loc="lower right", fontsize=22, prop=prop)
    ax = plt.gca()
    plt.savefig('./results/' + path + '/AUC/' + path_name + '.jpg')

dic_all_pre = create_dic(df_sum_ext1_pre.iloc[:,0], prob_pre)
dic_all_preintra = create_dic(df_sum_ext1_preintra.iloc[:,0], prob_preintra)
dic_all = create_dic(df_sum2.iloc[:,0], prob_all)

roc_auc_with_95CI(dic_all_pre, 'Basic Info + Preop Factors', 'pre')
roc_auc_with_95CI(dic_all_preintra, 'Basic Info + Preop + Intra Factors', 'preintra')
roc_auc_with_95CI(dic_all, 'All factors', 'all_variabels')

print('===================================ext6=================================')
print('\n', 'Only preoperative variables')
prob_pre = transtab.predict(model, df_ext6_pre.iloc[:, 1:], df_ext6_pre.iloc[:,0])
result_pre = get_final_result(threshold,prob_pre, df_ext6_pre.iloc[:,0])
print('\n', 'Only intraoperative variables')
prob_intra = transtab.predict(model, df_ext6_intra.iloc[:, 1:], df_ext6_intra.iloc[:,0])
result_intra = get_final_result(threshold,prob_intra, df_ext6_intra.iloc[:,0])
print('\n', 'Only postoperative variables')
prob_post = transtab.predict(model, df_ext6_post.iloc[:, 1:], df_ext6_post.iloc[:,0])
result_post = get_final_result(threshold,prob_post, df_ext6_post.iloc[:,0])
print('\n', 'The preoperative & intraoperative variables')
prob_preintra = transtab.predict(model, df_ext6_preintra.iloc[:, 1:], df_ext6_preintra.iloc[:,0])
result_preintra = get_final_result(threshold,prob_preintra, df_ext6_preintra.iloc[:,0])
print('\n', 'The intraoperative & postoperative variables')
prob_intrapost = transtab.predict(model, df_ext6_intrapost.iloc[:, 1:], df_ext6_intrapost.iloc[:,0])
result_intrapost = get_final_result(threshold,prob_intrapost, df_ext6_intrapost.iloc[:,0])
print('\n','====================================================================')

print('===================================ext7=================================')
print('\n', 'Only preoperative variables')
prob_pre = transtab.predict(model, df_ext7_pre.iloc[:, 1:], df_ext7_pre.iloc[:,0])
result_pre = get_final_result(threshold,prob_pre, df_ext7_pre.iloc[:,0])
print('\n', 'Only intraoperative variables')
prob_intra = transtab.predict(model, df_ext7_intra.iloc[:, 1:], df_ext7_intra.iloc[:,0])
result_intra = get_final_result(threshold,prob_intra, df_ext7_intra.iloc[:,0])
print('\n', 'Only postoperative variables')
prob_post = transtab.predict(model, df_ext7_post.iloc[:, 1:], df_ext7_post.iloc[:,0])
result_post = get_final_result(threshold,prob_post, df_ext7_post.iloc[:,0])
print('\n', 'The preoperative & intraoperative variables')
prob_preintra = transtab.predict(model, df_ext7_preintra.iloc[:, 1:], df_ext7_preintra.iloc[:,0])
result_preintra = get_final_result(threshold,prob_preintra, df_ext7_preintra.iloc[:,0])
print('\n', 'The intraoperative & postoperative variables')
prob_intrapost = transtab.predict(model, df_ext7_intrapost.iloc[:, 1:], df_ext7_intrapost.iloc[:,0])
result_intrapost = get_final_result(threshold,prob_intrapost, df_ext7_intrapost.iloc[:,0])
print('\n','====================================================================')


print('==================================mimic==================================')
print('\n', 'Only preoperative variables')
prob_pre = transtab.predict(model, df_mimic_pre.iloc[:, 1:], df_mimic_pre.iloc[:,0])
result_pre = get_final_result(threshold,prob_pre, df_mimic_pre.iloc[:,0])
print('\n', 'Only intraoperative variables')
prob_intra = transtab.predict(model, df_mimic_intra.iloc[:, 1:], df_mimic_intra.iloc[:,0])
result_intra = get_final_result(threshold,prob_intra, df_mimic_intra.iloc[:,0])
print('\n', 'Only postoperative variables')
prob_post = transtab.predict(model, df_mimic_post.iloc[:, 1:], df_mimic_post.iloc[:,0])
result_post = get_final_result(threshold,prob_post, df_mimic_post.iloc[:,0])
print('\n', 'The preoperative & intraoperative variables')
prob_preintra = transtab.predict(model, df_mimic_preintra.iloc[:, 1:], df_mimic_preintra.iloc[:,0])
result_preintra = get_final_result(threshold,prob_preintra, df_mimic_preintra.iloc[:,0])
print('\n', 'The intraoperative & postoperative variables')
prob_intrapost = transtab.predict(model, df_mimic_intrapost.iloc[:, 1:], df_mimic_intrapost.iloc[:,0])
result_intrapost = get_final_result(threshold,prob_intrapost, df_mimic_intrapost.iloc[:,0])
print('\n','====================================================================')