import json
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN,BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import make_scorer,accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate

# from DL_experiment_result import specificity
from analysis_utils import dict_result_obtain
import joblib

from transtab import random_seed


def plot_confuse(y_preds, y_labels, name):
    preds_class = []
    for i in range(y_preds.shape[0]):
        preds_class.append(y_preds[i,].argmax())
    preds_class = np.array(preds_class)
    cm = confusion_matrix(y_labels, preds_class)
    # draw the figure
    # plt.figure()
    labels = ['Class 0', 'Class 1', 'Class 2']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(name + 'Confusion Matrix')
    plt.show()

def get_final_result(preds,label,):
    label_one_hot = np.zeros_like(preds)
    for i in range(preds.shape[0]):
        label_one_hot[i, int(label[i])] = 1
    # auc = roc_auc_score(y_score=preds[:, 1], y_true=label_one_hot, average='macro', multi_class='ovr')
    # auc = roc_auc_score(label_one_hot, preds, average='macro')
    auc = roc_auc_score(y_true=label, y_score=preds[:,1])
    y_preds = preds.argmax(1) #argmax取出preds元素最大值所对应的索引,1代表维度，是指在第二维里取最大值的索引
    acc = accuracy_score(y_true=label, y_pred=y_preds)
    recall = recall_score(y_true=label, y_pred=y_preds, )
    precision = precision_score(y_true=label, y_pred=y_preds, labels=[0])
    f1 = f1_score(y_true=label, y_pred=y_preds, )
    kappa = cohen_kappa_score(y1=label, y2=y_preds, )
    mcc = matthews_corrcoef(y_true=label, y_pred= y_preds)
    con = confusion_matrix(label, y_preds)
    TN = con[0, 0]
    FP = con[0, 1]
    specificity = TN/(TN+FP)
    print(con)
    print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc), 'recall:','{:.3f}'.format(recall),
          'specificity:','{:.3f}'.format(specificity))
    return y_preds

    # return ['Accuracy:', '{:.5f}'.format(precision), ]
def get_final_result2(preds,label,):
    auc = roc_auc_score(y_score=preds, y_true=label)
    # y_preds = preds.argmax(1) #argmax取出preds元素最大值所对应的索引,1代表维度，是指在第二维里取最大值的索引
    acc = accuracy_score(y_true=label, y_pred=preds)
    # recall = recall_score(y_true=label, y_pred=preds,pos_label=0)
    f1 = f1_score(y_true=label, y_pred=preds)
    # con = confusion_matrix(label, preds)
    # print(con)
    return [acc, f1, auc]
def xgb (train_data,test_data,train_label,test_label,name,result,plot):
    clf_xgb = XGBClassifier(max_depth=8,
                            learning_rate=0.001,
                            n_estimators=150, )
    # clf_xgb = XGBClassifier()
    clf_xgb.fit(train_data.iloc[:, :-1], train_label,)
    preds = np.array(clf_xgb.predict_proba(test_data.iloc[:, :-1]))
    res_test = get_final_result(preds, test_label, )
    result.append(res_test)
    print(name,res_test)
    return result

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fp)

def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp/(tp+fn)

def model_train (train_data,test_data,ext_test12, ext_test_3, ext_test_4, model, name, opt_plot):
    if model == 'rf':
        clf = RandomForestClassifier(n_estimators=60,
                                        criterion="log_loss",
                                        max_depth=8,)
    elif model == 'xgb':
        clf = XGBClassifier(max_depth=4,
                            learning_rate=0.5,
                            n_estimators=300,
                            subsample = 0.7,
                            gamma=0.7,
                            )
    elif model == 'svc':
        clf = SVC(kernel="linear",
                  probability=True,
                  C=3,
                  degree=3,
                  gamma= 'auto',
                  shrinking=False,
                  class_weight='balanced'
        )
    elif model == 'mlp':
        clf = MLPClassifier()
    elif model == 'lr':
        # solver = 'liblinear'
        # class_weight = 'balanced'
         clf = LogisticRegression(penalty='l1',solver = 'liblinear',  C=5, class_weight='balanced',
                                 intercept_scaling=0.5)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score),
        'roc_auc_score': make_scorer(roc_auc_score),
        'kappa_score': make_scorer(matthews_corrcoef),
        'mcc': make_scorer(matthews_corrcoef),
        'specificity_score': make_scorer(specificity_score),
        'sensitivity_score': make_scorer(sensitivity_score),
    }
    # results = cross_validate(clf,)


    clf.fit(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # print(name, pd.DataFrame(train_label).value_counts())
    joblib.dump(clf,'./checkpoint/'+name+'2.pkl')
    preds_int = np.array(clf.predict_proba(test_data.iloc[:, 1:]))
    preds_test = get_final_result(preds_int, test_data.iloc[:, 0], )

    if opt_plot:
        plot_confuse(preds_int, test_data.iloc[:, 0], name)

    preds_ext12 = np.array(clf.predict_proba(ext_test12.iloc[:, 1:]))
    y_pred_test = get_final_result(preds_ext12, ext_test12.iloc[:, 0], )

    preds_ext3 = np.array(clf.predict_proba(ext_test_3.iloc[:, 1:]))
    res_test = get_final_result(preds_ext3, ext_test_3.iloc[:, 0], )

    preds_ext4 = np.array(clf.predict_proba(ext_test_4.iloc[:, 1:]))
    res_test = get_final_result(preds_ext4, ext_test_4.iloc[:, 0], )

    return preds_int[:, 1], preds_ext12[:,1], preds_ext3[:,1], preds_ext4[:,1]

def model_load (test_data,ext_test12, ext_test_3, ext_test_4, model, name, opt_plot):
    clf = joblib.load('./checkpoint/'+name+'.pkl')

    preds_int = np.array(clf.predict_proba(test_data.iloc[:, 1:]))
    preds_test = get_final_result(preds_int, test_data.iloc[:, 0], )

    if opt_plot:
        plot_confuse(preds_int, test_data.iloc[:, 0], name)

    preds_ext12 = np.array(clf.predict_proba(ext_test12.iloc[:, 1:]))
    y_pred_test = get_final_result(preds_ext12, ext_test12.iloc[:, 0], )

    preds_ext3 = np.array(clf.predict_proba(ext_test_3.iloc[:, 1:]))
    res_test = get_final_result(preds_ext3, ext_test_3.iloc[:, 0], )

    preds_ext4 = np.array(clf.predict_proba(ext_test_4.iloc[:, 1:]))
    res_test = get_final_result(preds_ext4, ext_test_4.iloc[:, 0], )

    return preds_int[:, 1], preds_ext12[:,1], preds_ext3[:,1], preds_ext4[:,1]

def average_result(perform_list,name):
    acc = []
    # recall = []
    f1 = []
    auc = []
    kappa = []
    mcc = []
    for j in range(5):
        acc.append(float(perform_list[j][1]))
        auc.append(float(perform_list[j][5]))
        f1.append(float(perform_list[j][3]))
        kappa.append(float(perform_list[j][7]))
        mcc.append(float(perform_list[j][9]))
        # auc.append(perform_list[j][3])
    acc = np.array(acc)
    # recall = np.array(recall)
    f1 = np.array(f1)
    auc = np.array(auc)
    kappa = np.array(kappa)
    mcc = np.array(mcc)
    # print(name,': acc:','{:.3f}'.format(acc.mean()),'{:.3f}'.format(acc.std()),'f1:','{:.3f}'.format(f1.mean()),
    #             '{:.3f}'.format(f1.std()),'auc:','{:.3f}'.format(auc.mean()),'{:.3f}'.format(auc.std()))

    #
    # print(name,'acc:','{:.3f}'.format(acc.mean()),'f1:','{:.3f}'.format(f1.mean()),
    #             'auc:','{:.3f}'.format(auc.mean()), 'kappa:','{:.3f}'.format(kappa.mean()),
    #             'MCC:','{:.3f}'.format(mcc.mean()), )

    # return str([name,'acc:','{:.3f}'.format(acc.mean()),'f1:','{:.3f}'.format(f1.mean()),
    #             'auc:','{:.3f}'.format(auc.mean())])
    # return str([name,'{:.3f}'.format(acc.mean()),'{:.3f}'.format(acc.std()),'{:.3f}'.format(f1.mean()),
    #             '{:.3f}'.format(f1.std()),'{:.3f}'.format(auc.mean()),'{:.3f}'.format(auc.std()),])
    return [name,'{:.3f}'.format(acc.mean()),'{:.3f}'.format(acc.std()),'{:.3f}'.format(f1.mean()),
                '{:.3f}'.format(f1.std()),'{:.3f}'.format(auc.mean()),'{:.3f}'.format(auc.std()),
                '{:.3f}'.format(kappa.mean()),'{:.3f}'.format(kappa.std()),
                '{:.3f}'.format(mcc.mean()),'{:.3f}'.format(mcc.std()),]

def average_result_print(perform_list,name):
    acc = []
    recall = []
    f1 = []
    auc = []
    for j in range(5):
        acc.append(perform_list[j][0])
        auc.append(perform_list[j][2])
        f1.append(perform_list[j][1])
        # auc.append(perform_list[j][3])
    acc = np.array(acc)
    recall = np.array(recall)
    f1 = np.array(f1)
    auc = np.array(auc)
    print(name,'\t','\t','\t',': acc:','{:.3f}'.format(acc.mean()),'f1:','{:.3f}'.format(f1.mean()),
                'auc:','{:.3f}'.format(auc.mean()),)
    return str([name,': acc:','{:.3f}'.format(acc.mean()),'{:.3f}'.format(acc.std()),'f1:','{:.3f}'.format(f1.mean()),
                '{:.3f}'.format(f1.std()),'auc:','{:.3f}'.format(auc.mean()),'{:.3f}'.format(auc.std()),])

major_minor = 2

opt_compare = False
opt_file = True
file_value = 0.58
# opt_compare = False
sample_size = 500

df_alin_train = pd.read_csv('../data_process/df_int_alin_train.csv')
df_alin_test = pd.read_csv('../data_process/df_int_alin_test.csv')
df_alin_ext12 = pd.read_csv('../data_process/df_ext12_alin.csv')
df_alin_ext3 = pd.read_csv('../data_process/New_0702_ext_3_alin.csv')
df_alin_ext4 = pd.read_csv('../data_process/df_ext4_alin.csv')
col_idx = json.load(open('../data_process/col_idx.json','r'))

df_alin_ext3 = df_alin_ext3.rename(columns={'Transfusion_1':'Transfusion'})
# df_alin_ext3 = df_alin_ext3.rename(columns=col_idx)
df_alin_ext3 = df_alin_ext3[df_alin_train.columns]


info = json.load(open('../data_process/info_0603.json'))


df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]
df_alin_ext12 = df_alin_ext12.iloc[:,1:]
df_alin_ext3 = df_alin_ext3.iloc[:,1:]
df_alin_ext4 = df_alin_ext4.iloc[:,1:]

df_alin_train = df_alin_train.fillna(0)
df_alin_test = df_alin_test.fillna(0)
df_alin_ext12 = df_alin_ext12.fillna(0)
df_alin_ext3 = df_alin_ext3.fillna(0)
df_alin_ext4 = df_alin_ext4.fillna(0)


model = 'rf'

preds_int, preds_ext12, preds_ext3, preds_ext4 = model_train(df_alin_train,df_alin_test,df_alin_ext12,df_alin_ext3, df_alin_ext4,model,model,False)
# preds_int, preds_ext12, preds_ext3, preds_ext4 = model_load(df_alin_test,df_alin_ext12,df_alin_ext3, df_alin_ext4,model,model,False)
print('============================================================================')
# model = 'svc3'
dict_result_obtain(preds_int, np.array(df_alin_test.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model+'_int_result.json')
dict_result_obtain(preds_ext12, np.array(df_alin_ext12.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model+'_ext12_result.json')
dict_result_obtain(preds_ext3, np.array(df_alin_ext3.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model+'_ext3_result.json')
dict_result_obtain(preds_ext4, np.array(df_alin_ext4.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+model+'_ext4_result.json')

# model = rf
# lc_result = model(lc_syn, test, lc_label, test.iloc[:, -1], 'lcGAN', lcgan_perform, False)

