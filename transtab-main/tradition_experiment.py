import json
import matplotlib.pyplot as plt
import pandas as pd
# from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN,BorderlineSMOTE
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import train_test_split

from Banchmark.tabnet_test import df_alin_train
# from DL_experiment_result_external import df_alin_mimic, df_alin_ext6, df_alin_ext7
# from DL_experiment_result import specificity
from transtab.analysis_utils import dict_result_obtain
import joblib

# def get_final_result(preds,label,):
#     # auc = roc_auc_score(y_score=preds[:, 1], y_true=label)
#     y_preds = preds.argmax(1)
#     acc = accuracy_score(y_true=label, y_pred=y_preds)
#     # f1 = f1_score(y_true=label, y_pred=y_preds,average='macro')
#     f1 = f1_score(y_true=label, y_pred=y_preds)
#     label_one_hot = np.zeros_like(preds)
#     for i in range(preds.shape[0]):
#         label_one_hot[i,int(label[i])] = 1
#     # auc = roc_auc_score(label_one_hot, preds,  average='micro')
#     auc = roc_auc_score(label_one_hot, preds,)
#     return [acc,f1, auc]

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
    FN = con[1, 0]
    TP = con[1, 1]
    if (TP + FP) > 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 0  # 或者可以设置为 NaN，取决于你的需求

    if (TN + FN) > 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 0
    specificity = TN/(TN+FP)
    print(con)
    print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc), 'recall:','{:.3f}'.format(recall),
          'specificity:','{:.3f}'.format(specificity), 'PPV:','{:.3f}'.format(PPV), 'NPV:','{:.3f}'.format(NPV))
    print('{:.3f}'.format(auc), '{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall),
          '{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV),'\n')
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

def rf (train_data,test_data,ext_test12, ext_test_3, ext_test_4, model, name, opt_plot):
    if model == 'rf':
        clf = RandomForestClassifier(
                                        n_estimators=60,
                                        # n_estimators=50,
                                        criterion="log_loss",
                                        # max_depth=8,
                                        max_depth=7,
                                        # 19, 36, 54, 73
                                        random_state=76
        )
    elif model == 'xgb':
        clf = XGBClassifier(
                            max_depth=5,
                            # learning_rate=0.5,
                            learning_rate=0.5,
                            # n_estimators=300,
                            n_estimators=100,
                            # subsample = 0.9,
                            subsample=0.1,
                            # gamma=0.9,
                            gamma=0.4,
                            )
    elif model == 'svc':
        clf = SVC(kernel="linear",
                  probability=True,
                  # C=3,
                  C=25,
                  degree=3,
                  # degree=4,
                  gamma= 'auto',
                  shrinking=False,
                  class_weight='balanced'
        )
    elif model == 'mlp':
        clf = MLPClassifier(solver='adam', learning_rate_init=1e-3,hidden_layer_sizes=(128,64))
    elif model == 'lr':
        # solver = 'liblinear'
        # class_weight = 'balanced'
        # clf = LogisticRegression(penalty='l1',solver = 'liblinear',  C=5, class_weight='balanced',
        #                          intercept_scaling=0.5)
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=20, class_weight='balanced',
                                 intercept_scaling=0.5)
    elif model == 'lda':
        positive_prior = 10 / 11
        negative_prior = 1 / 11
        priors = [negative_prior, positive_prior]
        clf = LDA(
            n_components=1,
            store_covariance = True,
        )
    elif model == 'lgb':
        clf = lgb.LGBMClassifier(
            # objective='binary',
            # boosting_type='gbdt',
            learning_rate=0.005,
            feature_fraction=0.9,
            # 9: 845, 10:839,
            # 12, 28:904, 41:901
            random_state=41,
            class_weight = 'balanced',
            scale_pos_weight = 6

        )

    clf.fit(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # print(name, pd.DataFrame(train_label).value_counts())
    joblib.dump(clf,'./checkpoint/'+name+'.pkl')
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

def model_train (train_data,test_data,ext_test12, ext_test_3, ext_test_4, ext_test_mimic, ext_test_6, ext_test_7,model, name, opt_plot):
    if model == 'rf':
        clf = RandomForestClassifier(
                                        n_estimators=60,
                                        # n_estimators=50,
                                        # criterion="log_loss",
                                        # max_depth=8,
                                        max_depth=5,
                                        # 9,
                                        random_state=9
        )
    elif model == 'xgb':
        clf = XGBClassifier(
                            max_depth=5,
                            # learning_rate=0.5,
                            learning_rate=0.5,
                            # n_estimators=300,
                            n_estimators=100,
                            # subsample = 0.9,
                            subsample=0.2,
                            # gamma=0.9,
                            gamma=0.4,
                            )
    elif model == 'svc':
        clf = SVC(kernel="linear",
                  probability=True,
                  # C=3,
                  C=25,
                  degree=3,
                  # degree=4,
                  gamma= 'auto',
                  shrinking=False,
                  class_weight='balanced'
        )
    elif model == 'mlp':
        clf = MLPClassifier(solver='adam', learning_rate_init=1e-3,hidden_layer_sizes=(128,64))
    elif model == 'lr':
        # solver = 'liblinear'
        # class_weight = 'balanced'
        # clf = LogisticRegression(penalty='l1',solver = 'liblinear',  C=5, class_weight='balanced',
        #                          intercept_scaling=0.5)
        clf = LogisticRegression(penalty='l2',
                                 tol=1e-5,
                                 # 'newton-cg', 'liblinear', 'saga', 'sag', 'lbfgs', 'newton-cholesky'
                                 random_state=3,
                                 solver='sag',
                                 l1_ratio=0.5,
                                 max_iter=50,
                                 C=100, class_weight='balanced',
                                 intercept_scaling=1)
    elif model == 'lda':
        positive_prior = 10 / 11
        negative_prior = 1 / 11
        priors = [negative_prior, positive_prior]
        clf = LDA(
            n_components=1,
            store_covariance = True,
        )
    elif model == 'lgb':
        clf = lgb.LGBMClassifier(
            # objective='binary',
            # boosting_type='gbdt',
            # learning_rate=0.005,
            learning_rate=0.0001,
            # feature_fraction=0.8,
            feature_fraction=0.9,
            # 9: 845, 10:839,
            # 12, 28:904, 41:901
            # 41
            # 15,
            # 38
            # 6-893, 8-900,11-906, 16-894, 20-904, 33-903, 62-910, 75-900, 76-904, 79-901,
            # 4-844, 5-843, 6-839, 16-838, 44, 45, 87, 124, 139-839, 50, 172-836, 62,83-838, 108, 146-837,
            # 125-834,
            random_state=125,
            class_weight = 'balanced',
            # scale_pos_weight = 9,
            scale_pos_weight = 0.9

        )

    clf.fit(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # print(name, pd.DataFrame(train_label).value_counts())
    joblib.dump(clf,'./checkpoint/'+name+'.pkl')
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

    preds_mimic = np.array(clf.predict_proba(ext_test_mimic.iloc[:, 1:]))
    res_test = get_final_result(preds_mimic, ext_test_mimic.iloc[:, 0], )

    preds_ext6 = np.array(clf.predict_proba(ext_test_6.iloc[:, 1:]))
    res_test = get_final_result(preds_ext6, ext_test_6.iloc[:, 0], )

    preds_ext7 = np.array(clf.predict_proba(ext_test_7.iloc[:, 1:]))
    res_test = get_final_result(preds_ext7, ext_test_7.iloc[:, 0], )

    return preds_int[:, 1], preds_ext12[:,1], preds_ext3[:,1], preds_ext4[:,1], preds_mimic[:,1], preds_ext6[:,1], preds_ext7[:,1]


def model_load (test_data,ext_test12, ext_test_3, ext_test_4,name, opt_plot):
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

from joblib import dump, load
import pickle


def model_load_traintest_ext(train_data,test_data, ext_test12, ext_test_3, ext_test_4, ext_mimic, ext_6, ext_7, name, opt_plot):
    clf = joblib.load('./checkpoint/' + name + '.pkl')

    print('--train--')
    preds_train = np.array(clf.predict_proba(train_data.iloc[:, 1:]))
    train_result = get_final_result(preds_train, train_data.iloc[:, 0], )

    print('--test--')
    preds_int = np.array(clf.predict_proba(test_data.iloc[:, 1:]))
    preds_test = get_final_result(preds_int, test_data.iloc[:, 0], )

    if opt_plot:
        plot_confuse(preds_int, test_data.iloc[:, 0], name)

    print('--ext12--')
    preds_ext12 = np.array(clf.predict_proba(ext_test12.iloc[:, 1:]))
    y_pred_test = get_final_result(preds_ext12, ext_test12.iloc[:, 0], )
    print('--ext3--')
    preds_ext3 = np.array(clf.predict_proba(ext_test_3.iloc[:, 1:]))
    res_test = get_final_result(preds_ext3, ext_test_3.iloc[:, 0], )
    print('--ext4--')
    preds_ext4 = np.array(clf.predict_proba(ext_test_4.iloc[:, 1:]))
    res_test = get_final_result(preds_ext4, ext_test_4.iloc[:, 0], )
    print('--mimic--')
    preds_ext_mimic = np.array(clf.predict_proba(ext_mimic.iloc[:, 1:]))
    res_test = get_final_result(preds_ext_mimic, ext_mimic.iloc[:, 0], )
    print('--ext6--')
    preds_ext6 = np.array(clf.predict_proba(ext_6.iloc[:, 1:]))
    res_test = get_final_result(preds_ext6, ext_6.iloc[:, 0], )
    print('--ext7--')
    preds_ext7 = np.array(clf.predict_proba(ext_7.iloc[:, 1:]))
    res_test = get_final_result(preds_ext7, ext_7.iloc[:, 0], )

    return (preds_train[:, 1], preds_int[:, 1], preds_ext12[:, 1], preds_ext3[:, 1],
            preds_ext4[:, 1], preds_ext_mimic[:, 1], preds_ext6[:,1], preds_ext7[:, 1])

def model_load_ext (test_data,ext_test12, ext_test_3, ext_test_4,ext_mimic, ext_6, ext_7,name, opt_plot):
    clf = joblib.load('./checkpoint/' + name + '.pkl')

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

    preds_ext_mimic = np.array(clf.predict_proba(ext_mimic.iloc[:, 1:]))
    res_test = get_final_result(preds_ext_mimic, ext_mimic.iloc[:, 0], )

    preds_ext6 = np.array(clf.predict_proba(ext_6.iloc[:, 1:]))
    res_test = get_final_result(preds_ext6, ext_6.iloc[:, 0], )

    preds_ext7 = np.array(clf.predict_proba(ext_7.iloc[:, 1:]))
    res_test = get_final_result(preds_ext7, ext_7.iloc[:, 0], )

    return preds_int[:, 1], preds_ext12[:,1], preds_ext3[:,1], preds_ext4[:,1], preds_ext_mimic[:,1], preds_ext6[:,1], preds_ext7[:,1]

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
df_alin_ext3 = pd.read_csv('../data_process/df_New_ext_alin_3_240625_best_018.csv')
df_alin_ext4 = pd.read_csv('../data_process/df_ext4_alin.csv')
# df_alin_mimic = pd.read_csv('../data_process/df_alin_mimic_few_null.csv')
df_alin_mimic = pd.read_csv('../data_process/df_alin_mimic.csv')
df_alin_ext6 = pd.read_csv('../data_process/df_alin_ext6.csv')
df_alin_ext7 = pd.read_csv('../data_process/df_alin_ext7.csv')

df_alin_mimic.columns = df_alin_train.columns
df_alin_ext6.columns = df_alin_train.columns
df_alin_ext7.columns = df_alin_train.columns

col_idx = json.load(open('../data_process/col_idx.json','r'))


df_alin_ext3 = df_alin_ext3.rename(columns={'Transfusion_1':'Transfusion'})
# df_alin_ext3 = df_alin_ext3.rename(columns=col_idx)
df_alin_ext3 = df_alin_ext3[df_alin_train.columns]

# df_alin_ext3.to_csv('../data_process/New_0702_ext_3_alin.csv',index=False)

info = json.load(open('../data_process/info_0603.json'))


df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]
df_alin_ext12 = df_alin_ext12.iloc[:,1:]
df_alin_ext3 = df_alin_ext3.iloc[:,1:]
df_alin_ext4 = df_alin_ext4.iloc[:,1:]
df_alin_mimic = df_alin_mimic.iloc[:,1:]
df_alin_ext6 = df_alin_ext6.iloc[:,1:]
df_alin_ext7 = df_alin_ext7.iloc[:,1:]

df_alin_train = df_alin_train.fillna(0)
df_alin_test = df_alin_test.fillna(0)
df_alin_ext12 = df_alin_ext12.fillna(0)
df_alin_ext3 = df_alin_ext3.fillna(0)
df_alin_ext4 = df_alin_ext4.fillna(0)
df_alin_mimic = df_alin_mimic.fillna(0)
df_alin_ext6 = df_alin_ext6.fillna(0)
df_alin_ext7 = df_alin_ext7.fillna(0)

# rf6, xgb2, mlp2
model = 'lgb'
# name = 'rf14'
#
# lr4, rf19, xgb4, svc1, lda, lgb3, mlp5, saint3, tabnet2, tabpfn4
name = 'mlp5'
# preds_int, preds_ext12, preds_ext3, preds_ext4 = rf(df_alin_train,df_alin_test,df_alin_ext12,df_alin_ext3, df_alin_ext4,model,name,False)
# preds_int, preds_ext12, preds_ext3, preds_ext4 = model_load(df_alin_test,df_alin_ext12,df_alin_ext3, df_alin_ext4,name,False)

# preds_int, preds_ext12, preds_ext3, preds_ext4, preds_mimic, preds_ext6, preds_ext7= (
#     model_train(df_alin_train, df_alin_test,df_alin_ext12,df_alin_ext3, df_alin_ext4,df_alin_mimic,df_alin_ext6, df_alin_ext7,model, name,False))
# preds_int, preds_ext12, preds_ext3, preds_ext4, preds_mimic, preds_ext6, preds_ext7= (
#     model_load_ext(df_alin_test,df_alin_ext12,df_alin_ext3, df_alin_ext4,df_alin_mimic,df_alin_ext6, df_alin_ext7,name,False))

preds_train ,preds_int, preds_ext12, preds_ext3, preds_ext4, preds_mimic, preds_ext6, preds_ext7= (
    model_load_traintest_ext(df_alin_train,df_alin_test,df_alin_ext12,df_alin_ext3, df_alin_ext4,
                             df_alin_mimic,df_alin_ext6, df_alin_ext7,name,False))

print('============================================================================')
dict_result_obtain(0.5, preds_train, np.array(df_alin_train.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+name+'_train_result.json')
'''
print('============================================================================')
print("===========================internal======================================")
dict_result_obtain(0.5, preds_int, np.array(df_alin_test.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+name+'_int_result.json')
print("===========================external 12 ======================================")
dict_result_obtain(0.5, preds_ext12, np.array(df_alin_ext12.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+name+'_ext12_result.json')
print("===========================external 3 ======================================")
dict_result_obtain(0.5, preds_ext3, np.array(df_alin_ext3.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+name+'_ext3_result.json')
print("===========================external 4 ======================================")
dict_result_obtain(0.5, preds_ext4, np.array(df_alin_ext4.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+name+'_ext4_result.json')
print("=========================== mimic ======================================")
dict_result_obtain(0.5, preds_mimic, np.array(df_alin_mimic.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+name+'_mimic_result.json')
print("===========================external 6 ======================================")
dict_result_obtain(0.5, preds_ext6, np.array(df_alin_ext6.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+name+'_ext6_result.json')
print("===========================external 7 ======================================")
dict_result_obtain(0.5, preds_ext7, np.array(df_alin_ext7.iloc[:,0]),
                   '/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_'+name+'_ext7_result.json')
'''
# model = rf
# lc_result = model(lc_syn, test, lc_label, test.iloc[:, -1], 'lcGAN', lcgan_perform, False)

