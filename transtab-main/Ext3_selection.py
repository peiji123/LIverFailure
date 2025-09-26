import pandas as pd
import numpy as np
import transtab
import os
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

def get_final_result(preds,label,):
    y_preds = np.zeros_like(preds)
    for idx, val in enumerate(preds):
        if val > 0.5:
            y_preds[idx] = 1

    # label_one_hot = np.zeros_like(preds)
    # for i in range(preds.shape[0]):
    #     label_one_hot[i, int(label[i])] = 1
    # auc = roc_auc_score(y_score=preds[:, 1], y_true=label_one_hot, average='macro', multi_class='ovr')
    # auc = roc_auc_score(label_one_hot, preds, average='macro')
    auc = roc_auc_score(y_true=label, y_score=preds, multi_class='ovr')
    # y_preds = preds.argmax(1) #argmax取出preds元素最大值所对应的索引,1代表维度，是指在第二维里取最大值的索引
    acc = accuracy_score(y_true=label, y_pred=y_preds)
    recall = recall_score(y_true=label, y_pred=y_preds, average='macro')
    precision = precision_score(y_true=label, y_pred=y_preds, average='macro', labels=[0])
    f1 = f1_score(y_true=label, y_pred=y_preds, average='macro')
    kappa = cohen_kappa_score(y1=label, y2=y_preds, )
    mcc = matthews_corrcoef(y_true=label, y_pred= y_preds)
    con = confusion_matrix(label, y_preds)
    labels = ['Normal', 'Liver Failure']
    plt.figure(figsize=(13, 10))
    plt.rcParams['font.size'] = 16
    sns.heatmap(con, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,cmap="Blues")
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Truth', fontsize=18)
    plt.show()

    print(con)
    print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc), 'recall:','{:.3f}'.format(recall))
    return y_preds

def dict_result_obtain(model, data, label, name, ori_data):
    y_pred = transtab.predict(model, data, label)
    result = get_final_result(y_pred, label)

    label_list = [float(x) for x in list(label)]
    y_pred_list = [float(x) for x in list(y_pred)]
    result_list = [float(x) for x in list(result)]

    dict_result = {
        'real': label_list,
        'pred_pro': y_pred_list,
        'pred': result_list,
    }
    # os.makedirs(os.path.dirname('./Real_Prediction/dict_our_'+name+'_result.json'), exist_ok=True)
    with open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_'+name+'_result.json', 'w') as f:
        json.dump(dict_result, f)

    error_data = data[result != label]
    ori_erro_data = ori_data.iloc[error_data.index,:]
    # filtered_df = ori_data[result != label]
    print(ori_erro_data['PHLFG'].value_counts())

    return dict_result

info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

df_alin_train = pd.read_csv('../data_process/df_int_train.csv')
df_alin_test = pd.read_csv('../data_process/df_int_test.csv')
df_alin_ext12 = pd.read_csv('../data_process/df_ext12.csv')
df_alin_ext3 = pd.read_csv('../data_process/df_ext3.csv')

df_inter =  pd.read_excel('../data_process/PHLF本中心-601.xlsx')
original_ext12 = pd.read_excel('../data_process/外部验证中心12-601.xlsx')
original_ext3 = pd.read_excel('../data_process/外部验证中心3-601.xlsx')

original_inter_test = df_inter.loc[df_alin_test['Unnamed: 0'],:]

df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]
df_alin_ext12 = df_alin_ext12.iloc[:,1:]
df_alin_ext3 = df_alin_ext3.iloc[:,1:]

# df_alin_ext3 = df_alin_ext3.dropna()

model_path = '/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/LF_bio_240614_best_4'
model = transtab.build_classifier(checkpoint=model_path,device='cuda:0')
model.eval()

dict_int_result = dict_result_obtain(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0], 'int', original_inter_test)
dict_ext12_result = dict_result_obtain(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0], 'ext1', original_ext12)
dict_ext12_result = dict_result_obtain(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0], 'ext3', original_ext3)





# y_pred = transtab.predict(model, df_alin_ext12.iloc[:, 1:], df_alin_ext12.iloc[:, 0])
# result_ext3 = get_final_result(y_pred, df_alin_ext12.iloc[:, 0])
#
# y_pred = transtab.predict(model, df_alin_ext3.iloc[:, 1:], df_alin_ext3.iloc[:, 0])
# result_ext3 = get_final_result(y_pred, df_alin_ext3.iloc[:, 0])

# dict_ext3_result = {
#     'real': list(df_alin_ext3.iloc[:, 0]),
#     'pred_pro': list(y_pred),
#     'pred': list(result_ext3),
# }
# with open('./Real_Prediction/dict_our_ext3_result.json', 'w') as f:
#     json.dump(dict_ext3_result, f)

# filtered_df = original_ext3[result_ext3 != df_alin_ext3.iloc[:, 0]]
# filtered_df['PHLFG'].value_counts()