'''
没有纳入最后文章
分肝衰等级的混淆矩阵绘制
'''
import pandas as pd
import transtab
import torch
import json
from sklearn.model_selection import train_test_split
from child_class import GateChildHead
from Result_process.metrics_process import metrics_with_youden, metrics_multiclass
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
prop = fm.FontProperties(fname=font_path, size=22)

df_PHLF_train = pd.read_csv('../data_process/df_int_train.csv',index_col=0)
df_PHLF_test = pd.read_csv('../data_process/df_int_test.csv',index_col=0)
df_PHLF_valid_grade = pd.read_csv('../data_process/df_PHLF_valid_grade.csv')
df_PHLF_extvalid_grade = pd.read_csv('../data_process/df_PHLF_extvalid_grade.csv')
df_PHLF_train_with_grade = pd.read_csv('../data_process/df_PHLF_train_with_grade.csv')

df_PHLF_test_with_grade= pd.read_csv('../data_process/df_PHLF_test_with_grade.csv')
df_PHLF_ext12_with_grade= pd.read_csv('../data_process/df_PHLF_ext12_with_grade.csv')
df_PHLF_ext3_with_grade= pd.read_csv('../data_process/df_PHLF_ext3_with_grade.csv')
df_PHLF_ext4_with_grade= pd.read_csv('../data_process/df_PHLF_ext4_with_grade.csv')
df_PHLF_ext6_with_grade= pd.read_csv('../data_process/df_PHLF_ext6_with_grade.csv')
df_PHLF_ext7_with_grade= pd.read_csv('../data_process/df_PHLF_ext7_with_grade.csv')

ori_data_ext3 = pd.read_csv('../data_process/ori_240625_best_018_new_ext_3.csv',index_col=0)
ori_data_ext12 = pd.read_excel('../data_process/外部验证中心12-702.xlsx')
ori_data_int = pd.read_excel('../data_process/PHLF本中心-601.xlsx')
ori_data_ext4 = pd.read_excel('../data_process/外部验证中心4.xlsx')
ori_df_mimic = pd.read_excel('../data_process/mimic.xlsx',index_col=0)
ori_df_ext6 = pd.read_excel('../data_process/EX6.xlsx')
ori_df_ext7 = pd.read_excel('../data_process/EX7.xlsx')
ori_int_test = ori_data_int.loc[df_PHLF_test.index, :]
ori_int_train = ori_data_int.loc[df_PHLF_train.index, :]
ori_PHLF_valid_grade = pd.concat([ori_int_test, ori_data_ext12, ori_data_ext3, ori_data_ext4,
                                    ori_df_ext6, ori_df_ext7], axis=0, ignore_index=True)
# train_data, valid_data = train_test_split(df_PHLF_train_with_grade, test_size=0.2, random_state=42)

path = 'LF_bio_240625_best_018'
threshold=0.6
info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']
model = transtab.build_classifier_multi_task_onlyclss(cate_cols, num_cols, bin_cols)
model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+path)

weights = [1.0, 6.0, 16.0]
clf = GateChildHead(
    parent_classes=2,
    weights=weights,
    child_classes=3)
clf.load_state_dict(torch.load('./checkpoint/LF_bio_240625_best_018_grade_420.pth'))


def Grade_class_result(model, clf, data, ori_data, path):
    prob_parent, encoder_output = transtab.evaluator.predict_all_prob(model, data.iloc[:, 2:],
                                                                       data.iloc[:, 1])
    logits_child, loss_child = clf(encoder_output, data.iloc[:, 0], prob_parent, threshold)
    y_pred_prob = logits_child.detach().cpu().numpy()
    y_pred_prob_pd = pd.DataFrame(y_pred_prob, columns=['non_PHLF', 'Grade A', 'Grade B/C'],index=ori_data.index)
    y_parent_pd = pd.DataFrame(prob_parent,columns=['PHLF_pred'])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = data['PHLFG']
    con = confusion_matrix(y_true, y_pred)
    print(con)
    print(sum(con[1,:]))
    y_pred_pd = pd.DataFrame(y_pred, columns=['PHLFG_pred'],index=ori_data.index)
    y_pred_pd = y_pred_pd.replace(0, np.nan)
    y_pred_pd = y_pred_pd.replace(1, 'A')
    y_pred_pd = y_pred_pd.replace(2, 'B')
    result_int_child, child_con = metrics_multiclass(data.iloc[:, 0], y_pred_prob)
    Grade_aount = sum(data.iloc[:, 0] == 2 )
    print('There are '+ str(Grade_aount)+ ' Grade B/C patients, \n'
                                          'and '+ str(child_con[2,2]) + ' patients are classified correctly')\

    ori_data_prob = pd.concat([ y_parent_pd,y_pred_prob_pd, y_pred_pd, ori_data], axis=1)
    ori_data_prob.to_csv(path)
    # incidence_GradeA = sum(data.iloc[:, 0] == 1 )/y_pred.shape[0]
    # incidence_GradeBC = sum(data.iloc[:, 0] == 2) / y_pred.shape[0]
    # print('Incidence of GradeA '+str(round(incidence_GradeA,3)*100)+'% \n Incidence of GradeB/C ' +
    #       str(round(incidence_GradeBC, 3)*100)+'%')
    # print(child_con)

print('==================The result of internal==================')
Grade_class_result(model, clf, df_PHLF_test_with_grade, ori_int_test
                   ,'./results/' + path + '/data_with_prob/ori_test_Grade2.csv')
print('\n,==================The result of ext12==================')
Grade_class_result(model, clf, df_PHLF_ext12_with_grade, ori_data_ext12
                   ,'./results/' + path + '/data_with_prob/ori_ext12_Grade2.csv')
print('\n,==================The result of ext3==================')
Grade_class_result(model, clf, df_PHLF_ext3_with_grade, ori_data_ext3
                   ,'./results/' + path + '/data_with_prob/ori_ext3_Grade2.csv')
print('\n,==================The result of ext4==================')
# Grade_class_result(model, clf, df_PHLF_ext4_with_grade, ori_data_ext4
#                    ,'./results/' + path + '/data_with_prob/ori_ext4_Grade.pdf')
Grade_class_result(model, clf, df_PHLF_ext6_with_grade, ori_df_ext6
                   ,'./results/' + path + '/data_with_prob/ori_ext6_Grade2.csv')
print('\n,==================The result of ext5==================')
Grade_class_result(model, clf, df_PHLF_ext7_with_grade, ori_df_ext7
                   ,'./results/' + path + '/data_with_prob/ori_ext7_Grade2.csv')
print('\n,==================The result of all data==================')
Grade_class_result(model, clf, df_PHLF_valid_grade, ori_PHLF_valid_grade
                   ,'./results/' + path + '/data_with_prob/ori_all_valid_Grade2.csv')








