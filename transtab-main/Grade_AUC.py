'''
肝衰分等级（A 轻微，B/C 临床意义）AUC 曲线图
Fig 6. a-b
'''
import pandas as pd
import transtab
import torch
import json
from sklearn.model_selection import train_test_split
from child_class import GateChildHead
from Result_process.metrics_process import metrics_with_youden, metrics_multiclass
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
prop = fm.FontProperties(fname=font_path, size=22)

df_PHLF_valid_grade = pd.read_csv('../data_process/df_PHLF_valid_grade.csv')
df_PHLF_extvalid_grade = pd.read_csv('../data_process/df_PHLF_extvalid_grade.csv')
df_PHLF_train_with_grade = pd.read_csv('../data_process/df_PHLF_train_with_grade.csv')

train_data = df_PHLF_train_with_grade
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
# clf = torch.load('./checkpoint/LF_bio_240625_best_018_grade_420.pth')
def Grade_AUC(data, figure_name):
    prob_parent, encoder_output = transtab.evaluator.predict_all_prob(model, data.iloc[:, 2:],
                                                                   data.iloc[:, 1])
    logits_child, loss_child = clf(encoder_output, data.iloc[:, 0], prob_parent, threshold)
    y_pred = logits_child.detach().cpu().numpy()
    result_int_child, child_con = metrics_multiclass(data.iloc[:, 0], y_pred)

    label = data.iloc[:, 0]
    y_true = label_binarize(label, classes=[0, 1, 2])

    y_parent = np.array((prob_parent>0.6).astype(int))
    y_pred_lab = np.argmax(y_pred, axis=1)

    data_with_pred = pd.concat([pd.DataFrame(y_pred_lab, columns=['Pred_Grade']),
                                pd.DataFrame(y_parent, columns=['Pred_PHLF']),data], axis=1)
    omission_grade = ((data_with_pred['PHLFG'] == 2 )& (data_with_pred['Pred_Grade'] == 0)).sum()
    omission_grade_total = ((data_with_pred['PHLFG'] == 2 )& (data_with_pred['Pred_Grade'] == 0) &
                            (data_with_pred['Pred_PHLF'] == 0)).sum()

    data_with_pred.to_csv('./results/' + path + '/AUC/data_with_pred_'+figure_name+'.csv')
    print('The omission grade is: ' + str(omission_grade), 'and the total omission is: ' + str(omission_grade_total))

    fpr = dict()
    fpr_mean = dict()
    tpr = dict()
    roc_auc = dict()
    tprs_lower_dict = dict()
    tprs_upper_dict = dict()
    CI = dict()

    for i in range(3):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]

        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_pred_class)
        roc_auc[i] = auc(fpr[i], tpr[i])

        n_bootstraps = 1000
        rng = np.random.RandomState(42)
        bootstrapped_score = []
        bootstrapped_tprs = []

        mean_fpr = np.linspace(0, 1, 100)  # Fixed grid for interpolation

        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_true_class), len(y_pred_class))
            if len(np.unique(y_true_class[indices])) < 2:
                continue
            score = roc_auc_score(y_true_class[indices], y_pred_class[indices])
            fpr_bs, tpr_bs, _ = roc_curve(y_true_class[indices], y_pred_class[indices])
            bootstrapped_score.append(score)
            bootstrapped_tprs.append(np.interp(mean_fpr, fpr_bs, tpr_bs))  # Interpolate TPR on fixed grid

        sorted_scores = np.array(bootstrapped_score)
        bootstrapped_tprs = np.array(bootstrapped_tprs)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(len(sorted_scores) * 0.025)]
        confidence_upper = sorted_scores[int(len(sorted_scores) * 0.975)]
        tprs_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
        tprs_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)

        CI[i] = ('( {}, {})'.format(round(confidence_lower, 3), round(confidence_upper, 3)))
        tprs_lower_dict[i] = tprs_lower
        tprs_upper_dict[i] = tprs_upper

        # Interpolate the original TPR onto the fixed FPR grid
        fpr_mean[i] = mean_fpr
        # tpr[i] = np.interp(mean_fpr, fpr[i], tpr[i])

    lw = 3
    plt.figure(figsize=[11, 10])
    plt.plot(fpr[0], tpr[0], color='#71b1d7', alpha=1, lw=lw,
                 label='Non-PHLF: AUC = %0.3f ' % roc_auc[0] + CI[0])
    plt.fill_between(fpr_mean[0], tprs_lower_dict[0], tprs_upper_dict[0],
                         color='#71b1d7',  alpha=0.1)

    plt.plot(fpr[1], tpr[1], color='#66bc98', alpha=1, lw=lw,
                 label='PHLF Grade A: AUC = %0.3f ' % roc_auc[1]+ CI[1])
    plt.fill_between(fpr_mean[1], tprs_lower_dict[1], tprs_upper_dict[1],
                         color='#66bc98',  alpha=0.1)

    plt.plot(fpr[2], tpr[2], color='#d24d3e', alpha=1, lw=lw,
                 label='PHLF Grade B/C: AUC = %0.3f ' % roc_auc[2]+ CI[2])
    plt.fill_between(fpr_mean[2], tprs_lower_dict[2], tprs_upper_dict[2],
                         color='#d24d3e',  alpha=0.1)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=22, fontproperties=prop)
    plt.yticks(fontsize=22, fontproperties=prop)
    plt.xlabel('FPR', fontsize=24, fontproperties=prop)
    plt.ylabel('TPR', fontsize=24, fontproperties=prop)
    plt.title('ROC curves for Grade B/C prediction', fontsize=26, pad=20,
              fontweight='bold', fontproperties=prop)
    plt.legend(loc="lower right", fontsize=22, prop=prop)
    ax = plt.gca()
    plt.show()
    # plt.savefig('./results/' + path + '/AUC/'+figure_name+'.pdf')

Grade_AUC (train_data, figure_name='Train_GradeBC_AUC')
Grade_AUC (df_PHLF_valid_grade, figure_name='Test_GradeBC_AUC')
