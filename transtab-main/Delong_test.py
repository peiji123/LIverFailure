import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score, roc_curve
from statsmodels.stats.proportion import proportion_confint
import json
import scipy.stats as st

from scipy.stats import norm
def clac_sennpv_and_CI(y_true, preds,alpha=0.05, optimal_threshold=0.6, method='wilson'):
    fpr, tpr, thresholds = roc_curve(y_true, preds)

    # Calculate Youden Index for each threshold
    youden_index = tpr - fpr  # J = sensitivity + specificity - 1; sensitivity = tpr, specificity = 1 - fpr
    optimal_idx = np.argmax(youden_index)  # Find the index of the maximum Youden Index
    optimal_threshold = thresholds[optimal_idx]  # Get the corresponding threshold

    sorted_indices = np.argsort(-youden_index)  # Descending order
    top_5_indices = sorted_indices[:10]
    top_5_youden = youden_index[top_5_indices]
    top_5_thresholds = thresholds[top_5_indices]

    optimal_threshold = max(top_5_thresholds)
    # optimal_threshold = top_5_thresholds[1]
    # Apply the optimal threshold to predictions
    y_pred = (np.array(preds) > optimal_threshold).astype(int)

    cn = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cn.ravel()
    sensitivity = tp / (tp + fn)
    sen_ci_lower, sen_ci_upper = proportion_confint(tp, tp + fn, alpha = alpha, method=method)
    npv = tn/(tn + fn)
    npv_ci_lower, npv_ci_upper = proportion_confint(tn, tn + fn, alpha = alpha, method=method)
    print('Sensitivity: %.3f' % sensitivity,
          '( {}, {})'.format(round(sen_ci_lower, 3), round(sen_ci_upper, 3)))
    print('npv: %.3f' % npv,
          '( {}, {})'.format(round(npv_ci_lower, 3), round(npv_ci_upper, 3)))
    return sensitivity, sen_ci_lower, sen_ci_upper, npv, npv_ci_lower, npv_ci_upper


class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y


    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)
        print('{:.3f}'.format(auc_A), '{:.3f}'.format(auc_B))

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2

        return z, p
    def other_metrics (self):
        auc = roc_auc_score(y_true=self._label, y_score=self._preds2)
        y_preds = (np.array(self._preds2) > 0.5).astype(int)
        acc = accuracy_score(y_true=self._label, y_pred=y_preds)
        recall = recall_score(y_true=self._label, y_pred=y_preds)
        precision = precision_score(y_true=self._label, y_pred=y_preds, labels=[0])
        f1 = f1_score(y_true=self._label, y_pred=y_preds, )
        kappa = cohen_kappa_score(y1=self._label, y2=y_preds, )
        mcc = matthews_corrcoef(y_true=self._label, y_pred=y_preds)
        con = confusion_matrix(self._label, y_preds)
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
        specificity = TN / (TN + FP)
        z, p = self._compute_z_p()
        print(con)
        print('acc:', '{:.3f}'.format(acc), 'f1:', '{:.3f}'.format(f1), 'auc:', '{:.3f}'.format(auc),
              'kappa:', '{:.3f}'.format(kappa), 'MCC:', '{:.3f}'.format(mcc), 'recall:', '{:.3f}'.format(recall),
              'specificity:', '{:.3f}'.format(specificity), 'PPV:', '{:.3f}'.format(PPV), 'NPV:', '{:.3f}'.format(NPV))
        print('{:.3f}'.format(auc), '{:.8f}'.format(p),'{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall), \
              '{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV), '\n')
        return '{:.3f}'.format(auc), '{:.8f}'.format(p),'{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall), \
              '{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV)
    def other_metrics_youden (self, preds):
        fpr, tpr, thresholds = roc_curve(self._label, preds)

        # Calculate Youden Index for each threshold
        youden_index = tpr - fpr  # J = sensitivity + specificity - 1; sensitivity = tpr, specificity = 1 - fpr
        optimal_idx = np.argmax(youden_index)  # Find the index of the maximum Youden Index
        optimal_threshold = thresholds[optimal_idx]  # Get the corresponding threshold

        sorted_indices = np.argsort(-youden_index)  # Descending order
        top_5_indices = sorted_indices[:10]
        top_5_youden = youden_index[top_5_indices]
        top_5_thresholds = thresholds[top_5_indices]

        optimal_threshold = max(top_5_thresholds)
        # optimal_threshold = top_5_thresholds[1]
        # Apply the optimal threshold to predictions
        y_preds = (np.array(preds) > optimal_threshold).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_true=self._label, y_pred=y_preds)
        recall = recall_score(y_true=self._label, y_pred=y_preds)
        precision = precision_score(y_true=self._label, y_pred=y_preds, labels=[0])
        f1 = f1_score(y_true=self._label, y_pred=y_preds)
        kappa = cohen_kappa_score(y1=self._label, y2=y_preds)
        mcc = matthews_corrcoef(y_true=self._label, y_pred=y_preds)
        con = confusion_matrix(self._label, y_preds)

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
        specificity = TN / (TN + FP)
        # Compute AUC and DeLong test p-value
        auc = roc_auc_score(y_true=self._label, y_score=preds)
        z, p = self._compute_z_p()

        print(con)
        print('Optimal Threshold:', '{:.3f}'.format(optimal_threshold))
        print('acc:', '{:.3f}'.format(acc), 'f1:', '{:.3f}'.format(f1), 'auc:', '{:.3f}'.format(auc),
              'kappa:', '{:.3f}'.format(kappa), 'MCC:', '{:.3f}'.format(mcc), 'recall:', '{:.3f}'.format(recall),
              'specificity:', '{:.3f}'.format(specificity), 'PPV:', '{:.3f}'.format(PPV), 'NPV:', '{:.3f}'.format(NPV))
        print('{:.3f}'.format(auc), '{:.8f}'.format(p), '{:.3f}'.format(acc), '{:.3f}'.format(f1),
              '{:.3f}'.format(recall),'{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV), '\n')
        return (
            '{:.3f}'.format(auc), '{:.8f}'.format(p), '{:.3f}'.format(acc), '{:.3f}'.format(f1),
            '{:.3f}'.format(recall), '{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV),
            '{:.3f}'.format(optimal_threshold)  # Add the optimal threshold to the returned tuple
        )
    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"z score = {z:.8f};\np value = {p:.8f};")
        if p < self.threshold:
            print("There is a significant difference")
        else:
            print("There is NO significant difference")


# 示例数据
# 模型2的预测分数
'''
lr, rf, xgb, svc, lda, lgb, mlp, saint, tabnet, tabpfn, transtab
'''
benchmark = 'lr'

result_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_int_result.json'))
result_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext12_result.json'))
result_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext3_result.json'))
result_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext4_result.json'))
result_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_mimic_result.json'))
result_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext6_result.json'))
result_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_our_ext7_result.json'))

if benchmark == 'lr':
    result_lr_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_int_result.json'))
    result_lr_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext12_result.json'))
    result_lr_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext3_result.json'))
    result_lr_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext4_result.json'))
    result_lr_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_mimic_result.json'))
    result_lr_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext6_result.json'))
    result_lr_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lr4_ext7_result.json'))

    print('Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_lr_int_dict['pred'], result_lr_int_dict['real'])
    print('\n','Result of the lr in external_12 dataset')
    DelongTest(result_ext12_dict['pred'], result_lr_ext12_dict['pred'], result_lr_ext12_dict['real'])
    print('\n','Result of the lr in external_3 dataset')
    DelongTest(result_ext3_dict['pred'], result_lr_ext3_dict['pred'], result_lr_ext3_dict['real'])
    print('\n','Result of the lr in external_4 dataset')
    DelongTest(result_ext4_dict['pred'], result_lr_ext4_dict['pred'], result_lr_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_lr_mimic_dict['pred'], result_lr_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_lr_ext6_dict['pred'], result_lr_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_lr_ext7_dict['pred'], result_lr_ext7_dict['real'])

    '''
    Average results
    '''
    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_lr_pred = (result_lr_int_dict['pred'] + result_lr_ext12_dict['pred'] + result_lr_ext3_dict['pred'] +
                    result_lr_ext4_dict['pred'])
    print('\n', 'Result of the lr in average four dataset')
    DelongTest(four_pred, four_lr_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_lr_pred = (result_lr_ext12_dict['pred'] + result_lr_ext3_dict['pred'] +
                    result_lr_ext4_dict['pred']+result_lr_ext6_dict['pred']+result_lr_ext7_dict['pred'])
    print('\n', 'Result of the lr in average five dataset')
    # DelongTest(five_pred, five_lr_pred, five_real)
    delong_test = DelongTest(five_pred, five_lr_pred, five_real)
    delong_test.other_metrics()
    delong_test.other_metrics_youden(five_pred)
    clac_sennpv_and_CI(five_real,five_pred)


    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_lr_pred = four_lr_pred  + result_lr_ext6_dict['pred'] + result_lr_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_lr_pred, six_real)
    delong_test.other_metrics_youden(six_lr_pred)
    delong_test.other_metrics_youden(six_pred)
    clac_sennpv_and_CI(six_real, six_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_lr_pred = six_lr_pred + result_lr_mimic_dict['pred']
    print('\n', 'Result of the lr in average seven dataset')
    DelongTest(seven_pred, seven_lr_pred, seven_real)

if benchmark == 'rf':
    result_rf_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf14_int_result.json'))
    result_rf_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf14_ext12_result.json'))
    result_rf_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf14_ext3_result.json'))
    result_rf_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf14_ext4_result.json'))
    result_rf_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf19_mimic_result.json'))
    result_rf_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf19_ext6_result.json'))
    result_rf_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_rf19_ext7_result.json'))

    print('========================================================================','\n')
    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_rf_int_dict['pred'], result_rf_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_rf_ext12_dict['pred'], result_rf_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_rf_ext3_dict['pred'], result_rf_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_rf_ext4_dict['pred'], result_rf_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_rf_mimic_dict['pred'], result_rf_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_rf_ext6_dict['pred'], result_rf_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_rf_ext7_dict['pred'], result_rf_ext7_dict['real'])


    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_rf_pred = (result_rf_int_dict['pred'] + result_rf_ext12_dict['pred'] + result_rf_ext3_dict['pred'] +
                    result_rf_ext4_dict['pred'])
    print('\n', 'Result of the rf in average four dataset')
    DelongTest(four_pred, four_rf_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_rf_pred = (result_rf_ext12_dict['pred'] + result_rf_ext3_dict['pred'] +
                    result_rf_ext4_dict['pred']+result_rf_ext6_dict['pred']+result_rf_ext7_dict['pred'])
    print('\n', 'Result of the rf in average five dataset')
    delong_test = DelongTest(five_pred, five_rf_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)

    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_rf_pred = four_rf_pred  + result_rf_ext6_dict['pred'] + result_rf_ext7_dict['pred']
    print('\n', 'Result of the rf in average six dataset')
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_rf_pred, six_real)
    delong_test.other_metrics_youden(six_rf_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_rf_pred = six_rf_pred + result_rf_mimic_dict['pred']
    print('\n', 'Result of the rf in average seven dataset')
    DelongTest(seven_pred, seven_rf_pred, seven_real)

if benchmark == 'xgb':
    result_xgb_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb2_int_result.json'))
    result_xgb_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb2_ext12_result.json'))
    result_xgb_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb2_ext3_result.json'))
    result_xgb_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb2_ext4_result.json'))
    result_xgb_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb4_mimic_result.json'))
    result_xgb_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb4_ext6_result.json'))
    result_xgb_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_xgb4_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_xgb_int_dict['pred'], result_xgb_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_xgb_ext12_dict['pred'], result_xgb_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_xgb_ext3_dict['pred'], result_xgb_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_xgb_ext4_dict['pred'], result_xgb_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_xgb_mimic_dict['pred'], result_xgb_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_xgb_ext6_dict['pred'], result_xgb_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_xgb_ext7_dict['pred'], result_xgb_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_xgb_pred = (result_xgb_int_dict['pred'] + result_xgb_ext12_dict['pred'] + result_xgb_ext3_dict['pred'] +
                    result_xgb_ext4_dict['pred'])
    print('\n', 'Result of the xgb in average four dataset')
    DelongTest(four_pred, four_xgb_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_xgb_pred = (result_xgb_ext12_dict['pred'] + result_xgb_ext3_dict['pred'] +
                     result_xgb_ext4_dict['pred']+result_xgb_ext6_dict['pred']+result_xgb_ext7_dict['pred'])
    print('\n', 'Result of the xgb in average five dataset')
    delong_test = DelongTest(five_pred, five_xgb_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)


    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_xgb_pred = four_xgb_pred  + result_xgb_ext6_dict['pred'] + result_xgb_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_xgb_pred, six_real)
    delong_test.other_metrics_youden(six_xgb_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_xgb_pred = six_xgb_pred + result_xgb_mimic_dict['pred']
    print('\n', 'Result of the xgb in average seven dataset')
    DelongTest(seven_pred, seven_xgb_pred, seven_real)


if benchmark == 'svc':
    result_svc_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_int_result.json'))
    result_svc_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext12_result.json'))
    result_svc_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext3_result.json'))
    result_svc_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext4_result.json'))
    result_svc_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_mimic_result.json'))
    result_svc_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext6_result.json'))
    result_svc_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_svc1_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_svc_int_dict['pred'], result_svc_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_svc_ext12_dict['pred'], result_svc_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_svc_ext3_dict['pred'], result_svc_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_svc_ext4_dict['pred'], result_svc_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_svc_mimic_dict['pred'], result_svc_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_svc_ext6_dict['pred'], result_svc_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_svc_ext7_dict['pred'], result_svc_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_svc_pred = (result_svc_int_dict['pred'] + result_svc_ext12_dict['pred'] + result_svc_ext3_dict['pred'] +
                    result_svc_ext4_dict['pred'])
    print('\n', 'Result of the svc in average four dataset')
    DelongTest(four_pred, four_svc_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'])
    five_svc_pred = (result_svc_ext12_dict['pred'] + result_svc_ext3_dict['pred'] +
                     result_svc_ext4_dict['pred'])
    print('\n', 'Result of the svc in average five dataset')
    delong_test = DelongTest(five_pred, five_svc_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)

    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_svc_pred = four_svc_pred  + result_svc_ext6_dict['pred'] + result_svc_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_svc_pred, six_real)
    delong_test.other_metrics_youden(six_svc_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_svc_pred = six_svc_pred + result_svc_mimic_dict['pred']
    print('\n', 'Result of the svc in average seven dataset')
    DelongTest(seven_pred, seven_svc_pred, seven_real)


if benchmark == 'lda':
    result_lda_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_int_result.json'))
    result_lda_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext12_result.json'))
    result_lda_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext3_result.json'))
    result_lda_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext4_result.json'))
    result_lda_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_mimic_result.json'))
    result_lda_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext6_result.json'))
    result_lda_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lda_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_lda_int_dict['pred'], result_lda_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_lda_ext12_dict['pred'], result_lda_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_lda_ext3_dict['pred'], result_lda_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_lda_ext4_dict['pred'], result_lda_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_lda_mimic_dict['pred'], result_lda_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_lda_ext6_dict['pred'], result_lda_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_lda_ext7_dict['pred'], result_lda_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_lda_pred = (result_lda_int_dict['pred'] + result_lda_ext12_dict['pred'] + result_lda_ext3_dict['pred'] +
                    result_lda_ext4_dict['pred'])
    print('\n', 'Result of the lda in average four dataset')
    DelongTest(four_pred, four_lda_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_lda_pred = (result_lda_ext12_dict['pred'] + result_lda_ext3_dict['pred'] +
                     result_lda_ext4_dict['pred']+result_lda_ext6_dict['pred']+result_lda_ext7_dict['pred'])
    print('\n', 'Result of the lda in average five dataset')
    delong_test = DelongTest(five_pred, five_lda_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)

    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_lda_pred = four_lda_pred  + result_lda_ext6_dict['pred'] + result_lda_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_lda_pred, six_real)
    delong_test.other_metrics_youden(six_lda_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_lda_pred = six_lda_pred + result_lda_mimic_dict['pred']
    print('\n', 'Result of the lda in average seven dataset')
    DelongTest(seven_pred, seven_lda_pred, seven_real)


if benchmark == 'lgb':
    result_lgb_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_int_result.json'))
    result_lgb_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext12_result.json'))
    result_lgb_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext3_result.json'))
    result_lgb_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext4_result.json'))
    result_lgb_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_mimic_result.json'))
    result_lgb_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext6_result.json'))
    result_lgb_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_lgb3_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_lgb_int_dict['pred'], result_lgb_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_lgb_ext12_dict['pred'], result_lgb_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_lgb_ext3_dict['pred'], result_lgb_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_lgb_ext4_dict['pred'], result_lgb_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_lgb_mimic_dict['pred'], result_lgb_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_lgb_ext6_dict['pred'], result_lgb_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_lgb_ext7_dict['pred'], result_lgb_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_lgb_pred = (result_lgb_int_dict['pred'] + result_lgb_ext12_dict['pred'] + result_lgb_ext3_dict['pred'] +
                    result_lgb_ext4_dict['pred'])
    print('\n', 'Result of the lgb in average four dataset')
    DelongTest(four_pred, four_lgb_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_lgb_pred = (result_lgb_ext12_dict['pred'] + result_lgb_ext3_dict['pred'] +
                     result_lgb_ext4_dict['pred']+result_lgb_ext6_dict['pred']+result_lgb_ext7_dict['pred'])
    print('\n', 'Result of the lgb in average five dataset')
    delong_test = DelongTest(five_pred, five_lgb_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)

    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_lgb_pred = four_lgb_pred  + result_lgb_ext6_dict['pred'] + result_lgb_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_lgb_pred, six_real)
    delong_test.other_metrics_youden(six_lgb_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_lgb_pred = six_lgb_pred + result_lgb_mimic_dict['pred']
    print('\n', 'Result of the lgb in average seven dataset')
    DelongTest(seven_pred, seven_lgb_pred, seven_real)

if benchmark == 'mlp':
    result_mlp_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp_int_result.json'))
    result_mlp_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp_ext12_result.json'))
    result_mlp_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp_ext3_result.json'))
    result_mlp_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp_ext4_result.json'))
    result_mlp_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp5_mimic_result.json'))
    result_mlp_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp5_ext6_result.json'))
    result_mlp_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_mlp5_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_mlp_int_dict['pred'], result_mlp_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_mlp_ext12_dict['pred'], result_mlp_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_mlp_ext3_dict['pred'], result_mlp_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_mlp_ext4_dict['pred'], result_mlp_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_mlp_mimic_dict['pred'], result_mlp_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_mlp_ext6_dict['pred'], result_mlp_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_mlp_ext7_dict['pred'], result_mlp_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_mlp_pred = (result_mlp_int_dict['pred'] + result_mlp_ext12_dict['pred'] + result_mlp_ext3_dict['pred'] +
                    result_mlp_ext4_dict['pred'])
    print('\n', 'Result of the mlp in average four dataset')
    DelongTest(four_pred, four_mlp_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_mlp_pred = (result_mlp_ext12_dict['pred'] + result_mlp_ext3_dict['pred'] +
                    result_mlp_ext4_dict['pred']+result_mlp_ext6_dict['pred']+result_mlp_ext7_dict['pred'])
    print('\n', 'Result of the mlp in average five dataset')
    delong_test = DelongTest(five_pred, five_mlp_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)

    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_mlp_pred = four_mlp_pred  + result_mlp_ext6_dict['pred'] + result_mlp_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_mlp_pred, six_real)
    delong_test.other_metrics_youden(six_mlp_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_mlp_pred = six_mlp_pred + result_mlp_mimic_dict['pred']
    print('\n', 'Result of the mlp in average seven dataset')
    DelongTest(seven_pred, seven_mlp_pred, seven_real)

if benchmark == 'saint':
    result_saint_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint2_int_result.json'))
    result_saint_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint2_ext12_result.json'))
    result_saint_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint2_ext3_result.json'))
    result_saint_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint2_ext4_result.json'))
    result_saint_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint3_mimic_result.json'))
    result_saint_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint3_ext6_result.json'))
    result_saint_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_saint3_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_saint_int_dict['pred'], result_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_saint_ext12_dict['pred'], result_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_saint_ext3_dict['pred'], result_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_saint_ext4_dict['pred'], result_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_saint_mimic_dict['pred'], result_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_saint_ext6_dict['pred'], result_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_saint_ext7_dict['pred'], result_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_saint_pred = (result_saint_int_dict['pred'] + result_saint_ext12_dict['pred'] + result_saint_ext3_dict['pred'] +
                    result_saint_ext4_dict['pred'])
    print('\n', 'Result of the saint in average four dataset')
    DelongTest(four_pred, four_saint_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_saint_pred = (result_saint_ext12_dict['pred'] + result_saint_ext3_dict['pred'] +
                       result_saint_ext4_dict['pred']+result_saint_ext6_dict['pred']+result_saint_ext7_dict['pred'])
    print('\n', 'Result of the saint in average five dataset')
    delong_test = DelongTest(five_pred, five_saint_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)

    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_saint_pred = four_saint_pred  + result_saint_ext6_dict['pred'] + result_saint_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_saint_pred, six_real)
    delong_test.other_metrics_youden(six_saint_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_saint_pred = six_saint_pred + result_saint_mimic_dict['pred']
    print('\n', 'Result of the saint in average seven dataset')
    DelongTest(seven_pred, seven_saint_pred, seven_real)

if benchmark == 'tabnet':
    result_tabnet_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_int_result.json'))
    result_tabnet_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext12_result.json'))
    result_tabnet_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext3_result.json'))
    result_tabnet_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext4_result.json'))
    result_tabnet_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_mimic_result.json'))
    result_tabnet_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext6_result.json'))
    result_tabnet_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabnet2_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_tabnet_int_dict['pred'], result_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_tabnet_ext12_dict['pred'], result_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_tabnet_ext3_dict['pred'], result_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_tabnet_ext4_dict['pred'], result_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_tabnet_mimic_dict['pred'], result_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_tabnet_ext6_dict['pred'], result_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_tabnet_ext7_dict['pred'], result_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_tabnet_pred = (result_tabnet_int_dict['pred'] + result_tabnet_ext12_dict['pred'] + result_tabnet_ext3_dict['pred'] +
                    result_tabnet_ext4_dict['pred'])
    print('\n', 'Result of the tabnet in average four dataset')
    DelongTest(four_pred, four_tabnet_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_tabnet_pred = (result_tabnet_ext12_dict['pred'] + result_tabnet_ext3_dict['pred'] +
                        result_tabnet_ext4_dict['pred']+result_tabnet_ext6_dict['pred']+result_tabnet_ext7_dict['pred'])
    print('\n', 'Result of the tabnet in average five dataset')
    delong_test = DelongTest(five_pred, five_tabnet_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)


    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_tabnet_pred = four_tabnet_pred  + result_tabnet_ext6_dict['pred'] + result_tabnet_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_tabnet_pred, six_real)
    delong_test.other_metrics_youden(six_tabnet_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_tabnet_pred = six_tabnet_pred + result_tabnet_mimic_dict['pred']
    print('\n', 'Result of the tabnet in average seven dataset')
    DelongTest(seven_pred, seven_tabnet_pred, seven_real)

if benchmark == 'tabpfn':
    result_tabpfn_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn3_int_result.json'))
    result_tabpfn_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn3_ext12_result.json'))
    result_tabpfn_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn3_ext3_result.json'))
    result_tabpfn_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn3_ext4_result.json'))
    result_tabpfn_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn4_mimic_result.json'))
    result_tabpfn_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn4_ext6_result.json'))
    result_tabpfn_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_tabpfn4_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_tabpfn_int_dict['pred'], result_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_tabpfn_ext12_dict['pred'], result_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_tabpfn_ext3_dict['pred'], result_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_tabpfn_ext4_dict['pred'], result_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_tabpfn_mimic_dict['pred'], result_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_tabpfn_ext6_dict['pred'], result_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_tabpfn_ext7_dict['pred'], result_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_tabpfn_pred = (result_tabpfn_int_dict['pred'] + result_tabpfn_ext12_dict['pred'] + result_tabpfn_ext3_dict['pred'] +
                    result_tabpfn_ext4_dict['pred'])
    print('\n', 'Result of the tabpfn in average four dataset')
    DelongTest(four_pred, four_tabpfn_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_tabpfn_pred = (result_tabpfn_ext12_dict['pred'] + result_tabpfn_ext3_dict['pred'] +
                        result_tabpfn_ext4_dict['pred']+result_tabpfn_ext6_dict['pred']+result_tabpfn_ext7_dict['pred'])
    print('\n', 'Result of the tabpfn in average five dataset')
    delong_test = DelongTest(five_pred, five_tabpfn_pred, five_real)
    # delong_test.other_metrics()
    delong_test.other_metrics_youden(five_tabpfn_pred)

    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_tabpfn_pred = four_tabpfn_pred  + result_tabpfn_ext6_dict['pred'] + result_tabpfn_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_tabpfn_pred, six_real)
    delong_test.other_metrics_youden(six_tabpfn_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_tabpfn_pred = six_tabpfn_pred + result_tabpfn_mimic_dict['pred']
    print('\n', 'Result of the tabpfn in average seven dataset')
    DelongTest(seven_pred, seven_tabpfn_pred, seven_real)

if benchmark == 'transtab':
    result_transtab_int_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab3_int_result.json'))
    result_transtab_ext12_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab3_ext12_result.json'))
    result_transtab_ext3_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab3_ext3_result.json'))
    result_transtab_ext4_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab3_ext4_result.json'))
    result_transtab_mimic_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab2_mimic_result.json'))
    result_transtab_ext6_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab2_ext6_result.json'))
    result_transtab_ext7_dict = json.load(open('/home/hci/QYang/YQ_LiverFailure/transtab-main/Real_Prediction/dict_transtab2_ext7_result.json'))

    print('\n','Result of the lr in internal dataset')
    DelongTest(result_int_dict['pred'], result_transtab_int_dict['pred'], result_int_dict['real'])
    print('\n','Result of the lr in external12 dataset')
    DelongTest(result_ext12_dict['pred'], result_transtab_ext12_dict['pred'], result_ext12_dict['real'])
    print('\n','Result of the lr in external3 dataset')
    DelongTest(result_ext3_dict['pred'], result_transtab_ext3_dict['pred'], result_ext3_dict['real'])
    print('\n','Result of the lr in external4 dataset')
    DelongTest(result_ext4_dict['pred'], result_transtab_ext4_dict['pred'], result_ext4_dict['real'])
    print('\n','Result of the lr in mimic dataset')
    DelongTest(result_mimic_dict['pred'], result_transtab_mimic_dict['pred'], result_mimic_dict['real'])
    print('\n','Result of the lr in external6 dataset')
    DelongTest(result_ext6_dict['pred'], result_transtab_ext6_dict['pred'], result_ext6_dict['real'])
    print('\n','Result of the lr in external7 dataset')
    DelongTest(result_ext7_dict['pred'], result_transtab_ext7_dict['pred'], result_ext7_dict['real'])

    print('=====================================================')
    four_real = (result_int_dict['real'] + result_ext12_dict['real'] + result_ext3_dict['real'] +
                 result_ext4_dict['real'])
    four_pred = (result_int_dict['pred'] + result_ext12_dict['pred'] + result_ext3_dict['pred'] +
                 result_ext4_dict['pred'])
    four_transtab_pred = (result_transtab_int_dict['pred'] + result_transtab_ext12_dict['pred'] + result_transtab_ext3_dict['pred'] +
                    result_transtab_ext4_dict['pred'])
    print('\n', 'Result of the transtab in average four dataset')
    DelongTest(four_pred, four_transtab_pred, four_real)

    print('=====================================================')
    five_real = (result_ext12_dict['real'] + result_ext3_dict['real'] + result_ext4_dict['real'] +
                 result_ext6_dict['real'] + result_ext7_dict['real'])
    five_pred = (result_ext12_dict['pred'] + result_ext3_dict['pred'] + result_ext4_dict['pred'] +
                 result_ext6_dict['pred'] + result_ext7_dict['pred'])
    five_transtab_pred = (result_transtab_ext12_dict['pred'] + result_transtab_ext3_dict['pred'] +
                          result_transtab_ext4_dict['pred']+result_transtab_ext6_dict['pred']+result_transtab_ext7_dict['pred'])
    print('\n', 'Result of the transtab in average five dataset')
    delong_test = DelongTest(five_pred, five_transtab_pred, five_real)
    delong_test.other_metrics()
    # delong_test.other_metrics_youden(five_pred)

    six_real = four_real  + result_ext6_dict['real'] + result_ext7_dict['real']
    six_pred = four_pred + result_ext6_dict['pred'] + result_ext7_dict['pred']
    six_transtab_pred = four_transtab_pred  + result_transtab_ext6_dict['pred'] + result_transtab_ext7_dict['pred']
    print('=====================================================')
    print('\n', 'Result of the lr in average six dataset')
    # DelongTest(six_pred, six_lr_pred, six_real)
    # DelongTest.other_metrics()
    delong_test = DelongTest(six_pred, six_transtab_pred, six_real)
    delong_test.other_metrics_youden(six_transtab_pred)
    print('=====================================================')

    seven_real = six_real + result_mimic_dict['real']
    seven_pred = six_pred + result_mimic_dict['pred']
    seven_transtab_pred = six_transtab_pred + result_transtab_mimic_dict['pred']
    print('\n', 'Result of the transtab in average seven dataset')
    DelongTest(seven_pred, seven_transtab_pred, seven_real)

    # auc = roc_auc_score(y_true= result_lr_ext3_dict['real'], y_score = result_lr_ext3_dict['pred'])

