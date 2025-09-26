import transtab
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transtab.evaluator import predict_multi_task
from Result_process.metrics_process import metrics_with_youden, metrics_multiclass
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np
import itertools
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer, TabTransformer
import torch.optim as optim
from child_class import GateChildHead
import itertools

def custrm_metric(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    metrics_value = cm[1,1] + cm[2,2]- cm[2,0]
    return metrics_value, cm[2,2], cm[1,1]

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



info = json.load(open('../data_process/info_0603.json'))
cate_cols = info['cate_cols']
num_cols = info['cont_cols']
bin_cols = info['bin_cols']

df_PHLF_valid_grade = pd.read_csv('../data_process/df_PHLF_valid_grade.csv')
df_PHLF_extvalid_grade = pd.read_csv('../data_process/df_PHLF_extvalid_grade.csv')
df_PHLF_train_with_grade = pd.read_csv('../data_process/df_PHLF_train_with_grade.csv')

train_data, valid_data = train_test_split(df_PHLF_train_with_grade, test_size=0.2, random_state=42)
train_set = [train_data.iloc[:,2:], train_data.iloc[:,0]]
valid_set = [valid_data.iloc[:,2:], valid_data.iloc[:,0]]

param_grid = {
    'batch_size': [8, 16, 32,64],
    'lr': [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, ],
    'weight_decay': [1e-4, 1e-5, 1e-3],
    'eval_batch_size': [64, 128, 256],
    'imb_weight' :[1.5, 2, 2.5, 3., 3.5,]
}

training_arguments = {
    'batch_size':8,
    'lr':0.0005,
    'weight_decay':1e-05,
    'patience' :5,
    'eval_batch_size':128,
    'num_epoch' :200,
    'imb_weight': 2
}

path = 'LF_bio_240625_best_018'
threshold=0.6
model = transtab.build_classifier_multi_task_onlyclss(cate_cols, num_cols, bin_cols)
model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+path)

prob_parent, encoder_output = transtab.evaluator.predict_all_prob(model, train_data.iloc[:, 2:],
                                                               train_data.iloc[:, 1])
prob_parent_valid, encoder_output_valid = transtab.evaluator.predict_all_prob(model, train_data.iloc[:, 2:],
                                                               train_data.iloc[:, 1])

lr_values = [0.0001,0.0002,0.0005,0.001,0.002, 0.005, 0.01, 0.02, 0.05,0.1]
epochs = 500
weight_decay_values = [1e-3,1e-4,1e-5,1e-6]
list1 = [1.0]
list2 = np.arange(2.0, 10.0, 1.).tolist()
list3 = np.arange(3.0, 20.0, 1.).tolist()
weights_combinations = list(itertools.product(list1, list2, list3))

PATH = 'LF_bio_241006_best_0'
count = 0
for lr in lr_values:
    for weight_decay in weight_decay_values:
        for weight in weights_combinations:

            clf = GateChildHead(
                parent_classes=2,
                weights=weight,
                child_classes=3)
            optimizer = optim.Adam(
                clf.parameters(),  # Pass the model's parameters to optimize
                lr=lr,            # Learning rate # Coefficients for computing running averages of gradient and its square
                eps=1e-08,           # Term added to the denominator to improve numerical stability
                weight_decay=weight_decay,    # Weight decay (L2 penalty)
                amsgrad=False        # Whether to use the AMSGrad variant of Adam
            )
            early_stopping = EarlyStopping()
            prob_parent_test, encoder_output_test = transtab.evaluator.predict_all_prob(model, df_PHLF_valid_grade.iloc[:, 2:],
                                                                           df_PHLF_valid_grade.iloc[:, 1])
            for ep in range(epochs):
                clf.train()
                optimizer.zero_grad()
                logits_child, loss_child = clf(encoder_output, train_data.iloc[:, 0], prob_parent, threshold)
                loss_child.backward()
                optimizer.step()  # Update the model's parameters

                # print(f"Epoch [{ep + 1}/{epochs}], Loss: {loss_child.item():.4f}")
                clf.eval()
                with torch.no_grad():
                    logits_child, loss_child = clf(encoder_output_valid, train_data.iloc[:, 0],
                                                   prob_parent_valid, threshold)
                    logits_child = logits_child.detach().cpu().numpy()
                    # result_int_child, child_con = metrics_multiclass(df_PHLF_valid_grade.iloc[:, 0], logits_child)
                metric_value, GradeBC, GradeA = custrm_metric(train_data.iloc[:, 0], logits_child)
                early_stopping(metric_value)
                if early_stopping.early_stop:

                    logits_child, loss_child = clf(encoder_output_test, df_PHLF_valid_grade.iloc[:, 0],
                                                   prob_parent_test, threshold)
                    logits_child = logits_child.detach().cpu().numpy()
                    result_int_child, child_con = metrics_multiclass(df_PHLF_valid_grade.iloc[:, 0], logits_child)
                    metric_value, GradeBC, GradeA = custrm_metric(df_PHLF_valid_grade.iloc[:, 0], logits_child)
                    print(result_int_child,'\n', child_con)
                    print(f'Parameter: lr={lr},weight_decay={weight_decay},weight={weight} ')
                    print(f"Early stopping at epoch {ep + 1}, metric_value is {metric_value}", '\n' )

                    if metric_value >15 and GradeBC > 15 and GradeA > 20:

                        torch.save(clf.state_dict(), './checkpoint/' + path + '_grade_debug_2Layers_' + str(count) + '.pth')
                        print('Model is saved in', './checkpoint/' + path + '_grade_debug_2Layers_' + str(count) + '.pth')
                        print('===========================','\n')
                        with open('./results/'+path+'/Confusion_matrix/OnlyClass3_2layers.txt', 'a') as f:
                            f.write(f'./checkpoint/'+path+'_grade_debug_2Layers_'+str(count)+'.pth \n'
                                    f'Parameter: lr={lr}, weight_decay={weight_decay}, weight={weight}, epoch={ep} \n'
                                    f'{child_con}\n'
                                    f'{result_int_child}\tmetric_value: {metric_value}\n'
                                    f'\n')
                        count += 1

                    break









# transtab.train_multi_task_only_class(clf, train_set, valid_set, **training_arguments)
# prob_int_two = torch.tensor(np.column_stack((1-prob_int, prob_int))).to('cuda:0')








# param_combinations = list(itertools.product(*param_grid.values()))
# def evaluate_params(params):
#     params_dict = dict(zip(param_grid.keys(), params))
#     training_arguments.update(params_dict)
#
#
#     model = transtab.build_classifier(cate_cols, num_cols, bin_cols, imb_weight=training_arguments['imb_weight'])
#     model = model.to('cuda')
#     transtab.train_multi_task(model, train_set, valid_set, **training_arguments)
#     print(params)
#
#     y_test = pd.concat([df_PHLF_valid_grade.iloc[:, 1], df_PHLF_valid_grade.iloc[:, 0]], axis=1)
#     prob_int_parent, prob_int_child = predict_multi_task(model, df_PHLF_valid_grade.iloc[:, 2:], y_test)
#
#     result_int_parent, parent_con = metrics_with_youden(df_PHLF_valid_grade.iloc[:, 1], prob_int_parent[:, 1])
#     print('\n')
#     result_int_child, child_con = metrics_multiclass(df_PHLF_valid_grade.iloc[:, 0], prob_int_child)
#
#     return result_int_parent, result_int_child,parent_con, child_con, model
#
# PATH = 'LF_bio_250220_best_0'
# best_params = None
# best_score = -float('inf')  # 我们是在最大化某个指标
# count = 0
# for params in param_combinations:
#     result_int_parent, result_int_child, parent_con, child_con,model = evaluate_params(params)
#     score = (float(result_int_parent[0])+float(result_int_child[0]))/2
#     if score > best_score:
#         best_score = score
#         best_params = params
#         best_model = model
#
#         model.save('./checkpoint/'+PATH + str(count))
#         count += 1
#         print('./checkpoint/'+PATH + str(count))
#         with open('./results/'+PATH+'/parameters.txt', 'a') as f:
#             f.write(str(best_params) + '\n' +
#                     str(parent_con) + '\n' +
#                     str(result_int_parent) + '\n'+
#                     str(child_con)+'\n'+
#                     str(result_int_child) + '\n' +
#                     '\n')
#         # torch.save(best_model, './checkpoint/LF_bio_240613_best.pth')
#         print(best_params, '\n',
#               '**************************************************************************************')
# print("Best parameters found:", best_params,'\n', "Best score:",best_score)

# model = transtab.build_classifier_multi_task(cate_cols, num_cols, bin_cols)
# model.load('/home/hci/QYang/YQ_LiverFailure/transtab-main/checkpoint/'+PATH)







#
#
# model.save('./checkpoint/'+PATH)
#
# best_params = None
# best_score = -float('inf')  # 我们是在最大化某个指标
# count = 0
# for params in param_combinations:
#     score, model, result_int, result_ext12, result_ext3 = evaluate_params(params)
#
#     if score > best_score:
#         best_score = score
#         best_params = params
#         best_model = model
#
#         model.save('./checkpoint/'+PATH + str(count))
#         count += 1
#         print('./checkpoint/'+PATH + str(count))
#         with open('./results/'+PATH+'.txt', 'a') as f:
#             f.write(str(best_params) + '\n' +
#                     str(result_int) + '\n' +
#                     str(result_ext12) + '\n'+
#                     str(result_ext3)+'\n'+'\n')
#         # torch.save(best_model, './checkpoint/LF_bio_240613_best.pth')
#         print(best_params, '\n',
#               '**************************************************************************************')
# print("Best parameters found:", best_params,'\n', "Best score:",best_score)


