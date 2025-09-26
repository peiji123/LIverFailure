import pandas as pd
from tableone import TableOne

df_ext = pd.read_excel('../data_process/10-24-External_data.xlsx')
table = TableOne(df_ext)

groupby = 'PHLF'
cate_cols_list = ['Gender','Hypertension', 'Diabetes', 'Cirrhosis', 'Fatty_liver',
       'Portal_hypertension', 'Ascites','HBVs Ag', 'HCV', 'Methods', 'Major', 'S_Number', 'ALR',
                  'Tumor_number']
cont_cols_list = ['Age', 'BMI','AFP', 'Tumor_size','E_PT',
       'E_PT-INR', 'E_WBC', 'E_LY', 'E_NE', 'E_RBC', 'E_HGB', 'E_PLT', 'E_ALT',
       'E_AST', 'E_TP', 'E_ALB', 'E_TBIL', 'E_GGT', 'E_TBA', 'E_CR', 'E_K',
       'E_Na','Operation_time',
       'Bleeding', 'Pringle', 'Transfusion_1', 'D1_PT', 'D1_PT-INR', 'D1_WBC',
       'D1_LY', 'D1_NE', 'D1_RBC', 'D1_HGB', 'D1_PLT', 'D1_ALT', 'D1_AST',
       'D1_TP', 'D1_ALB', 'D1_TBIL', 'D1_GGT', 'D1_TBA', 'D1_CR', 'D1_K',
       'D1_Na',]
missing_categorical = set(cate_cols_list) - set(df_ext.columns)
missing_continuous = set(cont_cols_list) - set(df_ext.columns)

print("Missing categorical columns:", missing_categorical)
print("Missing continuous columns:", missing_continuous)
mytable = TableOne(df_ext, cate_cols_list, cont_cols_list, groupby)