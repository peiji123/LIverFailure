import shap
from transformers import BertTokenizerFast
import transtab
import pandas as pd
import json


def shap_predict(data):
    return transtab.predict_fun(model, data, X_test2.columns)
def X_y_split(df, lal):
    X = df.drop([lal], axis=1)
    y = df[lal]
    return X, y

df_test2 = pd.read_csv('../data_process/TransTab_dataset/Time2_PHLF_DropLiver_0508_test.csv')
info = json.load(open('../data_process/DropLiverseg_info.json'))

df_test2 = df_test2.iloc[:,1:]
X_test2, y_test2 = X_y_split(df_test2, info['target'][0])
x_test_clean = X_test2.dropna()
data = x_test_clean.iloc[:100,:]

model = transtab.build_classifier(checkpoint='./checkpoint/LF_bio_Non_240517_0',device='cuda:0')
tokenizer = BertTokenizerFast.from_pretrained("./transtab/Bio_tokenizer")
# explainer = shap.KernelExplainer(model=shap_predict, data=data)
explainer = shap.Explainer(shap_predict, tokenizer, output_names=[0, 1])
print(explainer)
shap_values = explainer(data.all())
print(shap_values)
