import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
from adjustText import adjust_text
from holoviews.plotting.bokeh.styles import font_size

file_path = 'LF_bio_240625_best_018'
similarity_matrix = pd.read_csv('/home/hci/QYang/YQ_LiverFailure/transtab-main/results/'+file_path+'/connection/all_cor_values.csv')
similarity_matrix  =similarity_matrix.set_index('Unnamed: 0')
dict_simple2form_col = json.load(open('../data_process/simple2form_cols.json'))
similarity_matrix = similarity_matrix.rename(columns=dict_simple2form_col, index=dict_simple2form_col)


threshold_pos = 0.8
threshold_neg = -0.25

G = nx.Graph()

for i, row in similarity_matrix.iterrows():
    for j, value in row.iteritems():
        if i != j and (value > threshold_pos or value < threshold_neg):
        # if i != j:
            G.add_edge(i, j, weight=value)
'''
边属性设置
    边粗细 -> weight value
    边颜色 -> weight pos/neg
'''
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
edge_colors = ['blue' if weight<0 else 'red' for weight in weights]
edge_widths = [3 * abs(weight) for weight in weights]
# color map 获得颜色映射
pos_cmap = plt.cm.Blues
neg_cmap = plt.cm.Greens
edge_colors = [pos_cmap(weight) if weight>0 else neg_cmap(-weight) for weight in weights]

'''
节点设置
    节点大小 <- 所有链接权重绝对值和
'''
node_weight_sum = {}
for node in G.nodes():
    total_weight = sum(abs(data['weight']) for _, _, data in G.edges(node, data=True))
    node_weight_sum[node] = total_weight
node_size = [node_weight_sum[node]*200 for node in G.nodes()]

'''
label
    1. 重命名
    2. 位置和大小进行调整
'''
pos = nx.spring_layout(G, k=0.8, iterations=50)
# pos = nx.kamada_kawai_layout(G)
label_pos = {key:(value[0], value[1]+0.05) for key, value in pos.items()}


plt.subplots(figsize=(20, 16))


nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size, edgecolors='royalblue',linewidths=4)
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths, alpha=0.5)

texts = []
for node, (x, y) in label_pos.items():
    texts.append(plt.text(x, y, node, ha='center',va='bottom',fontsize=18))
adjust_text(texts)
# nx.draw_networkx_labels(G, label_pos, font_size=17, verticalalignment='bottom')
plt.savefig('./results/'+file_path+'/connection/GNN.png')
plt.show()



