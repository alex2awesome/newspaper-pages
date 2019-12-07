from collections import defaultdict
from tqdm import tqdm
import json
import pandas as pd 
import numpy as np

bert_data = pd.read_csv('../data/bert-data/train.tsv', sep='\t')
attention = {
    0: {i: defaultdict(list) for i in range(12)},
    1: {i: defaultdict(list) for i in range(12)}
}

sent_idx = 0
label_idx = int(sent_idx / (12 * 12))
label = bert_data.loc[label_idx, 'label']       
layer = int(sent_idx / 12)  % 12
with open('attention-output-roberta.txt', encoding='utf-8') as f:
    for line in tqdm(f, total=462549168):
        line = line.strip()
        if line != '':
            word, attn = line.split('\t')
            attn = float(attn)
            attention[label][layer][word].append(attn)
        else:
            sent_idx += 1
            label_idx = int(sent_idx / (12 * 12))
            label = bert_data.iloc[label_idx, 'label']       
            layer = int(sent_idx / 12)  % 12


for label in [0, 1]:
    print('label: %s' % label)
    for layer in tqdm(range(12)):
        layer_words = attention[label][layer]
        mean_var = {}
        for word, attn_list in layer_words.items():
            mean_var[word] = [np.mean(attn_list), np.var(attn_list)]
        with open('roberta-attention-grouped__label-%s_layer-%s.json' % (label, layer), 'w') as f:
            json.dump(mean_var, f)