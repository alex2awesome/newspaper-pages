from collections import defaultdict
from tqdm import tqdm
import json
import pandas as pd 

bert_data = pd.read_csv('../data/bert-data/train.tsv', sep='\t', nrows=50000)
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
            label = bert_data.loc[label_idx, 'label']       
            layer = int(sent_idx / 12)  % 12

json.dump(attention, open('roberta-attention-grouped.json'))