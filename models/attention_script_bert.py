import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm as tqdm
import pandas as pd

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## read data
print('reading data...')
nyt_data = pd.read_csv('data/bert-data/train.tsv', sep='\t', nrows=50000)
## 
model_version = 'bert-base-uncased'
model = 'pytorch-transformers/examples/bert-runs'
do_lower_case = True

## load model
print('loading model...')
model = BertModel.from_pretrained(model, output_attentions=True)
model.eval()
model.to(device)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
body_text = nyt_data['processed_bodies'].apply(lambda x: x[:300])

## score
print('iterating...')
all_attention = []
for sentence in tqdm(body_text):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True).to(args.device)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    converted_text = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    ## 
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    last_layer_attention = attention[-1][0]
    ## 
    for head in last_layer_attention:
        head_attention = list(zip(converted_text, head[0].tolist()))#, key=lambda x: -x[1]
        all_attention.append(head_attention)

print('writing...')
with open('attention-output.txt', 'w') as f:
    for head_attention in tqdm(all_attention):
        for word, attn_score in head_attention:
            f.write('%s\t%s' % (word, attn_score))
        f.write('\n')
