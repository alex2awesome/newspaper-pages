import gzip
import re
import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
import pandas as pd
import os
import random
from collections import defaultdict
random.seed(0)

here = os.path.dirname(os.path.abspath(__file__))

SMALL_TRAIN_SIZE = 800

def preprocess_data(data_df, strip_punc):
    if strip_punc:
        data_df['processed_bodies'] = data_df['processed_bodies'].str.replace('\W+', ' ').str.lower().str.strip()
    return list(zip(
        data_df['processed_bodies'], 
        data_df['label'],
        data_df['label'].map({1:'A-1', 0:'not-A-1'}),
    ))


@RegisterDataset('nytimes_data')
class NYTimesDataset(AbstractDataset):
    def __init__(self, args, word_to_indx, name):
        self.args = args
        self.args.num_class = 20
        self.name = name
        self.dataset = []
        self.word_to_indx = word_to_indx
        self.max_length = int(args.word_cutoff) if args.word_cutoff != 'None' else None
        self.class_balance = defaultdict(int)

        if name in ['train', 'dev']:
            if args.training_data_path:
                fname = args.training_data
            else: ## local
                fname = os.path.join(here, '..', '..', '..', '..', '..', 'data', 'processed_train_time_balanced_df.csv')
            print('reading %s...' % fname)
            data = preprocess_data(pd.read_csv(fname), strip_punc=args.strip_punc)
            random.shuffle(data)
            num_train = int(len(data)*.9)
            if name != 'train':
                data = data[num_train:]
        else:
            if args.test_data_path:
                fname = args.test_data
            else:
                fname = os.path.join(here, '..', '..', '..', '..', '..', 'data', 'processed_test_time_unbalanced_df.csv')
            print('reading %s...' % fname)
            data = preprocess_data(pd.read_csv(fname), strip_punc=args.strip_punc)

        self.max_length = self.max_length if (self.max_length != None) else max_len
        for indx, _sample in tqdm.tqdm(enumerate(data)):
            sample = self.processLine(_sample)
            self.class_balance[ sample['y'] ] += 1
            self.dataset.append(sample)

        print ("Class balance", self.class_balance)
        if args.class_balance:
            raise NotImplementedError("NYTimes dataset doesn't support balanced sampling")
        if args.objective == 'mse':
            raise NotImplementedError("NYTimes dataset does not support Regression objective")

    def processLine(self, row):
        text, label, label_name = row
        text = " ".join(text.split()[:self.max_length])
        x =  get_indices_tensor(text.split(), self.word_to_indx, self.max_length)
        sample = {'text':text, 'x':x, 'y':label, 'y_name': label_name}
        return sample