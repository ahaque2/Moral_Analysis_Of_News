import pandas as pd
import numpy as np
import sys

import torch
from transformers import AdamW, AutoModel, AutoTokenizer

from sklearn.decomposition import PCA
from Word_Pairs import Word_Pairs
import warnings
warnings.filterwarnings('ignore')

import argparse

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--high', type=int, default=-1,  help='Count of virtue words')
parser.add_argument('--low', type=int, default=-1, help='Count of vice words')
parser.add_argument('--num_keep', type=int, default=-1, help='Count of words pairs to keep')
parser.add_argument('--moral_foundation', type=str, default='care', help='Moral Foundation')
parser.add_argument('--subspace_dim', type=int, default=1, help='Subspace Dimensions (set to 1 to get moral foundation scores or higher to get moral foundation embeddings with dim dimension)')

args = parser.parse_args()

best_params = {'care': (275, 475, 175), 
          'fairness': (350, 475, 325), 
          'loyalty': (300, 375, 175), 
          'authority': (350, 475, 275), 
          'sanctity': (275, 375, 225)}


def get_word_embeddings(wordlist):

    embeddings = []
    model.eval()
    for word in wordlist:
    
        tokenized_inputs = tokenizer(word, return_tensors='pt', padding=True, truncation=True).to(device)

        # Get BERT model outputs
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            
        # print(outputs.keys())

        # print("Here ", outputs.keys())
        hs = outputs.hidden_states
        emb = torch.stack([x.mean(axis = 1) for x in hs]).mean(axis = 0).detach().cpu().numpy()

        embeddings.append(np.squeeze(emb))
         
    return np.array(embeddings)


def get_sub(high, low, k, moral_foundation, dim):

    word_pairs = WordPairs.get_word_pairs_with_scores(high, low, k, moral_foundation)

    w1_emb = get_word_embeddings(word_pairs.Word1.tolist())
    w2_emb = get_word_embeddings(word_pairs.Word2.tolist())

    subspace_1, subspace_dim = WordPairs.get_subspace(w1_emb, w2_emb, dim)
    
    return subspace_1, subspace_dim
    


emfd_df = pd.read_csv('../data/eMFD_wordlist.csv')

model_name = 'bert-base-cased'
model = AutoModel.from_pretrained(model_name, output_hidden_states = True, output_attentions = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ModernBertModel
tokenizer = AutoTokenizer.from_pretrained(model_name)

all_words = emfd_df.word.tolist()

emb = get_word_embeddings(all_words)

emb_dict = dict()
for word, e in zip(all_words, emb):
    emb_dict[word] = e
    
    
WordPairs = Word_Pairs(model, emb_dict, emfd_df)

moral_foundation = args.moral_foundation
dim = args.subspace_dim
high, low, k = args.high, args.low, args.num_keep
    
if high == -1 or low == -1 or k == -1:
    print("Using default values")
    high, low, k = best_params[moral_foundation]
    print(f'high = {high} \t low = {low} \t k = {k}')
    

subspace, _ = get_sub(high, low, k, moral_foundation + '_sent', dim)

np.save(f'subspace/{moral_foundation}.npy', subspace)

print(f'Successfully Identified the Subspace for {moral_foundation}. \nSaved it under subspace/')