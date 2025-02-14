import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr

import torch
from transformers import AdamW, AutoModel, AutoTokenizer

from sklearn.decomposition import PCA
from Word_Pairs import Word_Pairs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--moral_foundation', type=str, default='care', help='Moral Foundation')

args = parser.parse_args()

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


def get_subspace(high, low, k, attr, dim):

    word_pairs = WordPairs.get_word_pairs_with_scores(high, low, k, attr)

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

moral_foundation = args.moral_foundation
try:
    subspace = np.load(f'subspace/{moral_foundation}.npy')
    
except Exception as e:
    print('Subspace not found under the subspace directory. First run 1_identify_moral_subspace.py to create the subspace.')
    sys.exit(0)

scores = np.dot(emb, subspace)

true_scores = emfd_df[f'{moral_foundation}_sent'].tolist()

corr, _ = pearsonr(scores, true_scores)
print('Pearson Correlation ', corr)