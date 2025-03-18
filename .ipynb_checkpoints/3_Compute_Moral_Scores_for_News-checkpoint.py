import pandas as pd
import numpy as np
import sys

from transformers import AutoModel, AutoTokenizer
import torch


import argparse

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--moral_foundation', type=str, default='care', help='Moral Foundation')
args = parser.parse_args()

moral_foundation = args.moral_foundation
df = pd.read_csv('data/GazaNews.csv')


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

def normalize(data):
    
    min_val, max_val = min(data), max(data)
    normalized = [(2 * (x - min_val) / (max_val - min_val)) - 1 for x in data]
    return normalized


model_name = 'bert-base-cased'
model = AutoModel.from_pretrained(model_name, output_hidden_states = True, output_attentions = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ModernBertModel
tokenizer = AutoTokenizer.from_pretrained(model_name)

subspace = np.load(f'subspace/{moral_foundation}.npy')
emb = get_word_embeddings(df.headline.tolist())

## Multiple the subspace with -1 if you get a negative Pearson correlation in 2_Model_Evaluation.py
score = np.dot(emb, subspace)

score_norm = normalize(score)
df[moral_foundation] = score_norm

df.to_csv('results/GazaNews_with_moral_scores.csv')
print('Save the results to results/GazaNews_with_moral_scores.csv')