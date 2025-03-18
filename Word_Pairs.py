import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth' , None)
from torch import nn
import torch
from transformers import AdamW, AutoModel, AutoTokenizer, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import operator
from scipy.spatial.distance import cosine
import scipy.stats


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Word_Pairs:
    
    def __init__(self, model, emb_dict, emfd_df):
        
        self.model = model
        self.emb_dict = emb_dict
        self.emfd_df = emfd_df

    def pair_words(self, words1, words2, array1, array2, num_keep, pairs_flag = 'contrast'):
        sim_matrix = cosine_similarity(array1, array2)
        words_high, words_low = [], []

        while sim_matrix.size > 0 and len(words_high) < num_keep:
            i, j = 0, 0
            if pairs_flag == 'contrast':
                i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)

            elif pairs_flag == 'similar':
                 i, j = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
            words_high.append(words1[i])
            words_low.append(words2[j])
            sim_matrix = np.delete(sim_matrix, i, axis=0)  # Remove the selected row
            sim_matrix = np.delete(sim_matrix, j, axis=1)  # Remove the selected column
            words1.pop(i)
            words2.pop(j)

        return words_high, words_low


    def load_emfd_scores(self, df, header):

        df = df.dropna(subset = ['word'])
        word_to_score = {}
        # print(header)
        # print(df)
        
        for i,row in df.iterrows():
            # print(row, row["word"], row[header])
            word_to_score[row["word"]] = row[header]
        return word_to_score


    def filter_word_pairs(self, high_words, low_words, words_hs_vec, words_ls_vec, num_keep):

        pair_to_dist = {}

        for i, h in enumerate(high_words):
            for j, l in enumerate(low_words):

                h_embed = words_hs_vec[i]
                l_embed = words_ls_vec[j]

                pair_to_dist[(h, l)] = 1-cosine(l_embed, h_embed)

        # return pair_to_dist

        final_low = []
        final_high = []
        used_words = set()

        for best_pair in sorted(pair_to_dist.items(), key=operator.itemgetter(1), reverse = True):

            w1 = best_pair[0][0]
            w2 = best_pair[0][1]
            if w1 in used_words or w2 in used_words:
                continue
            final_high.append(w1)
            final_low.append(w2)
            used_words.add(w1)
            used_words.add(w2)
            if len(final_high) >= num_keep:
                break

        return final_high, final_low

    def get_word_pairs(self, low_start, high_start, num_keep, attr):

        attr_so_score = self.load_emfd_scores(self.emfd_df, attr)

        low = sorted(attr_so_score.items(), key=operator.itemgetter(1))[:low_start]
        high = sorted(attr_so_score.items(), key=operator.itemgetter(1))[-high_start:]

        low_words = [l[0] for l in low]
        high_words = [l[0] for l in high]

        words_ls_vec = np.array([self.emb_dict[w] for w in low_words])
        words_hs_vec = np.array([self.emb_dict[w] for w in high_words])

        # return filter_word_pairs(high_words, low_words, words_hs_vec, words_ls_vec, num_keep)
        return self.pair_words(high_words, low_words, words_hs_vec, words_ls_vec, num_keep)


    def get_scores(self, words_list, attr):

        score = []
        for word in words_list:
            scr = self.emfd_df[self.emfd_df.word == word][attr]
            score.append(scr.values[0])

        return score

    def get_cos_sim(self, word_pairs_df):

        wordlist1 = word_pairs_df.Word1.tolist()
        wordlist2 = word_pairs_df.Word2.tolist()

        w1_vec = np.array([self.emb_dict[w] for w in wordlist1])
        w2_vec = np.array([self.emb_dict[w] for w in wordlist2])

        cos_sim = []

        for w1, w2 in zip (w1_vec, w2_vec):

            cos = 1-cosine(w1, w2)
            cos_sim.append(cos)

        return cos_sim

    def get_word_pairs_with_scores(self, high, low, num_keep, attr):

        high_words, low_words = self.get_word_pairs(low, high, num_keep, attr)

        word_pairs_df = pd.DataFrame()

        word_pairs_df['Word1'] = high_words
        word_pairs_df['Word2'] = low_words
        word_pairs_df['w1_score'] = self.get_scores(high_words, attr)
        word_pairs_df['w2_score'] = self.get_scores(low_words, attr)

        word_pairs_df['cos_sim'] = self.get_cos_sim(word_pairs_df)

        return word_pairs_df

    def get_subspace(self, w1_vec, w2_vec, dim):

        mu = (np.array(w1_vec) + np.array(w2_vec))/2
        w1_vec_norm = w1_vec - mu
        w2_vec_norm = w2_vec - mu
        M = np.concatenate((w1_vec_norm, w2_vec_norm), axis = 0)

        pca = PCA(n_components=dim, random_state=0)
        subspace = pca.fit_transform(M.T)[:,:dim]

        return np.array([x[0] for x in subspace]), subspace