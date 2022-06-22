# -*- coding: utf-8 -*-
# from coherence import coherenceAnalisys
# -*- coding: utf-8 -*-
# This module implements the algorithm used in "Automated analysis of
# free speech predicts psychosis onset in high-risk youths"
# http://www.nature.com/articles/npjschz201530

import json
import sys
import numpy as np
import scipy
import os
import importlib.resources


class LsaWrapper():
    def __init__(self, corpus='en_tasa'):
        package_path = importlib.resources.files(__name__)
        path = {"en_tasa": "models/tasa_150"}
        path_models = path[corpus]
        dictionary_package_path = os.path.join(path_models, 'dictionary.json')
        dictionary_path = os.path.join(package_path, dictionary_package_path)
        with open(dictionary_path) as f:
            dic_word2index = json.load(f)

        self.dic_word2index = dict(zip(dic_word2index,
                                       range(len(dic_word2index)))
                                   )
        self.dic_index2word = dict(zip(range(len(dic_word2index)),
                                       dic_word2index))
        matrix_package_path = os.path.join(path_models, 'matrix.npy')
        matrix_path = os.path.join(package_path, matrix_package_path)
        self.u = np.load(matrix_path)

    def get_vector(self, word, normalized=False, size=150):
        try:
            return self.u[self.dic_word2index[word], :][:int(size)]
        except:
            return None 

    def index2word(self, i):
        try:
            return self.dic_index2word[i]
        except:
            return None

    def word2index(self, w):
        try:
            return self.dic_word2index[w]
        except:
            return None

    def _unitvec(self, v):
        return v/np.linalg.norm(v)

    def similarity(self, word1, word2, size=150):
        return np.dot(self._unitvec(self.get_vector(word1)),
                      self._unitvec(self.get_vector(word2)))


class CoherenceAnalysis():

    def __init__(self, corpus='en_tasa', dims=150, 
                 word_tokenizer=lambda x: x.split(' '), 
                 sentence_tokenizer=lambda txt: txt.split('.')):
        self.corpus = LsaWrapper(corpus=corpus)
        self.word_tokenizer = word_tokenizer
        self.sentence_tokenizer = sentence_tokenizer

    def _unitvec(self, v):
        return v/np.linalg.norm(v)

    def text_analysis(self, text, max_order=10):
        sentences = self.sentence_tokenizer(text)
        vectorized_sentences = [
            [
                self.corpus.get_vector(w.lower()) 
                for w in self.word_tokenizer(s) 
            ] 
            for s in sentences
        ]
        non_vector_count = len([x 
                                for s in vectorized_sentences
                                for x in s
                                    if x is None
                               ])
        total_word_count = len([x 
                                for s in vectorized_sentences
                                for x in s
                               ])
        sentence_count = len(sentences)
        vectorized_sentences = [
            [v for v in s if v is not None]
            for s in vectorized_sentences
        ]
        mean_and_len = [(np.mean(vec_sent, 0), len(vec_sent))
                        for vec_sent in vectorized_sentences]
        try:
            mean_vectors_series, len_words_per_vectors = zip(
                *[t for t in mean_and_len if t[1] > 0])
        except:
            return {}
        
        m = np.array(list(map(self._unitvec, mean_vectors_series)))
        max_order = min(m.shape[0], max_order)
        similarity_matrix = np.dot(m, m.T)
        similarity_orders = [np.diag(similarity_matrix, i)
                             for i in range(1, max_order)]
        similarity_metrics = {
            'order_'+str(i): self._get_statistics(s)
            for i, s in enumerate(similarity_orders)
        }

        normalized_coeff = [
            list(map(np.mean, zip(len_words_per_vectors[:-i], 
                                  len_words_per_vectors[i:]))) 
            for i in range(1, max_order)
        ]
        similarity_orders_normalized = [
            s / np.array(coeff_list) 
            for s, coeff_list in zip(similarity_orders, normalized_coeff)
        ]
        similarity_metrics_normalized = {
            'normalized_order_'+str(i): self._get_statistics(s) 
            for i, s in enumerate(similarity_orders_normalized)
        }

        similarity_metrics.update(similarity_metrics_normalized)
        similarity_metrics.update({
            'vector_serie_'+str(i): s 
            for i, s in enumerate(similarity_orders)
        })

        similarity_metrics['non_vector_count'] = non_vector_count
        similarity_metrics['word_count'] = total_word_count
        similarity_metrics['sentence_count'] = sentence_count

        return similarity_metrics

    def _get_statistics(self, s):
        res = {'mean': np.mean(s), 'std': np.std(
            s), 'min': np.min(s), 'max': np.max(s)}
        for i in range(0, 110, 10):
            res['percentile_'+str(i)] = np.percentile(s, i)
        return res
