import zipfile
import collections
import numpy as np
import random

class Data:
    def __init__(self, data_dir="data/", fname="text8", window_size=5, 
                 min_occurrences=0, subsample=False, t=1e-3, cutoff=None):
        self.window_size = window_size
        raw_data = self.read_data("%s%s.zip" % (data_dir, fname))
        self.cutoff = cutoff if cutoff else len(raw_data)
        raw_data = raw_data[:cutoff]
        self.data, self.word_counts, self.word_to_idx, self.idx_to_word = \
                self.build_dataset(raw_data, min_occurrences=min_occurrences, 
                subsample=subsample, t=t)
        del raw_data
        self.cooccurrence_counts, self.w_probs, self.c_probs = \
                    self.generate_statistics(window_size=window_size)
        print("Number of data points: %d" %len(self.data))
        print("Vocabulary size: %d" %len(self.word_counts))


    def read_data(self, filename):
        """
        Extract the first file enclosed in a zip file as a list of words.
        """
        with zipfile.ZipFile(filename) as f:
            raw_data = f.read(f.namelist()[0]).split()
        return raw_data

    def build_dataset(self, words, min_occurrences=0, subsample=False, t=1e-3):
        """
        Process raw inputs into a dataset.
        """
        counts = collections.Counter(words)
        counts = {k:v for k,v in counts.items() if v >= min_occurrences}
        words = [w for w in words if w in counts]
        if subsample:
            sum_counts = float(sum(counts.values()))
            freqs = {k:v/sum_counts for k, v in counts.items()}
            kept_words = []
            for w in words:
                p = np.random.uniform()
                if p > (1-np.sqrt(t/freqs[w])):
                    kept_words.append(w)
            words = kept_words
            counts = collections.Counter(words)
            counts = {k:v for k,v in counts.items() if v >= min_occurrences}
            words = [w for w in words if w in counts]
        word_to_idx = {}
        for word, _ in counts.items():
            word_to_idx[word] = len(word_to_idx)
        data = [word_to_idx[word] for word in words]
        idx_to_word = dict(zip(word_to_idx.values(), word_to_idx.keys()))
        return data, counts, word_to_idx, idx_to_word

    def generate_statistics(self, window_size=5):
        """
        Generate the dataset statistics, i.e. joint and marginal probabilities.
        """
        cooccurrence_counts = collections.defaultdict(int)
        w_probs = collections.defaultdict(int)
        c_probs = collections.defaultdict(int)

        num_pairs = 0.
        for i in range(0, len(self.data)):
            word = self.data[i]
            context_words = self.data[max(i-window_size, 0):i] + \
                            self.data[i+1:min(i+1+window_size, len(self.data))]
            pos_weights = np.concatenate([np.arange(1., window_size+1), 
                          np.arange(window_size, 0., -1.)])/float(window_size)
            for ci, c_word in enumerate(context_words):
                num_pairs += pos_weights[ci]
                cooccurrence_counts[(word, c_word)] += pos_weights[ci]
                w_probs[word] += pos_weights[ci]
                c_probs[c_word] += pos_weights[ci]
        cooccurrence_counts = {k:v/num_pairs for k, v in 
                               cooccurrence_counts.items()}
        normalizer_w = sum([wp**0.75 for wp in w_probs.values()])
        w_probs = {k:(v**0.75)/normalizer_w for k, v in w_probs.items()}
        normalizer_c = sum([cp**0.75 for cp in c_probs.values()])
        c_probs = {k:(v**0.75)/normalizer_c for k, v in c_probs.items()}
        return cooccurrence_counts, w_probs, c_probs
