import string
import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataset import *

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None):

        if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen)

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()


    # --------- Create indexs from training data
    def __create_indexs(self, data, maxlen, suflen) :
        self.maxlen = maxlen
        self.suflen = suflen
        self.preflen = suflen

        words = set([])
        lc_words = set([])
        sufs = set([])
        prefs = set([])
        labels = set([])

        for s in data.sentences() :
            for t in s :
                words.add(t['form'])
                lc_words.add(t['lc_form'])
                sufs.add(t['lc_form'][-self.suflen:])
                prefs.add(t['lc_form'][:self.preflen])
                labels.add(t['tag'])

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0
        self.word_index['UNK'] = 1

        self.lc_word_index = {w: i+2 for i,w in enumerate(list(lc_words))}
        self.lc_word_index['PAD'] = 0
        self.lc_word_index['UNK'] = 1

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0
        self.suf_index['UNK'] = 1

        self.pref_index = {p: i+2 for i,p in enumerate(list(prefs))}
        self.pref_index['PAD'] = 0
        self.pref_index['UNK'] = 1

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0

    ## --------- load indexs -----------
    def __load(self, name) :
        self.maxlen = 0
        self.suflen = 0
        self.preflen = 0
        self.word_index = {}
        self.lc_word_index = {}
        self.suf_index = {}
        self.pref_index = {}
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines():
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)
                elif t == 'PREFLEN' : self.preflen = int(k)
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'LCWORD': self.lc_word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'PREF': self.pref_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)

    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        with open(name+".idx","w") as f :
            print('MAXLEN', self.maxlen, "-", file=f)
            print('SUFLEN', self.suflen, "-", file=f)
            print('PREFLEN', self.preflen, "-", file=f)
            for key in self.label_index :
                print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index :
                print('WORD', key, self.word_index[key], file=f)
            for key in self.lc_word_index :
                print('LCWORD', key, self.lc_word_index[key], file=f)
            for key in self.suf_index :
                print('SUF', key, self.suf_index[key], file=f)
            for key in self.pref_index :
                print('PREF', key, self.pref_index[key], file=f)


    ## --------- encode X from given data -----------
    def encode_words(self, data) :
        # encode and pad sentence words
        Xw = [[self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s] for s in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index['PAD'])

        # encode and pad lowercase words
        Xlw = [[self.lc_word_index[w['lc_form']] if w['lc_form'] in self.lc_word_index else self.lc_word_index['UNK'] for w in s] for s in data.sentences()]
        Xlw = pad_sequences(maxlen=self.maxlen, sequences=Xlw, padding="post", value=self.lc_word_index['PAD'])

        # encode and pad suffixes
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s] for s in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index['PAD'])

        # encode and pad prefixes
        Xp = [[self.pref_index[w['lc_form'][:self.preflen]] if w['lc_form'][:self.preflen] in self.pref_index else self.pref_index['UNK'] for w in s] for s in data.sentences()]
        Xp = pad_sequences(maxlen=self.maxlen, sequences=Xp, padding="post", value=self.pref_index['PAD'])

        # orthographic features:
        # [is_capitalized, is_all_caps, is_all_lower, has_digit,
        #  has_hyphen, has_slash, has_dot, is_alnum_mixed]
        Xf = []
        for s in data.sentences():
            sent_feats = []
            for w in s:
                form = w['form']
                is_capitalized = 1 if len(form) > 0 and form[0].isupper() else 0
                is_all_caps = 1 if len(form) > 0 and form.isupper() else 0
                is_all_lower = 1 if len(form) > 0 and form.islower() else 0
                has_digit = 1 if any(ch.isdigit() for ch in form) else 0
                has_hyphen = 1 if '-' in form else 0
                has_slash = 1 if '/' in form else 0
                has_dot = 1 if '.' in form else 0
                has_alpha = any(ch.isalpha() for ch in form)
                is_alnum_mixed = 1 if has_alpha and has_digit else 0

                sent_feats.append([
                    is_capitalized,
                    is_all_caps,
                    is_all_lower,
                    has_digit,
                    has_hyphen,
                    has_slash,
                    has_dot,
                    is_alnum_mixed
                ])
            Xf.append(sent_feats)

        padded_Xf = []
        for sent_feats in Xf:
            if len(sent_feats) < self.maxlen:
                sent_feats = sent_feats + [[0]*8] * (self.maxlen - len(sent_feats))
            else:
                sent_feats = sent_feats[:self.maxlen]
            padded_Xf.append(sent_feats)
        Xf = np.array(padded_Xf, dtype=np.float32)

        return [Xw, Xs, Xlw, Xp, Xf]


    ## --------- encode Y from given data -----------
    def encode_labels(self, data) :
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- getters ---------
    def get_n_words(self) :
        return len(self.word_index)

    def get_n_lc_words(self) :
        return len(self.lc_word_index)

    def get_n_sufs(self) :
        return len(self.suf_index)

    def get_n_prefs(self) :
        return len(self.pref_index)

    def get_n_labels(self) :
        return len(self.label_index)

    def get_n_chars(self) :
        return len(self.char_index)

    def word2idx(self, w) :
        return self.word_index[w]

    def suff2idx(self, s) :
        return self.suf_index[s]

    def label2idx(self, l) :
        return self.label_index[l]

    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError