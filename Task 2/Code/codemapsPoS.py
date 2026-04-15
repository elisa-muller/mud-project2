import string
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dataset import *
import spacy
nlp = spacy.load("en_core_web_sm")

class Codemaps :
    # Changed arguments to max_len and suf_len to match your notebook call
    def __init__(self, data, max_len=None, suf_len=None, drugbank_path=None):
        self.drug_vocab = set()
        self.drugbank_path = drugbank_path
        
        if drugbank_path:
            self._load_drugbank()

        # Matches the logic in your original working version
        if isinstance(data, Dataset) and max_len is not None and suf_len is not None:
            self._create_indexs(data, max_len, suf_len)
        elif type(data) == str and max_len is None and suf_len is None:
            self._load(data)
        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

    # Drug vocabulary loading
    def _load_drugbank(self):
        try:
            with open(self.drugbank_path, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip().split('|')[0]
                    self.drug_vocab.add(name.lower())
            print(f"Loaded {len(self.drug_vocab)} entries from DrugBank.")
        except Exception as e:
            print(f"Warning: Could not load DrugBank file: {e}")

    # PoS tagging method using spaCy
    def _get_pos_tags(self, sentence):
        ## Reconstruct sentence text from token forms and tag with spaCy.
        ## Returns a list of PoS strings aligned to the input token list.
        text = " ".join([t['form'] for t in sentence])
        spacy_doc = nlp(text)
        pos_tags = [token.pos_ for token in spacy_doc]

        if len(pos_tags) != len(sentence):
            return ['NOUN'] * len(sentence)

        return pos_tags
    
    # Removed double underscores to prevent name mangling errors
    def _create_indexs(self, data, maxlen, suflen) :
        self.maxlen = maxlen
        self.suflen = suflen
        self.preflen = suflen
        self.maxcharlen = 20

        words, lc_words, sufs, prefs, labels, chars, pos_tags = set(), set(), set(), set(), set(), set(), set()

        for s in data.sentences():
            pos_tags_s = self._get_pos_tags(s) # tag whole sentence
            for t, pos in zip(s, pos_tags_s):
                words.add(t['form'])
                lc_words.add(t['lc_form'])
                sufs.add(t['lc_form'][-self.suflen:])
                prefs.add(t['lc_form'][:self.preflen])
                labels.add(t['tag'])
                pos_tags.add(pos)                     
                for c in t['lc_form']:
                    chars.add(c)

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index.update({'PAD': 0, 'UNK': 1})
        self.lc_word_index = {w: i+2 for i,w in enumerate(list(lc_words))}
        self.lc_word_index.update({'PAD': 0, 'UNK': 1})
        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index.update({'PAD': 0, 'UNK': 1})
        self.pref_index = {p: i+2 for i,p in enumerate(list(prefs))}
        self.pref_index.update({'PAD': 0, 'UNK': 1})
        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0
        self.char_index = {c: i+2 for i,c in enumerate(list(chars))}
        self.char_index.update({'PAD': 0, 'UNK': 1})
        self.pos_index = {p: i+2 for i,p in enumerate(list(pos_tags))}  
        self.pos_index.update({'PAD': 0, 'UNK': 1}) 

    def encode_words(self, data) :
        Xw = pad_sequences(maxlen=self.maxlen, padding="post", value=0,
            sequences=[[self.word_index.get(w['form'], 1) for w in s] for s in data.sentences()])
        Xlw = pad_sequences(maxlen=self.maxlen, padding="post", value=0,
            sequences=[[self.lc_word_index.get(w['lc_form'], 1) for w in s] for s in data.sentences()])
        Xs = pad_sequences(maxlen=self.maxlen, padding="post", value=0,
            sequences=[[self.suf_index.get(w['lc_form'][-self.suflen:], 1) for w in s] for s in data.sentences()])
        Xp = pad_sequences(maxlen=self.maxlen, padding="post", value=0,
            sequences=[[self.pref_index.get(w['lc_form'][:self.preflen], 1) for w in s] for s in data.sentences()])
        Xpos = pad_sequences(maxlen=self.maxlen, padding="post", value=0,
            sequences=[[self.pos_index.get(pos, 1) for pos in self._get_pos_tags(s)]
                       for s in data.sentences()])

        Xf = []
        for s in data.sentences():
            sent_feats = []
            for w in s:
                form, lc_form = w['form'], w['lc_form']
                feats = [
                    1 if len(form) > 0 and form[0].isupper() else 0,
                    1 if form.isupper() else 0,
                    1 if form.islower() else 0,
                    1 if any(ch.isdigit() for ch in form) else 0,
                    1 if '-' in form else 0,
                    1 if '/' in form else 0,
                    1 if '.' in form else 0,
                    1 if any(ch.isalpha() for ch in form) and any(ch.isdigit() for ch in form) else 0,
                    1 if lc_form in self.drug_vocab else 0
                ]
                sent_feats.append(feats)
            
            if len(sent_feats) < self.maxlen:
                sent_feats += [[0]*9] * (self.maxlen - len(sent_feats))
            Xf.append(sent_feats[:self.maxlen])
        Xf = np.array(Xf, dtype=np.float32)

        Xc = []
        for s in data.sentences():
            sent_chars = []
            for w in s:
                char_ids = [self.char_index.get(c, 1) for c in w['lc_form'][:self.maxcharlen]]
                char_ids += [0] * (self.maxcharlen - len(char_ids))
                sent_chars.append(char_ids)
            if len(sent_chars) < self.maxlen:
                sent_chars += [[0]*self.maxcharlen] * (self.maxlen - len(sent_chars))
            Xc.append(sent_chars[:self.maxlen])
        Xc = np.array(Xc, dtype=np.int32)
 
        return [Xw, Xs, Xlw, Xp, Xpos, Xf, Xc]

    def encode_labels(self, data) :
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    def get_n_words(self) : return len(self.word_index)
    def get_n_lc_words(self) : return len(self.lc_word_index)
    def get_n_sufs(self) : return len(self.suf_index)
    def get_n_prefs(self) : return len(self.pref_index)
    def get_n_labels(self) : return len(self.label_index)
    def get_n_chars(self) : return len(self.char_index)
    def get_n_pos(self) : return len(self.pos_index)
    def word2idx(self, w) : return self.word_index[w]
    def suff2idx(self, s) : return self.suf_index[s]
    def label2idx(self, l) : return self.label_index[l]

    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i: return l
        raise KeyError

    def _load(self, name) :
        self.maxlen, self.suflen, self.preflen, self.maxcharlen = 0, 0, 0, 20
        self.word_index, self.lc_word_index, self.suf_index, self.pref_index, self.label_index, self.char_index, self.pos_index = {}, {}, {}, {}, {}, {}, {} 
        with open(name+".idx") as f :
            for line in f.readlines():
                parts = line.split()
                if len(parts) < 3: continue
                t, k, i = parts[0], parts[1], parts[2]
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)
                elif t == 'PREFLEN' : self.preflen = int(k)
                elif t == 'MAXCHARLEN': self.maxcharlen = int(k)
                elif t == 'WORD' : self.word_index[k] = int(i)
                elif t == 'LCWORD' : self.lc_word_index[k] = int(i)
                elif t == 'SUF' : self.suf_index[k] = int(i)
                elif t == 'PREF' : self.pref_index[k] = int(i)
                elif t == 'LABEL' : self.label_index[k] = int(i)
                elif t == 'CHAR' : self.char_index[k] = int(i)
                elif t == 'POS' : self.pos_index[k] = int(i) 

    def save(self, name) :
        with open(name+".idx","w") as f :
            print('MAXLEN', self.maxlen, "-", file=f)
            print('SUFLEN', self.suflen, "-", file=f)
            print('PREFLEN', self.preflen, "-", file=f)
            print('MAXCHARLEN',self.maxcharlen,"-", file=f)
            for key, val in self.label_index.items(): print('LABEL', key, val, file=f)
            for key, val in self.word_index.items(): print('WORD', key, val, file=f)
            for key, val in self.lc_word_index.items():print('LCWORD',key, val, file=f)
            for key, val in self.suf_index.items(): print('SUF', key, val, file=f)
            for key, val in self.pref_index.items(): print('PREF', key, val, file=f)
            for key, val in self.char_index.items(): print('CHAR', key, val, file=f)
            for key, val in self.pos_index.items(): print('POS', key, val, file=f)