#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize


def load_drug_dictionaries():
   drugbank = set()
   drugbank_types = {}
   hsdb = set()

   with open("DDI/resources/DrugBank.txt", encoding="utf-8") as f:
      for line in f:
         line = line.strip()
         if not line:
            continue
         fields = line.split("|")
         name = fields[0].lower()
         typ = fields[1] if len(fields) > 1 else "unknown"
         drugbank.add(name)
         drugbank_types[name] = typ

   with open("DDI/resources/HSDB.txt", encoding="utf-8") as f:
      for line in f:
         name = line.strip().lower()
         if name:
            hsdb.add(name)

   return drugbank, drugbank_types, hsdb


DRUGBANK, DRUGBANK_TYPES, HSDB = load_drug_dictionaries()


   
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt, preserve_line=True):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens) :

   # for each token, generate list of features and add it to the result
   result = []
   for k in range(0,len(tokens)):
      tokenFeatures = [];
      t = tokens[k][0]

      tokenFeatures.append("form="+t)
      tokenFeatures.append("lower="+t.lower())
      tokenFeatures.append("pref3="+t[:3])
      tokenFeatures.append("pref4="+t[:4])
      tokenFeatures.append("pref5="+t[:5])
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("isCapitalized="+str(t[:1].isupper()))
      tokenFeatures.append("isUpperCase="+str(t.isupper()))
      tokenFeatures.append("containsDigit="+str(any(c.isdigit() for c in t)))
      tokenFeatures.append("containsHyphen="+str("-" in t))
      tokenFeatures.append("inDrugBank="+str(t.lower() in DRUGBANK))
      tokenFeatures.append("inHSDB="+str(t.lower() in HSDB))
      tokenFeatures.append("drugBankType="+DRUGBANK_TYPES.get(t.lower(), "none"))

      if k>0 :
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
      else :
         tokenFeatures.append("BoS")

      if k>1 :
         tPrev2 = tokens[k-2][0]
         tokenFeatures.append("formPrev2="+tPrev2)
         tokenFeatures.append("suf3Prev2="+tPrev2[-3:])
      else :
         tokenFeatures.append("formPrev2=BoS2")
         tokenFeatures.append("suf3Prev2=BoS2")

      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("suf3Next="+tNext[-3:])
      else:
         tokenFeatures.append("EoS")

      if k<len(tokens)-2 :
         tNext2 = tokens[k+2][0]
         tokenFeatures.append("formNext2="+tNext2)
         tokenFeatures.append("suf3Next2="+tNext2[-3:])
      else:
         tokenFeatures.append("formNext2=EoS2")
         tokenFeatures.append("suf3Next2=EoS2")
    
      result.append(tokenFeatures)
    
   return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
