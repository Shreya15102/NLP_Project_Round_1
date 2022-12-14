# -*- coding: utf-8 -*-
"""NLP_round2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wNRj41M7YnJ4_HkgtOG8rM0SlCzI0Vez
"""

import nltk
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from urllib.request import urlopen
import re
import inflect
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist
from collections import Counter
nltk.download('brown')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

p = inflect.engine()
lemmatizer = WordNetLemmatizer()
file = open('NLP_text.txt', encoding = 'utf-8')
text = file.read()

ef noun(text):
  is_noun = lambda pos: pos[:1] == 'N'
  tokenized = nltk.word_tokenize(text)
  nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
  return nouns

noun1=noun(text)

print("Number of nouns are "+ str(len(noun1)))

def verb(text):
  is_verb = lambda pos: pos[:1] == 'V'
  tokenized = nltk.word_tokenize(text)
  verbs = [word for (word, pos) in nltk.pos_tag(tokenized) if is_verb(pos)] 
  return verbs

verb1=verb(text)

print("Number of verbs are "+ str(len(verb1)))

from nltk.corpus import wordnet as wn

#gives the categories of nouns or verb that the word belongs to
from nltk.corpus import wordnet as wn
def synset(words):
  categories=[]
  for word in words:
    cat=[]
    for synset in wn.synsets(word):
      if(('noun' in synset.lexname()) & ('Tops' not in synset.lexname()) ):
        cat.append(synset.lexname())
      if('verb' in synset.lexname()):
        cat.append(synset.lexname())
    categories.append(cat)
  return categories

noun_synset1=synset(noun1)

verb_synset1=synset(verb1)


print(noun1[88])

print(noun_syn1[88][:])

#GIVES TOTAL NOUN LEXNAMES AND TOTAL VERB LEXNAMES FOR FREQUENCY DISTRIBUTIONS
def all_synsets(no,ve):
  nouns=[]
  verbs=[]
  for word in no:
    for synset in wn.synsets(word): 
      if(('noun' in synset.lexname()) & ('Tops' not in synset.lexname()) ):
        nouns.append(synset.lexname())
      if('verb' in synset.lexname()):
        verbs.append(synset.lexname())
  for word in ve:
    for synset in wn.synsets(word): 
      if(('noun' in synset.lexname()) & ('Tops' not in synset.lexname()) ):
        nouns.append(synset.lexname())
      if('verb' in synset.lexname()):
        verbs.append(synset.lexname())
      
  return nouns,verbs

noun_superset1,verb_superset1=all_synsets(noun1,verb1)

print(noun_superset1)

len(noun_superset1)

import numpy as np
labels, counts = np.unique(noun_superset1,return_counts=True)
import matplotlib.pyplot as plt 
ticks = range(len(counts))
plt.figure(figsize=(15,8))
plt.bar(ticks,counts, align='center')
plt.xticks(ticks, range(len(labels)))
labels, counts = np.unique(noun_superset2,return_counts=True)
ticks = range(len(counts))
plt.figure(figsize=(15,8))
plt.bar(ticks,counts, align='center')
plt.xticks(ticks, range(len(labels)))

print(labels)

labels, counts = np.unique(verb_superset1,return_counts=True)
ticks = range(len(counts))
plt.figure(figsize=(15,8))
plt.bar(ticks,counts, align='center')
plt.xticks(ticks, range(len(labels)))
labels, counts = np.unique(verb_superset2,return_counts=True)
ticks = range(len(counts))
plt.figure(figsize=(15,8))
plt.bar(ticks,counts, align='center')
plt.xticks(ticks, range(len(labels)))


    
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
import string
import spacy
from spacy import  displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
doc = nlp(text)

#f.close()
print("there are total " +str(len(doc.ents))+" entities")

print([(X,X.ent_iob_) for X in doc])

def entity_recognition(text):
    doc = nlp(text)
    person = []
    org = []
    location = []
    for X in doc:
        if(X.ent_type_ == 'PERSON') and X.text not in person:
            person.append(X.text)
        if(X.ent_type_ == 'ORG') and X.text not in org:
            org.append(X.text)
        if((X.ent_type_ == 'LOC') or (X.ent_type_ == 'GPE')) and X.text not in location:
            location.append(X.text)          
    return person,org,location



person1,org1,location1 = entity_recognition(text)
print("number of person entities: ",str(len(person1)))
print("number of org entities: ",str(len(org1)))
print("number of location entities: ",str(len(location1)))

def freq(str_list): 
    unique_words = set(str_list)
    counts = {}
    for words in unique_words : 
        counts[words] = str_list.count(words)
    return counts

X = freq(person1)
print(sorted(X.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))

X= freq(location1)
print(sorted(X.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))

X= freq(org1)
print(sorted(X.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))

import re
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.sem.relextract import extract_rels, rtuple
nltk.download('maxent_ne_chunker')
nltk.download('words')

BELONG = re.compile(r'.*\bin|from|belonged|lived\b.*')

sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

for i,sent in enumerate(tagged_sentences):
  sent = ne_chunk(sent)
  rels = extract_rels('PER','GPE',sent,corpus='ace',pattern = BELONG, window = 10)
  for rel in rels:
    print(rtuple(rel))

ORG = re.compile(r'.*\bwork|of|by|in\b.*')

for i,sent in enumerate(tagged_sentences):
  sent = ne_chunk(sent)
  rels = extract_rels('PER', 'ORG', sent, corpus = 'ace', pattern = ORG, window = 10)
  for rel in rels:
    print(rtuple(rel))



