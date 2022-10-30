import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import matplotlib as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import spacy
import pandas as pd

  
nlp = spacy.load("en_core_web_sm")

# import the text file 
with open('C:\\Users\\91817\\Desktop\\NLP_text.txt', 'r') as file:
  text = file.read()
  file.close()

# Removal of chapter names using regular expressions and python libarary-re
text = re.sub(r'Chapter.*\n.*', '\n\n', text)
 
# changing text to lowercase
text.lower()

# decontract certain words to normal form for better text understanding
def transforming(text):
    #removing URL
    text = re.sub(r"http\s+", "", text)

    #Decontracting most common words
    text = re.sub(r"couldn\'t", "could not", text) 
    text = re.sub(r"aren\'t", "are not", text) 
    text = re.sub(r"won\'t", "will not", text) 
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

text = transforming(text)

#with open('C:\\Users\\91817\\Desktop\\NLP_text.txt', 'w') as f:
    #f.write(text)


# tokenize the text
tokens = nltk.word_tokenize(text)

# frequency distribution of tokenized text
fdist = FreqDist(tokens)
fdist.plot(10)

# generate wordcloud of tokenized text
wc = WordCloud()
img = wc.generate_from_text(' '.join(tokens))
img.to_file('wordcloud_including_stopwords.jpeg') 


# remove stopwords
def remove_stopwords(tokens):
 return [word for word in tokens if word not in STOPWORDS]
tokens1 = remove_stopwords(tokens)

# frequency distribution without stopwords
fdist = FreqDist(tokens1)
fdist.plot(10)
wc = WordCloud()
img = wc.generate_from_text(' '.join(tokens1))
img.to_file('worcloud_excluding_stopwords.jpeg') 


#relation between word length and frequency of text
plt.figure(figsize=(16,8))
length=[len(words) for words in tokens]
pd.Series(length).value_counts()[:30].plot(kind='bar')

#POS Tagging
tagged=nltk.pos_tag(tokens)
print(tagged)

# frequency distribution of tags
def FrequencyDist (tags):
 wfd=FreqDist(t for (w, t) in tags)
 wfd
 wfd.plot(50)
FrequencyDist (tagged)

