# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:51:24 2020

@author: hritvikgupta
"""

import pandas as pd
import numpy as np
import nltk
lemmer  = nltk.stem.WordNetLemmatizer() 
import string
import re
import sklearn
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import string
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
#from sklearn.metrics import confusion metrics, plot_roc_curve
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

#for evaluating the generated summmary and the system summary
import rouge
for aggregator in ['Avg', 'Best', 'Individual']:
    print('Evaluation with {}'.format(aggregator))
apply_avg = aggregator == 'Avg'
apply_best = aggregator == 'Best'
evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                       max_n=4,
                       limit_length=True,
                       length_limit=100,
                       length_limit_type='words',
                       apply_avg=apply_avg,
                       apply_best=apply_best,
                       alpha=0.5, # Default F1_score
                       weight_factor=1.2,
                       stemming=True)

#dataset loading
n_s = pd.read_csv(r"C:\Users\hritvik gupta\Desktop\google_images\n_s1.csv", encoding = 'latin-1')

#data cleaning
def clean_text(text):
    text = ''.join([i for i in text if i not in string.punctuation])
    token = re.split('/W+', text)
    text = ''.join([lemmer.lemmatize(i) for i in token if token not in stopwords])
    return text
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

stop_words = set(stopwords) 
def text_cleaner(text):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

#svd model
def svd_bow(news):
    sentences = news.split('.')
    len_sent = len(sentences)
    sentences_cleaned_= [text_cleaner(i) for i in sentences]
    sentences_cleaned = [clean_text(i) for i in sentences]
    vect_2 = TfidfVectorizer()
    x = vect_2.fit_transform(sentences_cleaned_)
    svd_model = TruncatedSVD(n_components = 20, algorithm= 'randomized', n_iter = 100, random_state=122)
    svd_model.fit(x)
    word_lsa = []
    terms2 = vect_2.get_feature_names()
    for i, comp in enumerate(svd_model.components_):
        term_comp = zip(terms2, comp)
        sorted_items = sorted(term_comp, key = lambda x :x[1], reverse = True)[:7]
        for t in sorted_items:
            word_lsa.append(t[0])
        
        
    wordfreq = {}
    for sentence in sentences_cleaned_:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    
    import heapq
    most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)
    
    #len(sentence_vectors[0][:10])
    sentence_vectors = []
    for sentence in sentences_cleaned_:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append((1, token))
            else:
                sent_vec.append((0, token))
        sentence_vectors.append(sent_vec)
        
    top_bow = []
    
    for j in range(len(sentence_vectors)):
        lis = list()
        for k in range(len(sentence_vectors[j])):
            for i in word_lsa:
                if i == sentence_vectors[j][k][1]:
                    if sentence_vectors[j][k] not in lis:
                        lis.append(sentence_vectors[j][k])
        top_bow.append(lis)
  
    top_bow_sum = []
    for i in range(len(top_bow)):
        sum2 = 0
        for j in range(len(top_bow[i])):
            sum2 = sum2 + top_bow[i][j][0]
        top_bow_sum.append(sum2/j)
    
    top_bow_index = dict()
    for i in range(len(top_bow_sum)):
        top_bow_index[i] = top_bow_sum[i]
    
    ind = sorted(top_bow_index, key = top_bow_index.get, reverse = True)[:10]
    sorted_sum2 = ".".join([sentences_cleaned[i] for i in ind]) 
    return sorted_sum2

#calculating for the 100 documents
svd_top_bow1= []
for i in range(100):
    svd_top_bow1.append(svd_bow(n_s['ctext'][i]))

#given summary
tex = []
for i in range(100):
    tex.append(n_s['text'][i])

#avlauting summary 
#calculate function calculate the mean of the scores
def calculate(score1, score2,param,length):
    f_s1, p_s1, r_s1 = 0,0,0
    f_s2, p_s2, r_s2 = 0,0,0
    for i in range(length):
        f_s1 = score1[param][i]['f'][0] + f_s1
        p_s1 = score1[param][i]['p'][0] + p_s1
        r_s1 = score1[param][i]['r'][0] + r_s1
        
        f_s2 = score2[param][i]['f'][0] + f_s2
        p_s2 = score2[param][i]['p'][0]+ p_s2
        r_s2 = score2[param][i]['r'][0] + r_s2
    f_s1, p_s1, r_s1 = f_s1/length,p_s1/length,r_s1/length
    f_s2, p_s2, r_s2 = f_s2/length,p_s2/length,r_s2/length
    
    return (f_s1, p_s1, r_s1),(f_s2, p_s2, r_s2)

s_ = evaluator.get_scores(sum_top_bowl1, tex)
calculate(s_, s_, 'rouge-1', 100)
