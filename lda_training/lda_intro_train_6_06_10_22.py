# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 00:10:54 2022

@author: aq75iwit
"""
'''
Topic Modeling with LDA
References:
[1] LDA with Gensim: https://radimrehurek.com/gensim/models/ldamodel.html
'''

# Import dependencies
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stops = nlp.Defaults.stop_words

PATH='D:/01_Diss_Data/'

sample = pd.read_pickle(PATH+'data_transcripts_28_09_22.pkl')

dates_train_1=['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017']


train_1=sample[sample['year'].isin(dates_train_1)].reset_index()


   

def prepare(data):
    docs=[]
    for d in data['general_prepro']:
        docs.append(d)
    return docs
    


def normalize(comment, remove_stopwords):
    new_dat=[]
    for trans in docs:
        new_trans=[]
        for comment in trans:
            comment = nlp(comment)
            lemmatized = list()
            #lemmatized_docs = []
            for word in comment:
                lemma = word.lemma_.strip()
                if lemma:
                    if not remove_stopwords or (remove_stopwords and lemma not in stops):
                        lemmatized.append(lemma)
            new_trans.append(" ".join(lemmatized))
        new_dat.append(new_trans)
    return new_dat



def tokenize(docs):
    tokenized_docs = []
    for doc in lemmatized_docs:
        temp=[]
        for d in doc:
            tokens = gensim.utils.simple_preprocess(d, deacc=True)
            temp.append(tokens)
        tokenized_docs.append(temp)
    return tokenized_docs

def pre_dict(data):
    pre_dict=[]

    for t in tokenized_docs:
        for e in t:
            pre_dict.append(e)
    return pre_dict




# Pre-process input: stopwords, lemmatization and tokenization
docs = prepare(train_1)
lemmatized_docs = normalize(docs, remove_stopwords=True)
tokenized_docs = tokenize(lemmatized_docs)
corp=pre_dict(tokenized_docs)




# Mapping from word IDs to words
id2word = corpora.Dictionary(corp)

class MyCorpus(object):
    def __iter__(self):
        for text in corp:
            yield id2word.doc2bow(text)
corpus = MyCorpus() 

    
num_top=[20]


for n in num_top:
    # Fit LDA model: See [1] for more details
    topic_model=gensim.models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=n,
                                id2word=id2word, workers=4, chunksize=1000,
                                passes=25,
                                alpha=0.5,
                                iterations=1000,
                                random_state=4,
                                dtype=np.float64)
    topic_model.save(PATH+'LDA_gen_'+str(n)+'train_6'+'.model')