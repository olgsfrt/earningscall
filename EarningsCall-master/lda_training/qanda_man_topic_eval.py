# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 09:28:26 2022

@author: aq75iwit
"""
#import smart_open
import gensim
import gensim.corpora as corpora

#from gensim.utils import simple_preprocess
import spacy
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stops = nlp.Defaults.stop_words


PATH='D:/01_Diss_Data/'

dates_train=[['2007','2008','2009','2010','2011','2012']]

def prepare(data):
    docs=[]
    for d in data['qanda_manager_text']:
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
sample = pd.read_pickle(PATH+'data_transcripts_28_09_22.pkl')


x=0
for d in dates_train:
    print(f"Running for training dates {str(d)}")
    
    train_1=sample[sample['year'].isin(d)].reset_index()
    
    # Pre-process input: stopwords, lemmatization and tokenization
    docs = prepare(train_1)
    print(f"Preprocessing with {len(docs)} docs...")
    
    lemmatized_docs = normalize(docs, remove_stopwords=True)
    tokenized_docs = tokenize(lemmatized_docs)
    corp=pre_dict(tokenized_docs)
    
    #corp.to_pickle(PATH+'corp_train_'+str(x)+'.pkl')
    
    # Mapping from word IDs to words
    id2word = corpora.Dictionary(corp)
    id2word.save(PATH+'id2word_'+str(x)+'.dict')
    
    class MyCorpus(object):
        def __iter__(self):
            for text in corp:
                yield id2word.doc2bow(text)
    corpus = MyCorpus() 
    corpora.MmCorpus.serialize(PATH+'corp_'+str(x)+'.mm', corpus)
    
        
num_top=[5,10,15,20,25,30,35,40,45,50]


for n in num_top:
    # Fit LDA model: See [1] for more details
    topic_model=gensim.models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=n,
                                id2word=id2word, workers=4, chunksize=1000,
                                passes=25,
                                alpha=0.5,
                                iterations=1000,
                                random_state=4,
                                dtype=np.float64)
    topic_model.save(PATH+'LDA_qanda_man'+str(n)+'.model')