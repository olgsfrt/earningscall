# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:36:02 2023

@author: aq75iwit
"""



from laserembeddings import Laser
from bertopic.backend import BaseEmbedder
from bertopic import BERTopic
import pandas as pd

class LaserEmbedder(BaseEmbedder):
    def __init__(self, language="en"):
        self.laser = Laser()
        self.language = language
        
    def embed(self, documents, verbose=False):
        return self.laser.embed_sentences(documents, lang=self.language)
    
def prepare(data):
    docs=[]
    for d in data['general_prepro']:
        docs.append(d)
    return docs

def train_and_save_model(train_data, model_name):
    # Pre-process input
    docs = prepare(train_data)
    docs = [doc for sublist in docs for doc in sublist]

    # Initialize the LaserEmbedder
    laser_embedder = LaserEmbedder()

    # Initialize the BERTopic model with the custom embedding model
    bertopic = BERTopic(embedding_model=laser_embedder, calculate_probabilities=True)

    # Fit the model to your corpus
    topics, _ = bertopic.fit_transform(docs)

    # Save the model
    bertopic.save(f"{model_name}.pkl")

    # Return the topics for further use, if needed
    return topics
    

sample = pd.read_pickle('/content/drive/MyDrive/Diss/Sentiment/data_transcripts_28_09_22.pkl')


dates_train=[['2007','2008','2009','2010','2011','2012'],
             ['2007','2008','2009','2010','2011','2012','2013'],
             ['2007','2008','2009','2010','2011','2012','2013','2014'],
             ['2007','2008','2009','2010','2011','2012','2013','2014','2015'],
             ['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'],
             ['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017'],
             ['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018'],
             ['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']]




for i, date_range in enumerate(dates_train):
    train_data = sample[sample['year'].isin(date_range)].reset_index()
    model_name = f"bertopic_model_{i+1}"
    
    topics = train_and_save_model(train_data, model_name)
    print(f"Model {model_name} trained and saved.")




