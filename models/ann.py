# -*- coding: utf-8 -*-

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import keras
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(27, 13), max_iter=500, random_state=0, early_stopping=True)
model = Pipeline([('scaler', QuantileTransformer()), ('ANN', model)])


def create_model():
    classifier = keras.Sequential()
    
    classifier.add(keras.layers.Dense(10, input_dim = 54, activation = 'tanh'))
    classifier.add(keras.layers.Dropout(0.2))
    classifier.add(keras.layers.Dense(10, activation = 'tanh'))
    classifier.add(keras.layers.Dropout(0.2))
    #classifier.add(keras.layers.Dense(10, activation = 'tanh'))
    #classifier.add(keras.layers.Dropout(0.2))
    classifier.add(keras.layers.Dense(2, activation = 'softmax'))
    
    opt = keras.optimizers.Nadam()
    
    classifier.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'], 
                       #callbacks=[earl]
                       )
    
    return classifier


from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn = create_model, batch_size = 512, epochs = 300, 
                        verbose = 0, validation_split = 0.1)

model = Pipeline([('scaler', QuantileTransformer(output_distribution='uniform')), ('ANN', model)])
