# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:37:49 2023

@author: Mels
"""
#import the necessary libraries
import numpy as np
from datasets import load_dataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from keras.utils import to_categorical
from gensim.models import Word2Vec


#%% import data
dataset = load_dataset("liar")   
    
if True:
    train_label = np.array(dataset["train"]['label'])
    test_label = np.array(dataset["test"]['label'])
    validation_label = np.array(dataset["validation"]['label'])
    
    train_label = np.where(train_label==1, 0, train_label)
    train_label = np.where(train_label==2, 1, train_label)
    train_label = np.where(train_label==3, 1, train_label)
    train_label = np.where(train_label==4, 2, train_label)
    train_label = np.where(train_label==5, 2, train_label)
    
    test_label = np.where(test_label==1, 0, test_label)
    test_label = np.where(test_label==2, 1, test_label)
    test_label = np.where(test_label==3, 1, test_label)
    test_label = np.where(test_label==4, 2, test_label)
    test_label = np.where(test_label==5, 2, test_label)
    
    validation_label = np.where(validation_label==1, 0, validation_label)
    validation_label = np.where(validation_label==2, 1, validation_label)
    validation_label = np.where(validation_label==3, 1, validation_label)
    validation_label = np.where(validation_label==4, 2, validation_label)
    validation_label = np.where(validation_label==5, 2, validation_label)

else:
    train_label = dataset["train"]['label']
    test_label = dataset["test"]['label']
    validation_label = dataset["validation"]['label']
        
print('\nLabels to be trained to: ',np.unique(train_label),'\n')


#%% create embedding
#tokenize the statements
max_words = 100                 
embedding_dim = 50
output_dim = tf.unique(train_label)[0].shape[0] 
 
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(dataset["train"]['statement'])
sequences_statement = tokenizer.texts_to_sequences(dataset["train"]['statement'])

#tokenize the statements
tokenizer.fit_on_texts(dataset["train"]['subject'])
sequences_subject = tokenizer.texts_to_sequences(dataset["train"]['subject'])
vocab_size = len(tokenizer.word_index) + 1  # add 1 for the unknown word token 

#pad the sequences
x_data_subject = pad_sequences(sequences_statement, maxlen=max_words)
x_data_statement = pad_sequences(sequences_subject, maxlen=max_words)
y_data = to_categorical(train_label, num_classes=output_dim)

x_data = [x_data_subject, x_data_statement]
sequences_combined = sequences_statement + sequences_subject

# Load the pre-trained Word2Vec model
#train Word2Vec model
sentences = dataset["train"]['statement'] + dataset["train"]['subject']
sentences = [sentence.split() for sentence in sentences]
w2v_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# Create the embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        try:
            embedding_vector = w2v_model.wv[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            pass

# Create the embedding layer
embedding_layer = layers.Embedding(input_dim=max_words, output_dim=embedding_dim,
                            weights=[embedding_matrix], input_length=max_words)

#%% validation data
# create the test data
tokenizer.fit_on_texts(dataset["test"]['statement'])
sequences_statement = tokenizer.texts_to_sequences(dataset["test"]['statement'])

tokenizer.fit_on_texts(dataset["test"]['subject'])
sequences_subject = tokenizer.texts_to_sequences(dataset["test"]['subject'])

#pad the sequences
x_test = [pad_sequences(sequences_statement, maxlen=max_words), pad_sequences(sequences_subject, maxlen=max_words)]
y_test = to_categorical(test_label, num_classes=output_dim)


#%% Model architecture 
def build_model(hp):    
    num_filters = hp.Int('num_filters', min_value=32, max_value=256, step=32)           
    kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])            
    pool_size = hp.Choice('pool_size', values=[2, 3])                
    hidden_dim = hp.Int('hidden_dim', min_value=32, max_value=128, step=32)          
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)             
    
    input_subject = layers.Input(shape=(max_words,), dtype='int32')
    input_statement = layers.Input(shape=(max_words,), dtype='int32')
    #embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    embedded_subject = embedding_layer(input_subject)
    embedded_statement = embedding_layer(input_statement)
    merged = layers.Concatenate(axis=-1)([embedded_subject, embedded_statement])
    conv_layer = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')(merged)
    pool_layer = layers.MaxPooling1D(pool_size=pool_size)(conv_layer)
    lstm_layer = layers.LSTM(units=lstm_units)(pool_layer)
    dense_layer = layers.Dense(units=hidden_dim, activation='relu')(lstm_layer)
    output = layers.Dense(output_dim, activation='softmax')(dense_layer)

    model = keras.models.Model(inputs=[input_subject, input_statement], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model



#%% Search space
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='my_project')


#%% Running 
# Define an objective function to train the model and return a metric to optimize
def objective(trial):
    model = build_model(trial)
    model.fit(x_data, y_data, validation_data=(x_test, y_test), epochs=10, batch_size=32)
    return model.evaluate(x_test, y_test, verbose=0)[1]

# Run the tuner to search for the best hyperparameters
tuner.search(x_data, y_data, validation_data=(x_test, y_test), epochs=10, batch_size=32)


#%% Print the best hyperparameters found by the tuner
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f'Best hyperparameters: {best_hyperparams.values}')