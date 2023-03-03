# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:45:00 2023

@author: Mels Habold
Data downloaded from https://huggingface.co/datasets/liar
Based on research from https://arxiv.org/abs/1705.00648
And an article posted in https://www.analyticssteps.com/blogs/detection-fake-and-false-news-text-analysis-approaches-and-cnn-deep-learning-model 

"""

#import the necessary libraries
from datasets import load_dataset
import numpy as np
import tensorflow as tf
import warnings
 
warnings.filterwarnings(action = 'ignore')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import gensim
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
    
print('Labels to be trained to: ',np.unique(train_label))


#%% Define the hyperparameters
max_words = 100                 # max number of words in a statement
embedding_dim = 50              # dimension of the word vector
output_dim = tf.unique(train_label)[0].shape[0]  # number of output labels
num_filters = 128            # number of convolutional filters
kernel_size = 3              # size of convolutional kernel
pool_size = 2                # size of pooling window
hidden_dim = 50              # dimension of fully connected layers


#%% create embedding
#tokenize the statements
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(dataset["train"]['statement'])
sequences_statement = tokenizer.texts_to_sequences(dataset["train"]['statement'])

#tokenize the statements
tokenizer.fit_on_texts(dataset["train"]['subject'])
sequences_subject = tokenizer.texts_to_sequences(dataset["train"]['subject'])

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
embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim,
                            weights=[embedding_matrix], input_length=max_words)


#%% Model Architecture
# Define the two inputs
input_subject = Input(shape=(max_words,), dtype='int32')
input_statement = Input(shape=(max_words,), dtype='int32')

# Define the embedding layer
#embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_words)

# Apply the embedding layer to both inputs
embedded_subject = embedding_layer(input_subject)
embedded_statement = embedding_layer(input_statement)

# Merge the two embedded inputs using the Concatenate layer
merged = Concatenate(axis=-1)([embedded_subject, embedded_statement])

# Define the model
conv_layer = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='softmax')(merged)
pool_layer = MaxPooling1D(pool_size=pool_size)(conv_layer)
flatten_layer = Flatten()(pool_layer)
hidden_layer = Dense(hidden_dim, activation='relu')(flatten_layer)
output = Dense(output_dim, activation='softmax')(hidden_layer)

model = Model(inputs=[input_subject, input_statement], outputs=output)
model.summary()


#%% Train model
early_stop = EarlyStopping(monitor='val_loss', patience=3)

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
#model.fit(x_data, y_data, batch_size=32, epochs=10)
model.fit(x_data, y_data, validation_split=0.1, batch_size=32, epochs=10, callbacks=[early_stop])


#%% Evaluation
# create the test data
tokenizer.fit_on_texts(dataset["test"]['statement'])
sequences_statement = tokenizer.texts_to_sequences(dataset["test"]['statement'])

tokenizer.fit_on_texts(dataset["test"]['subject'])
sequences_subject = tokenizer.texts_to_sequences(dataset["test"]['subject'])

#pad the sequences
x_test = [pad_sequences(sequences_statement, maxlen=max_words), pad_sequences(sequences_subject, maxlen=max_words)]
y_test = to_categorical(test_label, num_classes=output_dim)

#evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

#print the results
print('Test loss:', loss)
print('Test accuracy:', accuracy)