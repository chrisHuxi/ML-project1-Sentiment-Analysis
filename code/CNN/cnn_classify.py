# -*- coding: utf-8 -*-
import numpy as np
import json
import random
from os.path import exists
import itertools
from collections import Counter

from gensim.models import word2vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D, Convolution2D, Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import backend as K
from keras.layers.core import Lambda

import data_load as dl
import train_w2v


def ReadListAndDictFromFile(readFileName):
    with open(readFileName, "r",encoding='UTF-8') as f:
        readedList =  json.loads(f.read())
        return readedList

        
def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding>= 0:
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        else:
            padded_sentences.append(sentence[0:sequence_length])
    return padded_sentences
        
def loadEmbedding(id2word,embeddingType = "w2v"):
    if embeddingType == "w2v":
        model_name = "w2v_word_embedding"
        if exists(model_name):
            embedding_model_w2v = word2vec.Word2Vec.load(model_name)
            print("Load existing Word2Vec model ", model_name)
            embedding_weights_w2v = {key: embedding_model_w2v[word] if word in embedding_model_w2v else  \
                                  np.random.uniform(-0.25, 0.25, embedding_model_w2v.vector_size)  \
                             for key, word in id2word.items()}        
            print("amount of word embedding:",len(embedding_weights_w2v))
            return embedding_weights_w2v
        else:
            print("not find Word2Vec model ", model_name)
            return None
            
    elif embeddingType == "glove":
        gloveModel = dict()
        #function: 按行读取大文件
        try:
            file = open(r'glove.twitter.27B.50d.txt', 'r',encoding='UTF-8')
            for line in file:
                vec = line.strip().split(' ')
                try:
                    floatList = []
                    for i in vec[1:]:
                        floatList.append(float(i))
                    gloveModel[vec[0]] = np.array(floatList)
                except UnicodeError:
                    continue
        except IOError as err:
            print('File error: ' + str(err))
        finally:
            if 'file' in locals():
                file.close()
        modelKeys = set(gloveModel.keys())
        vectorSize = 50
        embedding_weights_glove = {key: gloveModel[word] if word in modelKeys else  \
                                  np.random.uniform(-0.25, 0.25, vectorSize)  \
                             for key, word in id2word.items()}           
        print("amount of word embedding:",len(embedding_weights_glove))
        return embedding_weights_glove
        
    elif embeddingType == "sswe":
        ssweModel = dict()
        #function: 按行读取大文件
        try:
            file = open(r'sswe-u.txt', 'r',encoding='UTF-8')
            for line in file:
                vec = line.strip().split('	')
                try:
                    floatList = []
                    for i in vec[1:]:
                        floatList.append(float(i))
                    ssweModel[vec[0]] = np.array(floatList)
                except UnicodeError:
                    continue
        except IOError as err:
            print('File error: ' + str(err))
        finally:
            if 'file' in locals():
                file.close()
        modelKeys = set(ssweModel.keys())
        vectorSize = 50
        embedding_weights_sswe = {key: ssweModel[word] if word in modelKeys else  \
                                  np.random.uniform(-0.25, 0.25, vectorSize)  \
                             for key, word in id2word.items()}
        print("amount of word embedding:",len(embedding_weights_sswe))
        return embedding_weights_sswe
        
def expand_dims_backend(x):
    x1 = K.expand_dims(x,-1)
    return x1
    
def CreateModel(id2word,sequence_length):
    # Model Hyperparameters
    embedding_dim = 50
    filter_sizes = [(2,embedding_dim),(4,embedding_dim),(8,embedding_dim),(16,embedding_dim)]
    #num_filters = 256
    num_filters = 256
    dropout_prob = (0.5, 0.5)
    dropout_prob_conv = 0.5
    #hidden_dims = 128
    hidden_dims = 128

    input_shape = (sequence_length,)
    
    
    model_input_w2v = Input(shape=input_shape)
    model_input_glove = Input(shape=input_shape)
    model_input3_sswe = Input(shape=input_shape)
    
    emb1 = Embedding(len(id2word), embedding_dim, input_length=sequence_length, name="embedding_w2v")(model_input_w2v)
    emb2 = Embedding(len(id2word), embedding_dim, input_length=sequence_length, name="embedding_glove")(model_input_glove)
    emb3 = Embedding(len(id2word), embedding_dim, input_length=sequence_length, name="embedding_sswe")(model_input3_sswe)
    
    emb1_expandDim = Lambda(expand_dims_backend)(emb1) #这行还需要考虑考虑
    emb2_expandDim = Lambda(expand_dims_backend)(emb2)
    emb3_expandDim = Lambda(expand_dims_backend)(emb3)
    
    z = Concatenate()([emb1_expandDim,emb2_expandDim,emb3_expandDim])
    
    z = Dropout(dropout_prob[0])(z)
    
    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution2D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling2D(pool_size=(2,1))(conv)
        conv = Flatten()(conv) #这行还需要考虑考虑
        conv = Dropout(dropout_prob_conv)(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(3, activation="softmax")(z)

    model = Model([model_input_w2v,model_input_glove,model_input3_sswe], model_output)
    return model


def TrainModel(model,trainData,trainLabel,id2word,validData,validLabel):

    embedding_weights_w2v = loadEmbedding(id2word,embeddingType = "w2v")
    embedding_weights_glove = loadEmbedding(id2word,embeddingType = "glove")    #glove
    embedding_weights_sswe = loadEmbedding(id2word,embeddingType = "sswe")    #sswe
    
    
    # Training parameters
    batch_size = 256
    num_epochs = 15

    # Initialize weights with word2vec
    weights_w2v = np.array([v for v in embedding_weights_w2v.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights_w2v.shape)
    embedding_layer_w2v = model.get_layer("embedding_w2v")
    embedding_layer_w2v.set_weights([weights_w2v])
    
    # Initialize weights with Glove
    weights_glove = np.array([v for v in embedding_weights_glove.values()])
    print("Initializing embedding layer with glove weights, shape", weights_glove.shape)
    embedding_layer_glove  = model.get_layer("embedding_glove")
    embedding_layer_glove.set_weights([weights_glove])
    
    # Initialize weights with sswe
    weights_sswe = np.array([v for v in embedding_weights_sswe.values()])
    print("Initializing embedding layer with sswe weights, shape", weights_sswe.shape)
    embedding_layer_sswe  = model.get_layer("embedding_sswe")
    embedding_layer_sswe.set_weights([weights_sswe])
    
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])    
    model.summary()
    # Train the model
    model.fit([trainData,trainData,trainData], trainLabel, batch_size=batch_size, epochs=num_epochs,
              validation_data=([validData,validData,validData], validLabel), verbose=2)
    model.save('model.h5')
#def EvaluateModel():

def bulidWordVocabulary(sentenceList):
    wholeWord = []
    for sentence in sentenceList:
        wholeWord.extend(sentence)        
    wordSet = set(wholeWord)
    word2IdDict = dict()
    id2WordDict = dict()
    i = 0
    for everyWord in wordSet:
        word2IdDict[everyWord] = i
        id2WordDict[i] = everyWord
        i = i + 1
    return word2IdDict, id2WordDict


def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return x, y
    
def main():
    minLength = 5
    maxLength = 32
    newTrainDataList,newTrainLabelList,newValidDataList,newValidLabelList,newTestDataList,newTestLabelList = dl.GetNewDataAndLabel(minLength,maxLength)
    sequence_length = maxLength 
    
    padTrainData = pad_sentences(newTrainDataList, sequence_length, padding_word="<PAD/>")
    oneHotTrainLabel = to_categorical(newTrainLabelList,num_classes=3)
    
    padValidData = pad_sentences(newValidDataList, sequence_length, padding_word="<PAD/>")
    oneHotValidLabel = to_categorical(newValidLabelList,num_classes=3)
    
    newTestDataList,newTestLabelList = dl.ExtendLabel(newTestDataList,newTestLabelList)
    padTestData = pad_sentences(newTestDataList, sequence_length, padding_word="<PAD/>")
    
    allSentenceList = padTrainData[:]
    
    print("train set size: ",len(allSentenceList))
    allSentenceList.extend(padValidData)
    print("train and valid set size: ",len(allSentenceList))
    allSentenceList.extend(padTestData)
    print("train and valid set and test set size: ",len(allSentenceList))
    
    word2id,id2word = bulidWordVocabulary(allSentenceList)
    print("word2id size:",len(word2id))
    inputTrainData,inputTrainLabel = build_input_data(padTrainData , oneHotTrainLabel, word2id)
    inputValidData,inputValidLabel = build_input_data(padValidData , oneHotValidLabel, word2id)
    
    set_label = {}
    for i in newTrainLabelList:
        if i in set_label.keys():
            set_label[i] += 1
        else:
            set_label[i] = 1
    print(set_label)
    
    model = CreateModel(id2word,sequence_length)
    TrainModel(model,inputTrainData,inputTrainLabel,id2word,inputValidData,inputValidLabel)
    
    
if __name__ == '__main__':
    main()
    