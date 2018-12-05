import json
from gensim.models import word2vec
from collections import Counter
from os.path import exists
import numpy as np


def ReadListAndDictFromFile(readFileName):
    with open(readFileName, "r",encoding='UTF-8') as f:
        readedList =  json.loads(f.read())
        return readedList
        
def ExtendLabel(sentence_list,label_list,minLength = 0,maxLength = 999):
    newSentenceList = []
    newLabelList = []
    assert len(sentence_list) == len(label_list)
    for i in range(len(label_list)):
        for j in range(len(sentence_list[i])):
            if (len(sentence_list[i][j]) < minLength) or (len(sentence_list[i][j]) > maxLength):
                continue
            else:
                newSentenceList.append(sentence_list[i][j])
                if label_list[i] < 2.1:
                     newLabelList.append(1.0)
                elif label_list[i] == 3.0:
                     newLabelList.append(2.0)  
                else:
                     newLabelList.append(3.0)  
    assert len(newLabelList) == len(newSentenceList)   
    return newSentenceList, newLabelList        
        
def trainEmbedding_w2v(sentenceList, id2word,num_features=300,min_word_count=5, context=10):
    model_name = "w2v_word_embedding"
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print("Load existing Word2Vec model ", model_name)
    else:
        # Set values for various parameters

        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = sentenceList
        embedding_model = word2vec.Word2Vec(sentences, size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)
        # If we don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        # embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        print("Saving Word2Vec model ", model_name)
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                              np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in id2word.items()}
    return embedding_weights
    
if __name__ == '__main__':
    sentenceList = ReadListAndDictFromFile("./w2v_data/CNN_sentence_list.txt")
    labelList = ReadListAndDictFromFile("./w2v_data/label_list.txt")
    newSentenceList, newLabelList = ExtendLabel(sentenceList,labelList,minLength = 0,maxLength = 999)
    word2id = ReadListAndDictFromFile("./w2v_data/CNN_word2id.txt")
    id2word = ReadListAndDictFromFile("./w2v_data/CNN_id2word.txt")
    word2id["<PAD/>"] = len(word2id)
    id2word[len(id2word)] = "<PAD/>"
    
    embedding_weights = trainEmbedding_w2v(newSentenceList, id2word,num_features=50)
    
    
    