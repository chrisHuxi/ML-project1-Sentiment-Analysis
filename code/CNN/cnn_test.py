from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import json
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import data_load as dl


def WriteListAndDictToFile(writeList,writeFileName):
    writeList = json.dumps(writeList)
    with open(writeFileName, "w",encoding='UTF-8') as f:
        f.write(writeList)
        
        
def ReadListAndDictToFile(readFileName):
    with open(readFileName, "r",encoding='UTF-8') as f:
        readedList =  json.loads(f.read())
        return readedList    

def EvaluateModel(model_name,testData,testLabel,sentenceAmountPerReview,voteWeightDict={0.0:2,1.0:4,2.0:3}):
    model = load_model(model_name)
    # 评估模型
    '''
    for review in testData:
        predictions = model.predict(review)
    '''
    if (model_name.startswith("model_3channel")):
        print("3ch")
        predictions = model.predict([testData,testData,testData])
    elif(model_name.startswith("model_2ch")):
        print("2ch")
        #predictions = model.predict([testData,testData,testData])#
        predictions = model.predict([testData,testData])
    else:
        predictions = model.predict(testData)
    #print(predictions)#需要看看是one hot 还是 数值的结果
    predictions = [float(np.argmax(i)) for i in predictions] #对oh的处理方式
    #print(predictions)
    reviewPrediction = []    
    offset = 0
    for i in range(len(sentenceAmountPerReview)):
        reviewPrediction.append(predictions[offset: offset + sentenceAmountPerReview[i]])
        offset += sentenceAmountPerReview[i]
    predictionResult = []
    for p in reviewPrediction:
        result = np.argmax([(p.count(0.0) * voteWeightDict[0.0]) ,(p.count(1.0)) * voteWeightDict[1.0] , (p.count(2.0)) * voteWeightDict[2.0]])
        predictionResult.append(float(result))
    
    print(classification_report(testLabel, predictionResult, labels = [0.0,1.0,2.0], target_names=['0.0','1.0','2.0']))
    print(confusion_matrix(testLabel, predictionResult))
    
        
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
        
def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return x, y

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
    
def main():
    minLength = 4
    maxLength = 64#这里可以设置成 64, 128
    newTrainDataList,newTrainLabelList,newValidDataList,newValidLabelList,testDataList,testLabelList = dl.GetNewDataAndLabel(minLength,maxLength)
    sentenceAmountPerReview = []
    for review in testDataList:
        sentenceAmountPerReview.append(len(review))
    print("test set sentence amount per review: ")
    #print(sentenceAmountPerReview)
    
    sequence_length = maxLength 
    ''''''
    padTrainData = pad_sentences(newTrainDataList, sequence_length, padding_word="<PAD/>")
    oneHotTrainLabel = to_categorical(newTrainLabelList,num_classes=3)
    
    padValidData = pad_sentences(newValidDataList, sequence_length, padding_word="<PAD/>")
    oneHotValidLabel = to_categorical(newValidLabelList,num_classes=3)
    
    newTestDataList,newTestLabelList = dl.ExtendLabel_test(testDataList,testLabelList,minLength = 0,maxLength = maxLength)

    class3Label = []
    for i in testLabelList:
        if i < 2.1:
            class3Label.append(0.0)
        elif i == 3.0:
            class3Label.append(1.0)  
        else:
            class3Label.append(2.0)          
    
    padTestData = pad_sentences(newTestDataList, sequence_length, padding_word="<PAD/>")
    print(len(padTestData[1]))
    print(len(padTestData[50]))
    #print(set(testLabelList))
    oneHotTestLabel = to_categorical(newTestLabelList)
    #print(oneHotTestLabel)

    ''''''
    allSentenceList = padTrainData[:]    
    print("train set size: ",len(allSentenceList))
    allSentenceList.extend(padValidData)
    print("train and valid set size: ",len(allSentenceList))
    allSentenceList.extend(padTestData)
    print("train and valid set and test set size: ",len(allSentenceList))
    
    #这个要加载进来
    #word2id = ...
    word2id_tmp,id2word_tmp = bulidWordVocabulary(allSentenceList)
    print("word2id_tmp size:",len(word2id_tmp))
    word2id = ReadListAndDictToFile("word2id_longsentence.txt")
    id2word = ReadListAndDictToFile("id2word_longsentence.txt")
    print("word2id size:",len(word2id))
    inputTestData,inputTestLabel = build_input_data(padTestData , oneHotTestLabel, word2id)
        
    model_name = r"model_3channel_longSentence.h5"
    EvaluateModel(model_name,inputTestData,class3Label,sentenceAmountPerReview)
    
if __name__ == '__main__':
    main()
        