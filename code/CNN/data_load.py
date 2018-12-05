import json
import random


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
                     newLabelList.append(0.0)
                elif label_list[i] == 3.0:
                     newLabelList.append(1.0)  
                else:
                     newLabelList.append(2.0)  
    assert len(newLabelList) == len(newSentenceList)   
    return newSentenceList, newLabelList
    
def ExtendLabel_test(sentence_list,label_list,minLength = 0,maxLength = 999):
    newSentenceList = []
    newLabelList = []
    assert len(sentence_list) == len(label_list)
    for i in range(len(label_list)):
        for j in range(len(sentence_list[i])):
            if (len(sentence_list[i][j]) < minLength):
                continue
            elif (len(sentence_list[i][j]) > maxLength):
                newSentenceList.append(sentence_list[i][j][0:maxLength])
                if label_list[i] < 2.1:
                     newLabelList.append(0.0)
                elif label_list[i] == 3.0:
                     newLabelList.append(1.0)  
                else:
                     newLabelList.append(2.0)  
            else:
                newSentenceList.append(sentence_list[i][j])
                if label_list[i] < 2.1:
                     newLabelList.append(0.0)
                elif label_list[i] == 3.0:
                     newLabelList.append(1.0)  
                else:
                     newLabelList.append(2.0)  
    assert len(newLabelList) == len(newSentenceList)   
    return newSentenceList, newLabelList
    
def ShuffleAllData(dataList,labelList):
        #pass
        indexList = list(range(0,len(labelList)))
        random.shuffle(indexList)
        shuffledIndexList = indexList
        
        shuffledData = []
        shuffledLabel = []
        for i in range(0,len(dataList)):
            shuffledData.append(dataList[shuffledIndexList[i]])
            shuffledLabel.append(labelList[shuffledIndexList[i]])
        assert len(shuffledData) == len(shuffledLabel)    
        return shuffledData,shuffledLabel
        
def GetNewDataAndLabel(minLength = 0,maxLength = 999):
    sentenceList = ReadListAndDictFromFile("./balance_data/15000/balanceData.txt")
    labelList = ReadListAndDictFromFile("./balance_data/15000/balanceLabel.txt")
    
    print("================")
    print("data set amount:")
    print(len(sentenceList),len(labelList))
    print("================")
    
    validSetOffset = int((len(sentenceList))*0.8)
    testSetOffset = int((len(sentenceList))*0.9)
    
    trainDataList = sentenceList[0:validSetOffset]
    trainLabelList = labelList[0:validSetOffset]
    
    validDataList = sentenceList[validSetOffset:testSetOffset]
    validLabelList = labelList[validSetOffset:testSetOffset]
    
    testDataList = sentenceList[testSetOffset:]
    testLabelList = labelList[testSetOffset:]
    

    
    newTrainDataList,newTrainLabelList = ExtendLabel(trainDataList,trainLabelList,minLength,maxLength)
    
    shuffledData,shuffledLabel = ShuffleAllData(newTrainDataList,newTrainLabelList)
    
    #newValidDataList,newValidLabelList = ExtendLabel(validDataList,validLabelList,minLength = 0,maxLength = 999)
    newValidDataList,newValidLabelList = ExtendLabel(validDataList,validLabelList)
    #newTestDataList,newTestLabelList = ExtendLabel(testDataList,testDataList,minLength = 0,maxLength = 999)
    newTestDataList,newTestLabelList = testDataList,testLabelList
    
    return shuffledData,shuffledLabel,newValidDataList,newValidLabelList,newTestDataList,newTestLabelList
    


if __name__ == '__main__':
    minLength = 5
    maxLength = 32
    newTrainDataList,newTrainLabelList,newValidDataList,newValidLabelList,newTestDataList,newTestLabelList = GetNewDataAndLabel(minLength,maxLength)
    print("================")
    print("training data set amount:")
    print(len(newTrainDataList),len(newTrainLabelList))
    print((newTrainDataList)[1:100])
    print((newTrainLabelList)[1:100])
    
    
    print("\n")
    print("valid data set amount:")
    print(len(newValidDataList),len(newValidLabelList))
    print((newValidLabelList)[1:100])
    print("\n")
    print("valid data set amount:")
    print(len(newTestDataList),len(newTestLabelList))
    print("================")
    
    
