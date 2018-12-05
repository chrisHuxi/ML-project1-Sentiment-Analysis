from collections import Counter
import json
import numpy as np
import random

def ReadListAndDictFromFile(readFileName):
    with open(readFileName, "r",encoding='UTF-8') as f:
        readedList =  json.loads(f.read())
        return readedList

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
        
def BalanceData(originalData,originalLabel,distributionDict):
    originalData,originalLabel = ShuffleAllData(originalData,originalLabel)
    distributionKeys = distributionDict.keys()
    newDistributionDict = {}
    balanceData = []
    balanceLabel = []
    for key in distributionKeys:
        newDistributionDict[key] = 0
        
    for i in range(len(originalLabel)):
        if newDistributionDict[originalLabel[i]] < distributionDict[originalLabel[i]]:
            newDistributionDict[originalLabel[i]] += 1
            balanceData.append(originalData[i])
            balanceLabel.append(originalLabel[i])
        
    return balanceData,balanceLabel

def WriteListAndDictToFile(writeList,writeFileName):
    writeList = json.dumps(writeList)
    with open(writeFileName, "w",encoding='UTF-8') as f:
        f.write(writeList)
        
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
        
        
originalData = ReadListAndDictFromFile("./w2v_data/CNN_sentence_list.txt")
originalLabel = ReadListAndDictFromFile("./w2v_data/label_list.txt")
print ("----")


#originalData, originalLabel = ExtendLabel(originalData,originalLabel,minLength=4,maxLength = 32)

distributionDict = {1.0:40000,2.0:40000,3.0:80000,4.0:40000,5.0:40000}
balanceData,balanceLabel = BalanceData(originalData,originalLabel,distributionDict)
balanceData,balanceLabel = ShuffleAllData(balanceData,balanceLabel)
print(len(balanceLabel))
print(balanceLabel.count(3.0))

WriteListAndDictToFile(balanceData,"./balance_data/balanceData.txt")
WriteListAndDictToFile(balanceLabel,"./balance_data/balanceLabel.txt")
