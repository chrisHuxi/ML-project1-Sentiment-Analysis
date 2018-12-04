# -*- coding: UTF-8 -*-
from copy import *
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import string
import re


import sys
sys.path.append("..")

import imp
import text_read_wirte as Textrw
imp.reload(Textrw)


def readText(fileName, classifier = "CNN"):   
    textList = []
    with open(fileName,'r') as f:
        i = 0
        while(1):
            readedText = f.readline()
            i += 1
            if (i%11 - 10 == 0): #review text
                textList.append(readedText.strip()[13:])
            else:
                continue
            if readedText == '':
                break
            if i>10000*11:
                break
    stop = set(stopwords.words('english') + list(string.punctuation))
    if classifier == "CNN":
        paragraphList = []
        #split paragraph to sentence
        for text in textList:
            paragraphList.append(sent_tokenize(text))
        
        paragrapSentenceList = []
        #split sentence to word list
        for paragrap in paragraphList:
            sentenceList = []
            for sentence in paragrap:
                sentence = extendAbbreviations(sentence) #extend abbreviations
                tmpSentence = []
                for word in word_tokenize(text=sentence,language="english"):
                    wordLower = word.lower() 
                    if wordLower not in stop and wordLower.isalpha():
                        tmpSentence.append(wordLower)
                sentenceList.append(tmpSentence)
                tmpSentence = []
            paragrapSentenceList.append(sentenceList)
            sentenceList = []
        return paragrapSentenceList

    elif classifier == "NB":
        porter = PorterStemmer()
        lemmatiser = WordNetLemmatizer()
        paragraphList = []
        #split paragraph to sentence
        for text in textList:
            text = extendAbbreviations(text) #extend abbreviations
            tokenizeText = word_tokenize(text=text,language="english")
            tokens_pos = pos_tag(tokenizeText) 
            tmpSentence = []
            for word,pos in tokens_pos:
                wordLower = word.lower()
                simplePosTag = simplifyPosTag(pos)
                if wordLower not in stop and wordLower.isalpha():
                    wordStemmed = lemmatiser.lemmatize(wordLower, pos=simplePosTag)
                    tmpSentence.append(wordStemmed)
            paragraphList.append(tmpSentence)        
            tmpSentence = []
        return paragraphList


def extendAbbreviations(text):
    replace_patterns = [
    (r"can\'t", "cannot"),
    (r"won't", "will not"),
    (r"i'm", "i am"),
    (r"isn't", "is not"),
    (r"(\w+)'ll", "\g<1> will"),
    (r"(\w+)n't", "\g<1> not"),
    (r"(\w+)'ve", "\g<1> have"),
    (r"(\w+)'s", "\g<1> is"),
    (r"(\w+)'re", "\g<1> are"),
    (r"(\w+)'d", "\g<1> would"),
    ]
    parrents = [(re.compile(regex), repl) for regex, repl in replace_patterns]
    for parrent, repl in parrents:
            text, count = re.subn(pattern=parrent, repl=repl, string=text)
    return text    
    
    
def simplifyPosTag(posTag):
    if posTag.startswith('J'):
        return wordnet.ADJ
    elif posTag.startswith('V'):
        return wordnet.VERB
    elif posTag.startswith('N'):
        return wordnet.NOUN
    elif posTag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN
        
        
        
        
def BulidDictionary2D(resultList):
    wholeWord = []
    for sentence in resultList:
        wholeWord.extend(sentence)        
    wordSet = set(wholeWord)
    word2IdDict = dict()
    id2WordDict = dict()
    i = 0
    for everyWord in wordSet:
        word2IdDict[everyWord] = i
        id2WordDict[i] = everyWord
        i = i + 1
    return id2WordDict,word2IdDict
    
    

def BulidDictionary3D(resultList):
    wholeWord = []
    for paragrap in resultList:
        for sentence in paragrap:
            wholeWord.extend(sentence)        
    wordSet = set(wholeWord)
    word2IdDict = dict()
    id2WordDict = dict()
    i = 0
    for everyWord in wordSet:
        word2IdDict[everyWord] = i
        id2WordDict[i] = everyWord
        i = i + 1
    return id2WordDict,word2IdDict
        
def MapwordsToIndex2D(sentenceList,word2IdDict):
    sentenceIdList = []
    for sentence in sentenceList:
        wordIdList = []
        for word in sentence:
             wordIdList.append(word2IdDict[word])
        sentenceIdList.append(wordIdList)
    return sentenceIdList

        
def MapwordsToIndex3D(paragrapSentenceList,word2IdDict):
    paragrapIdList = []
    for paragrap in paragrapSentenceList:
        sentenceIdList = []
        for sentence in paragrap:
            wordIdList = []
            for word in sentence:
                wordIdList.append(word2IdDict[word])
            sentenceIdList.append(wordIdList)
        paragrapIdList.append(sentenceIdList)
    return paragrapIdList
 
    
    
if __name__ == '__main__':
    #≤‚ ‘¥˙¬Î
    fileName = r"Books.txt"
    
    textList = readText(fileName)
    print("=====================================")
    print("text includes: ")
    print(len(textList))
    print(" reviews ")
    print("=====================================")
    Textrw.WriteListAndDictToFile(textList,r"CNN_sentence_list.txt")
    id2WordDict,word2IdDict = BulidDictionary3D(textList)
    Textrw.WriteListAndDictToFile(id2WordDict,r"CNN_id2word.txt")
    Textrw.WriteListAndDictToFile(word2IdDict,r"CNN_word2id.txt")
    textidList = MapwordsToIndex3D(textList,word2IdDict)
    Textrw.WriteListAndDictToFile(textidList,r"CNN_sentence_id_list.txt")
    '''
    textList = readText(fileName,"NB")
    print("=====================================")
    print("text includes: ")
    print(len(textList))
    print(" reviews ")
    print("=====================================")
    Textrw.WriteListAndDictToFile(textList,r"NB_sentence_list.txt")
    id2WordDict,word2IdDict = BulidDictionary2D(textList)
    Textrw.WriteListAndDictToFile(id2WordDict,r"NB_id2word.txt")
    Textrw.WriteListAndDictToFile(word2IdDict,r"NB_word2id.txt")
    textidList = MapwordsToIndex2D(textList,word2IdDict)
    Textrw.WriteListAndDictToFile(textidList,r"NB_sentence_id_list.txt")
    '''    
    
    #pass