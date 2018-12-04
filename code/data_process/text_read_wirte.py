 # -*- coding: UTF-8 -*-
import json


def WriteListAndDictToFile(writeList,writeFileName):
    writeList = json.dumps(writeList)
    with open(writeFileName, "w",encoding='UTF-8') as f:
        f.write(writeList)
        
        
def ReadListAndDictToFile(readFileName):
    with open(readFileName, "r",encoding='UTF-8') as f:
        readedList =  json.loads(f.read())
        return readedList    
        
    
if __name__ == '__main__':
    #≤‚ ‘¥˙¬Î
    pass