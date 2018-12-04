import json
def ReadListAndDictFromFile(readFileName):
    with open(readFileName, "r",encoding='UTF-8') as f:
        readedList =  json.loads(f.read())
        return readedList
        
#for example
CNN_list = ReadListAndDictFromFile("CNN_sentence_id_list.txt")
label_list = ReadListAndDictFromFile("label_list.txt")

minLength = 4

new_CNN_list = []
new_label_list = []
for i in range(len(label_list)):
    for j in range(len(CNN_list[i])):
        if len(CNN_list[i][j]) < minLength:
            continue
        else:
            
            new_CNN_list.append(CNN_list[i][j])
            if label_list[i] < 2.1:
                new_label_list.append(1.0)
            elif label_list[i] == 3.0:
                new_label_list.append(2.0)  
            else:
                new_label_list.append(3.0)  
        
        
print(new_CNN_list[10:1000])
print(new_label_list[10:1000])
    

