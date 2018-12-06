from path import Path
import json
import numpy as np
import pandas as pd

# NB_sentence_list
# label_list

source_path = r'/Users/horczech/PycharmProjects/naviveBayes/data/preprocessed_data/dataset_1M/NB_sentence_list.txt'
target_path = r'/Users/horczech/PycharmProjects/naviveBayes/data/preprocessed_data/dataset_1M/NB_sentence_list.txt'


def preprocess_labels(labels):
    # 0 - negative
    # 1 - neutral
    # 2 - positive
    labels = np.asarray(labels, dtype=int)
    labels[labels <= 2] = 0
    labels[labels == 3] = 1
    labels[labels >= 4] = 2

    return pd.DataFrame(labels)

def preprocess_data(data):
    # modify the input data
    data = np.asarray([[' '.join(sentence)] for sentence in data])

    return pd.DataFrame(data)


def read_data_from_file(readFileName):
    with open(str(readFileName), "r", encoding='UTF-8') as f:
        readedList = json.loads(f.read())
        return readedList


print('Loading data')
data = read_data_from_file(source_path)

data = data


print('Modifiing data')
# data = preprocess_labels(data)
data = preprocess_data(data)



data.to_json(path_or_buf=target_path)

print('DONE')



