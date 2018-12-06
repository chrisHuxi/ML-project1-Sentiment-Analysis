from path import Path
import json
import numpy as np
import pandas as pd

source_path = r'/Users/horczech/PycharmProjects/naviveBayes/data/preprocessed_data/dataset_1M_neg/label_list.txt'
target_path = r'/Users/horczech/PycharmProjects/naviveBayes/data/preprocessed_data/dataset_500k_neg/NB_sentence_list.txt'


def read_data_from_file(readFileName):
    with open(str(readFileName), "r", encoding='UTF-8') as f:
        readedList = json.loads(f.read())
        return readedList


data = pd.read_json(source_path)

print('Data loaded')

print('Data splitted')

data.to_json(target_path)
# with open(target_path, 'w') as outfile:
#     json.dump(data, outfile)

print('DONE')