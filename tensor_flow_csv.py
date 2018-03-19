import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import csv
from collections import Counter

def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word]=i
    return word2index

vocab = Counter()
with open(sys.argv[1], newline='') as file:
    spamreader = csv.reader(file, delimiter=';')
    for row in spamreader:
        vocab[row[3]]+=1
print(vocab)
print("#########################################################################################")
word2index = get_word_2_index(vocab)
total_words=len(vocab)
print(word2index)
print("#########################################################################################")
matrix = np.zeros((total_words),dtype=float)
with open(sys.argv[1], newline='') as fil:
    spamreade = csv.reader(fil, delimiter=';')
    for row in spamreade:
        matrix[word2index[row[3]]]+=1
       
np.set_printoptions(threshold=np.nan)
print(matrix)
print(len(matrix))        



        
