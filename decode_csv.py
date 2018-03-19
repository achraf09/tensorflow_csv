import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import csv
import tflearn
import random
import json
import string
import unicodedata
###################Data Load and Pre-processing
categories = []
docs = []
words=[]
w1=[]
w2=[]
w3=[]
w4=[]
with open(sys.argv[1], newline='') as file:
    spamreader = csv.reader(file, delimiter=';')
    for row in spamreader:
        if row[0] == "city": categories=row
        else:
            if row[0]!="": w1.append(row[0])
            if row[1]!="": w2.append(row[1])
            if row[2]!="": w3.append(row[2])
            if row[3]!="": w4.append(row[3])
            words.extend(row)
docs.append((w1,categories[0]))
docs.append((w2,categories[1]))
docs.append((w3,categories[2]))
docs.append((w4,categories[3]))
#print(words)
#print(docs)           
print(categories)
############################################Training###########
trainig=[]
output=[]
output_empty=[0] * len(categories) #array for output
i=0
for doc in docs:
    #intialize our bag of words(bow) for each document in the list
    bow=[]
    #list of tokenised words for the pattern
    token_words= doc[0]
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)
        i+=1
        print(i)
    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1
    
    #our training set will contain a bag of words models and the output row
    trainig.append([bow, output_row])
    
# shuffle our features and turn into np.array as tensorflow takes numpy array
random.shuffle(trainig)
trainig = np.array(trainig)

#trainX contains the Bag of words and trainY contains the label/ category
train_x=list(trainig[:,0])
train_y=list(trainig[:,1])
print(train_x)
#####Text Classification#####

#reset underlying graph data
tf.reset_default_graph()
#Build neural Network

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

#Define Model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
#Start training (apply gradient decent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)

#############################Testing the tensorflow Text Classification##############
