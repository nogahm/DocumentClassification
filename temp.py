import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from os import listdir
from os.path import isfile, join

#variables
dirpath=os.getcwd()+'\\ohsumed-first-20000-docs'
trainDirs=[]
testDirs=[]
trainFiles=[]
testFiles=[]
# for table
Category=[]
FileName=[]
Path=[]
Text=[]
stopWords=set(stopwords.words('english'))

# get train and test dirs
trainDirs=os.listdir(dirpath+'\\training')
print(dirpath)
testDirs=os.listdir(dirpath+'\\test')
# extract category, name, path and text
for dir in trainDirs:
    for file in os.listdir(dirpath+'\\training\\'+dir):
        trainFiles.append(file)
        Category.append(dir)
        FileName.append(file)
        Path.append(dirpath+'\\training\\'+dir+'\\'+file)
        # Open a file: fileReader
        fileReader = open(dirpath+'\\training\\'+dir+'\\'+file,mode='r')
        # read all lines at once
        text = fileReader.read()
        # close the file
        fileReader.close()
        Text.append(text)
for dir in testDirs:
    for file in os.listdir(dirpath+'\\training\\'+dir):
        testFiles.append(file)
        Category.append(dir)
        FileName.append(file)
        Path.append(dirpath+'\\training\\'+dir+'\\'+file)
        # Open a file: fileReader
        fileReader = open(dirpath+'\\training\\'+dir+'\\'+file,mode='r')
        # read all lines at once
        text = fileReader.read()
        # close the file
        fileReader.close()
        Text.append(text.lower()) #get to lower case: pre-processing


# pre-process
for index in range(len(Text)):
    # tokenize
    tokenized=word_tokenize(Text[index])
    # remove stop words
    filtered_sentence = [w for w in tokenized if not w in stopWords]
    # stemming
    ps = PorterStemmer()
    for i in range(len(filtered_sentence)):
        filtered_sentence[i]=ps.stem(filtered_sentence[i])
    Text[index]=filtered_sentence

