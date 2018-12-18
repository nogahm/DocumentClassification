import os
import nltk
import sklearn
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from os import listdir
from os.path import isfile, join
import sklearn.datasets as sk

#variables
path='d:\\documents\\users\\nogahm\\Downloads\\testCorpus'
dirpath=path
trainDirs=[]
testDirs=[]
trainFiles=[]
testFiles=[]
# for table
Category=[]
FileName=[]
Path=[]
Text=[]

classNumOfDocs=[]
docsInfo=[] #[fileName,cleanText,class

stopWords=set(stopwords.words('english'))

# get train and test dirs
trainDirs=os.listdir(dirpath+'\\training')
testDirs=os.listdir(dirpath+'\\test')

# pre-process
def cleanText(text):
    # tokenize
    tokenized = word_tokenize(text)
    # remove stop words
    filtered_sentence = [w for w in tokenized if not w in stopWords]
    # stemming
    ps = PorterStemmer()
    for i in range(len(filtered_sentence)):
        filtered_sentence[i] = ps.stem(filtered_sentence[i])
    return " ".join(filtered_sentence)

# get train and test dirs
for dir in trainDirs:
    for file in os.listdir(dirpath+'\\training\\'+dir):
        trainFiles.append(file)
        Category.append(dir)
        FileName.append(file)
        Path.append(dirpath+'\\training\\'+dir+'\\'+file)
        # Open a file: fileReader
        fileReader = open(dirpath+'\\training\\'+dir+'\\'+file,mode='r')
        # read all lines at once
        text = fileReader.read().lower()
        # close the file
        fileReader.close()

        cleanedText=cleanText(text)
        docsInfo.append([file,cleanedText,dir])

for dir in testDirs:
    for file in os.listdir(dirpath+'\\test\\'+dir):
        testFiles.append(file)
        Category.append(dir)
        FileName.append(file)
        Path.append(dirpath+'\\test\\'+dir+'\\'+file)
        # Open a file: fileReader
        fileReader = open(dirpath+'\\test\\'+dir+'\\'+file,mode='r')
        # read all lines at once
        text = fileReader.read().lower()
        # close the file
        fileReader.close()

        cleanedText = cleanText(text)
        docsInfo.append([file, cleanedText, dir])

#get number of classes and number of docs for each class
docInfoDF=pd.DataFrame(docsInfo, columns=['fileName','cleanText','class'])
allClasses=np.unique(docInfoDF['class'])
print ('# of categories: ',allClasses.size)
numOfFiles=[]
termsFreqPerClass=[]
for currClass in allClasses:
    temp=(docInfoDF['class']).value_counts()[currClass]
    numOfFiles.append([currClass, temp])
    #get terms distibution
    currFiles=docInfoDF.loc[docInfoDF['class']==currClass]
    termsFreq=pd.Series(" ".join(currFiles['cleanText']).split()).value_counts()[:10]
    termsFreqPerClass.append([currClass] + termsFreq.index)

numOfFiles=pd.DataFrame(numOfFiles,columns=['class','#of docs'])
print(numOfFiles)
print(termsFreqPerClass)

