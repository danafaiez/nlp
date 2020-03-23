#contextuazlized BERT 2D plot
from sklearn.utils.extmath import softmax
from random import randint
from math import sqrt
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt; plt.style.use('ggplot')
#from matplotlib.patches import Rectangle
import pandas as pd
#from sklearn.metrics import euclidean_distances
import ot
from pyemd import emd, emd_with_flow
import gensim
from MulticoreTSNE import MulticoreTSNE as TSNE
import gensim.models.keyedvectors as word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import matplotlib
from time import time
import os
import itertools
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import string
import sys
import numpy as np
#import bmd as bmd
import bmd3 as bmd
from sklearn.decomposition import PCA
from scipy import spatial
 
b=bmd.Bert_Evaluator(max_seq_len=512)

words = [('my', 'all' , 'also' ,'from','make', 'with','apple','orange','fruits','banana','grapes','plants','berries','pear'),('with', 'just', 'it', 'the', 'anything', 'but','apartment','city','bedroom','bathroom','hall','bed','decor','architecture'),('is', 'on', 'for', 'basically', 'me', 'take', 'still','it','saturday','thursday','tuesday','wednesday','sunday','holiday','monday','friday')]
words_labels = ['my', 'all' , 'also' ,'from','make', 'with', 'just', 'it', 'the', 'anything', 'but', 'is', 'on', 'for', 'basically', 'me', 'take', 'still','apple','orange','fruits','banana','grapes','plants','berries','pear','apartment','city','bedroom','bathroom','hall','bed','decor','architecture','saturday','thursday','tuesday','wednesday','sunday','holiday','monday','friday']

sentences = ["My new diet includes all sorts of fruits and also food from plants; from apple and orange, to banana and pear. I make my oats in the morning with berries and grapes.",
             "I just moved to the city and I'm obsessed with the architecture here. I got an apartment with one bedroom and a bed in it, one bathroom, and giant hall. I made the decor in my place all nice but I wouldn't say it is anything fancy.",
             "Here is my daily plan: I work on monday, tuesday, and wednesday from the office. On thursday, I usually stay home but still work. friday and saturday and sunday are basically Holiday for me so I take it easy."]

out = b([sentences[0]])
embedding_length = len(out[0,1,:]) #as an example, get the length of the 1st word appearing in the sentence[0]
vec = []
word_ordered = []
for i in range(3): 
  token = b.tokenizer.tokenize(sentences[i]) #BERT's tokens
  out = b([sentences[i]])  
  print("tokens for sent " + str(i) + " is: " + str(token))
  for w in words[i]:
    word_ordered.append(w)
    word_token = b.tokenizer.tokenize(w)
    count = len(word_token)
    #print("word: "+ str(w) + " has token: " + str(word_token) + " with lenth: " + str(count))
    if w in token: ### The  +1  is there becasue the 0th case in out belongs to the cls token.
      vec.append(out[0,token.index(w)+1,:])
    else: #i.e. if the length of the tokenization>1
      temp_vec = np.zeros([1,embedding_length])
      for j in range(len(word_token)):
        #print("word_token[j]:",word_token[j])
        #print("token.index((word_token[j])):",token.index((word_token[j])))
        temp_vec+=out[0,token.index((word_token[j]))+1,:]
      temp_vec=temp_vec/(len(word_token))
      vec.append(temp_vec)
softmax_vec = softmax(vec)
#print("len of vec:",len(vec))

#vec = np.array(vec)
# Project the data in 2D
vec_pca = PCA().fit_transform(vec)
softmax_vec_pca = PCA().fit_transform(softmax_vec)

plt.subplot(111)
colors=[]
colors=["b"]*len(words[0]) 
colors.extend(["g"]*len(words[1]))
colors.extend(["purple"]*len(words[2]))
colors = itertools.cycle(colors)
#plt.scatter(Z_pca[:, 0], Z_2d[:, 1], c=color_index)

for i in range(0,len(word_ordered)):  
  plt.scatter(vec_pca[i, 0], vec_pca[i, 1],marker = 'x',color=next(colors),label='%d : %s' %(i, word_ordered[i]))
  plt.annotate(i, # this is the text
               (vec_pca[i, 0],vec_pca[i, 1]), # this is the point to label
               textcoords="offset points", # how to position the text
               xytext=(0,4), # distance from text to points (x,y)
               ha='center', # horizontal alignment can be left, right or center
               size=7) 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
          ncol=3, fancybox=True, shadow=True)  
plt.title("contextual BERT", loc='center')
plt.show()

plt.subplot(111)
for i in range(0,len(word_ordered)):  
  plt.scatter(softmax_vec_pca[i, 0], softmax_vec_pca[i, 1],marker = 'x',color=next(colors),label='%d : %s' %(i, word_ordered[i]))
  plt.annotate(i, # this is the text
               (softmax_vec_pca[i, 0],softmax_vec_pca[i, 1]), # this is the point to label
               textcoords="offset points", # how to position the text
               xytext=(0,4), # distance from text to points (x,y)
               ha='center', # horizontal alignment can be left, right or center
               size=7) 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
          ncol=3, fancybox=True, shadow=True)
plt.title("softmax contextual BERT", loc='center')
plt.show()
