##non-contextual BERT (its softmax) and (of course W2V) plots of words
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
import bmd as bmd

from sklearn.decomposition import PCA


words = [('my', 'all' , 'also' ,'from','make', 'with', 'just', 'it', 'the', 'anything', 'but', 'is', 'on', 'for', 'basically', 'me', 'take', 'still'),('apple','orange','fruits','banana','grapes','plants','berries','pear'),('apartment','city','bedroom','bathroom','hall','bed','decor','architecture'),('saturday','thursday','tuesday','wednesday','sunday','holiday','monday','friday')]
words_labels = ['my', 'all' , 'also' ,'from','make', 'with', 'just', 'it', 'the', 'anything', 'but', 'is', 'on', 'for', 'basically', 'me', 'take', 'still','apple','orange','fruits','banana','grapes','plants','berries','pear','apartment','city','bedroom','bathroom','hall','bed','decor','architecture','saturday','thursday','tuesday','wednesday','sunday','holiday','monday','friday']

######### Word2Vec
def load_embddeing_model():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)
    model.init_sims(replace=True)
    vocabulary = set(model.vocab)
    return model, vocabulary
model, vocabulary = load_embddeing_model()


###making sure every word is in vocab
#for i in words_labels:
#  if i not in vocabulary:
#    print("the word " + str(i) + " is not in the vocab.")


word_in_vocab = [word for word in words_labels if word in vocabulary]
embedding_length = len(model[word_in_vocab[0]])
if len(word_in_vocab) != len(words_labels):
  a = (set(words_labels).difference(set(word_in_vocab)))
  print("some worked were missing from vocabulary of W2V model and the missing word(s) is(are):\n",a)
X_w2v = np.zeros([len(word_in_vocab),embedding_length])
for i in range(len(word_in_vocab)):
    X_w2v[i,:] = model[word_in_vocab[i]]

Z_w2v = PCA().fit_transform(X_w2v)

######### BERT Visualize the data
plt.subplot(111)
colors=[]
colors=["r"]*len(words[0])
colors.extend(["b"]*len(words[1]))
colors.extend(["g"]*len(words[2]))
colors.extend(["purple"]*len(words[3]))
colors = itertools.cycle(colors)
#plt.scatter(Z_w2v[:, 0], Z_w2v[:, 1], c=color_index)
for i in range(0,len(word_in_vocab)):  
  plt.scatter(Z_w2v[i, 0], Z_w2v[i, 1],marker = 'o',color=next(colors),label='%d : %s' %(i, word_in_vocab[i]))
  plt.annotate(i, # this is the text
               (Z_w2v[i, 0],Z_w2v[i, 1]), # this is the point to label
               textcoords="offset points", # how to position the text
               xytext=(0,0), # distance from text to points (x,y)
               ha='center', # horizontal alignment can be left, right or center
               size=8) 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
          ncol=3, fancybox=True, shadow=True)  
plt.title("non-contextual Word2vec", loc='center')
#plt.legend()
plt.show()


######### BERT
b = bmd.Bert_Evaluator(max_seq_len = 512)
###checking to see if all tokenization is of length one:
#for word in words_labels:
#  if word ==  b.tokenizer.tokenize(word)[0]:
#    if len(b.tokenizer.tokenize(word)) !=1:
#      print("the word " + str(word) + "has tokenization: " + str(b.tokenizer.tokenize(word)))

#getting the embeddings
vector_list = []  
for word in words_labels:
  output = b([word])
  vector_list.append(output[0,1,:])
vector_list = np.array(vector_list)
softmax_vector_list = softmax(vector_list)

Z_pca = PCA().fit_transform(vector_list)
softmax_Z_pca = PCA().fit_transform(softmax_vector_list)

######### BERT Visualize the data
colors=[]
colors=["r"]*len(words[0])
colors.extend(["b"]*len(words[1]))
colors.extend(["g"]*len(words[2]))
colors.extend(["purple"]*len(words[3]))
colors = itertools.cycle(colors)
plt.subplot(111)
#plt.scatter(Z_pca[:, 0], Z_2d[:, 1], c=color_index)
for i in range(0,len(words_labels)):  
  plt.scatter(Z_pca[i, 0], Z_pca[i, 1],marker = 'x',color=next(colors),label='%d : %s' %(i, words_labels[i]))
  plt.annotate(i, # this is the text
               (Z_pca[i, 0],Z_pca[i, 1]), # this is the point to label
               textcoords="offset points", # how to position the text
               xytext=(0,4), # distance from text to points (x,y)
               ha='center', # horizontal alignment can be left, right or center
               size=7) 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
          ncol=3, fancybox=True, shadow=True)  
plt.title("non-contextual BERT", loc='center')
plt.show()

plt.subplot(111)
for i in range(0,len(words_labels)):  
  plt.scatter(softmax_Z_pca[i, 0], softmax_Z_pca[i, 1],marker = 'x',color=next(colors),label='%d : %s' %(i, words_labels[i]))
  plt.annotate(i, # this is the text
               (softmax_Z_pca[i, 0],softmax_Z_pca[i, 1]), # this is the point to label
               textcoords="offset points", # how to position the text
               xytext=(0,4), # distance from text to points (x,y)
               ha='center', # horizontal alignment can be left, right or center
               size=7) 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
          ncol=3, fancybox=True, shadow=True)
plt.title("softmax non-contextual BERT", loc='center')
plt.show()



