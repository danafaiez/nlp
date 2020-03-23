##this is human rated code but with the two context given to BERT separately instead.
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

def load_embddeing_model():
  model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)
  model.init_sims(replace=True)
  vocabulary = set(model.vocab)
  return model, vocabulary
model, vocabulary = load_embddeing_model()

file = "ratings.txt" #"5test_human_rate_texts.txt"  "2test_human_rate_texts.txt"  
num_lines = sum(1 for line in open(file))
Data_matrix = np.zeros([num_lines, 7]).astype(object)
Data_matrix[0][:] = ['word 1', 'word 2', 'Human rating', 'BERT Cosine', 'W2V Cosine', 'BERT Euc.', 'W2V Euc.']
vector_matrix = np.zeros([num_lines+1, 6]).astype(object)
vector_matrix[0][:] = ['word 1', 'word 2', 'word 1 W2V', 'word 2 W2V', 'word 1 BERT','word 1 BERT']
translator=str.maketrans('','',string.punctuation)
with open(file, "r") as f:
  missed = 0
  case = 0
  paragraphs = [line for line in f]
  #print("len(paragraphs):",len(paragraphs))
  

  for t in range(len(paragraphs)-1):
    texts = paragraphs[t]
    
    
    ##### splitting, getting rid of punctuations, getting the raw sent for BERT and WMD input, target words, and average human rating of similarity between the target words
    #print("\n:",texts)
    split_with_punc = texts.split() ### I took our the .lower() here
    #print("split_with_punc[5:len(split_with_punc)-11]:",split_with_punc[5:len(split_with_punc)-11])
    #print("split_with_punc:",split_with_punc)
    sentence0_with_punc = '' 
    sentence1_with_punc = '' 
    for w in split_with_punc[5:120-13]:
      sentence0_with_punc += w + ' ' ### This will be in input 0 for bert 
    #print("sentence0_with_punc:",sentence0_with_punc)   
    for w in split_with_punc[120-13:len(split_with_punc)-11]:
      sentence1_with_punc += w + ' ' ### This will be in input 1 for bert a
    #print("split_with_punc:",split_with_punc) 
    #print("len(split_with_punc):",len(split_with_punc))
    ### Getting the average human rate from the text:
    human_ave_rate = split_with_punc[len(split_with_punc)-11]
    ### Taking out punctuations:
    ##texts = texts.translate(translator) #i took out .lower() here
    ### List of words without punctuations
    split = texts.split()
    #print("len(split):",len(split))
    #print("split:",split)
    target_words = []
    target_words.append(split[1].lower())
    target_words.append(split[3].lower())
    #print("target words are: ",target_words)
    
    
    #Only go forward (at least for now) if the target words are in vocabulary of w2v and tokenization of bert is exactly the same as the original word
    if target_words[0] in vocabulary and target_words[1] in vocabulary:   
      vec = [] #for the csv file
      vec.append(target_words[0])
      vec.append(target_words[1])
      case += 1
      ##### Getting BERT and Word2Vec EMBEDDING
      #out = b([sentence_with_punc])
      out0 = b([sentence0_with_punc])
      out1 = b([sentence1_with_punc])
      embedding_length = len(out0[0,1,:]) #as an example, get the length of the 1st word appearing in the sentence[0]
      BERT_vec = []
      W2V_vec = []
      W2V_vec.append(model[target_words[0]])
      W2V_vec.append(model[target_words[1]])
      
      for i in range(len(target_words)):
        TargetWord = target_words[i]
        if i==0:
          out = out0
          token = b.tokenizer.tokenize(sentence0_with_punc) #BERT's tokens
          #print("Sent0 Token:",token) 
        else: #i.e. if i==1:
          out = out1
          token = b.tokenizer.tokenize(sentence1_with_punc) #BERT's tokens
          #print("Sent1 Token:",token) 
        if TargetWord in token: ### The  +1  is there becasue the 0th case in out belongs to the cls token. 
          #print("IF TargetWord:",TargetWord) 
          BERT_vec.append(out[0,token.index(TargetWord)+1,:])
        else: #i.e. if the length of the tokenization>1
          #print("TargetWord with tokenization of >1:",TargetWord) 
          word_token = b.tokenizer.tokenize(TargetWord)
          temp_vec = np.zeros([1,embedding_length])
          for j in range(len(word_token)):
            #print("word_token[j]:",word_token[j])
            #print("token.index((word_token[j])):",token.index((word_token[j])))
            temp_vec+=out[0,token.index((word_token[j]))+1,:]
          temp_vec=temp_vec/(len(word_token))
          BERT_vec.append(temp_vec)
      #print("BERT_vec[0]:",BERT_vec[0])
      #print("BERT_vec[1]:",BERT_vec[1])


      

      ##### Human rating
      ### Scale is from 0 to 10. the higher the rating, the more similar thr words are.
      #print("human_ave_rate:",human_ave_rate)
      vec.append(human_ave_rate)
      #save the words and their embedding vectors
      vector_matrix[case][0:2] = target_words
      vector_matrix[case][2:4] = W2V_vec
      vector_matrix[case][4:6] = BERT_vec

      ##### Getting Cosine distance
      ### Cosine = 1 means the words are most similar. 
      BERT_cosine_similarity = 1 - spatial.distance.cosine(BERT_vec[0], BERT_vec[1])
      #print("BERT Cosine_similarity between target word '" + str(target_words[0]) + "' and '" +str(target_words[1]) + "' is = " + str(BERT_cosine_similarity))
      vec.append(BERT_cosine_similarity)
      W2V_cosine_similarity = 1 - spatial.distance.cosine(W2V_vec[0], W2V_vec[1])
      #print("W2V Cosine_similarity between target word '" + str(target_words[0]) + "' and '" +str(target_words[1]) + "' is = " + str(W2V_cosine_similarity))
      vec.append(W2V_cosine_similarity)
      

      ##### Getting Euclidean distance
      ### Note: the higher the Euclidean distance between words, the less similar they are.
      BERT_euclidean = np.linalg.norm(np.array(BERT_vec[0])-np.array(BERT_vec[1]))
      #print("BERT Euclidean distance between target word '" + str(target_words[0]) + "' and '" +str(target_words[1]) + "' is = " + str(BERT_euclidean))
      vec.append(BERT_euclidean)
      W2V_euclidean = np.linalg.norm(np.array(W2V_vec[0])-np.array(W2V_vec[1]))
      #print("W2V Euclidean distance between target word '" + str(target_words[0]) + "' and '" +str(target_words[1]) + "' is = " + str(W2V_euclidean))
      vec.append(W2V_euclidean)

      Data_matrix[case][:]=vec

    else:
      #print("either of these words are not in W2V vovab: " + str(target_words) + ".")
      missed +=1
    
    print("\n")

#normalizing Eucl. distance:
min_bert_euc=min(i for i in Data_matrix[1:,5] if i > 0) 
min_w2v_euc=min(i for i in Data_matrix[1:,6] if i > 0)
print("min bert euc:",  min_bert_euc )    
print("min bert euc:",  min_w2v_euc)

#pd.DataFrame(Data_matrix).to_csv("/content/nlp/Human_rated_cos_euc_Data_matrix.csv")

for k in range(1,len(Data_matrix[:])):
  Data_matrix[k,5] = (Data_matrix[k,5]-min_bert_euc)/(max(Data_matrix[1:,5]) - min_bert_euc) #BERT Eucl
  Data_matrix[k,6] = (Data_matrix[k,6]-min_w2v_euc)/(max(Data_matrix[1:,6]) - min_w2v_euc) #W2V Eucl

pd.DataFrame(Data_matrix).to_csv("/content/nlp/Human_rated_cos_norm_euc_Data_matrix.csv")
pd.DataFrame(vector_matrix).to_csv("/content/nlp/Human_rated_vector_matrix.csv")
print(str(missed) + " word pairs were missed.")
