#import pdb
from random import randint
from math import sqrt
from itertools import product
#import os.path
from collections import Counter
import matplotlib.pyplot as plt; plt.style.use('ggplot')
#from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
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
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import os
import itertools
from nltk.tokenize import word_tokenize
import string
stop_words = stopwords.words('english')
import sentence_examples_file as examples
porter = nltk.PorterStemmer()
from numpy import linalg as LA

def load_embddeing_model():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)
    model.init_sims(replace=True)
    #model.save("mymodel")
    vocabulary = set(model.vocab)
    return model, vocabulary

class EMD:

    def preprocess_text(self,sent1,sent2):
        table = str.maketrans('', '', string.punctuation.replace("'",""))
        sents = [sent1,sent2] 
        sents_tokens =[ ]
        for s in sents:
            strip_s=[w.translate(table) for w in s.lower().split()]
            strip_s_nostopw=[w for w in strip_s if w not in stop_words]
            sents_tokens.append(strip_s_nostopw)
        return sents_tokens

    def missing_words(self,sents_tokens,vocabulary):
        sent1_tokens = [word for word in sents_tokens[0] if word in vocabulary]
        sent2_tokens = [word for word in sents_tokens[1] if word in vocabulary]
        return sent1_tokens, sent2_tokens

    def get_signitures(self,sent1_tokens, sent2_tokens):
        sent1_dict = dict(Counter(sent1_tokens))
        sent2_dict = dict(Counter(sent2_tokens))
        w2v_p1_signature=[]
        w2v_p2_signature=[]
        resultList= []
        for tokens1 in sent1_tokens:
            if tokens1 not in resultList:
                resultList.append(tokens1)
        for tokens2 in sent2_tokens:
            if tokens2 not in resultList:
                resultList.append(tokens2)
        for w in resultList:
            if w in sent1_dict:
                w2v_p1_signature.append(sent1_dict[w])
            else:
                w2v_p1_signature.append(0)
        w2v_p1_signature = [i/len(sent1_tokens) for i in w2v_p1_signature]
        print("*resultList:\n",resultList)

        for w in resultList:
            if w in sent2_dict:
                w2v_p2_signature.append(sent2_dict[w])
            else:
                w2v_p2_signature.append(0)
        w2v_p2_signature = [i/len(sent2_tokens) for i in w2v_p2_signature]
        #print("sig for sent1_tokens:\n " + str(sent1_tokens) + "is:\n " +str(w2v_p1_signature))

        #print("sig for sent2_tokens:\n " + str(sent2_tokens) + "is:\n " +str(w2v_p2_signature))
        return w2v_p1_signature, w2v_p2_signature, resultList

    def calculate_emd(self,signature_1, signature_2, distance_matrix):
        first_signature = np.array(signature_1, dtype=np.double)
        second_signature = np.array(signature_2, dtype=np.double)
        distances = np.array(distance_matrix, dtype=np.double)
        emd, flow = emd_with_flow(first_signature, second_signature, distances)
        flow = np.array(flow)
        return emd, flow

    def get_emd(self,sent1_tokens, sent2_tokens,model, vocabulary,normalized_dist_entries):
        w2v_p1_signature,w2v_p2_signature,resultList = self.get_signitures(sent1_tokens, sent2_tokens)
        w2v_distances = ot.dist([list(model[word]) for word in resultList],[list(model[word]) for word in resultList],metric='euclidean')
        w2v_emd, w2v_flow = self.calculate_emd(w2v_p1_signature, w2v_p2_signature,w2v_distances)
	return w2v_emd , w2v_flow,  w2v_distances
###################################################################################################
class Plotter:
    def __init__(self):
        self.test = "test"
        self.fontsize = 7
        self.format = """
            title = imgs[i]["title"]
            xticks = imgs[i]["xticks"]
            xtick_labels = imgs[i]["xtick_labels"]
            yticks = imgs[i]["yticks"]
            ytick_labels = imgs[i]["ytick_labels"]
            image = imgs[i]["image"]
            """
        self.w, self.h = 10,10
        self.figsize = (10,7)
        self.textwrap = True
    def build_figure(self, imgs,cols,rows):
        #imgs = list of dicts
        assert cols*rows <= len(imgs)
        w, h = self.w,self.h
        fig = plt.figure(figsize = self.figsize)
        # ax enables access to manipulate each of subplots
        ax = []

        for k in range(cols*rows):
            # create subplot and append to ax
            ax.append( fig.add_subplot(rows, cols, k+1) )
            # get tick indices and labels
            title = imgs[k]["title"]
            xticks = imgs[k]["xticks"]
            xtick_labels = imgs[k]["xtick_labels"]
            yticks = imgs[k]["yticks"]
            ytick_labels = imgs[k]["ytick_labels"]

            if self.textwrap == True:
                for i,label in enumerate(ytick_labels):
                    ytick_labels[i] = self.insert_newline_every(6, label)
                for i,label in enumerate(xtick_labels):
                    xtick_labels[i] = self.insert_newline_every(6, label)

            ax[-1].set_title("subplot ax#"+str(k)+": " + str(title), fontsize = self.fontsize)  # set title
            #ax[-1].set_xlabel("Bert input text, output vectors of contexts by position",fontsize=8) #str(tokens2),fontsiz
            #ax[-1].set_ylabel("Bert input text, output vectors of contexts by position",fontsize=8) #str(list(reversed(to
            ax[-1].set_xticks(xticks)
            ax[-1].set_xticklabels(xtick_labels, fontsize=self.fontsize-1)
            ax[-1].set_yticks(yticks)
            ax[-1].set_yticklabels(ytick_labels, fontsize=self.fontsize-1)
            norm = matplotlib.colors.Normalize(vmin = 0, vmax = 1, clip = False)
            for tick in ax[-1].get_xticklabels():
                tick.set_rotation(90)
            plt.imshow(imgs[k]["image"],norm=norm)  #, alpha=0.99)
            plt.colorbar()
        fig.subplots_adjust(hspace=.5, bottom=.27)
        plt.show()
    def insert_newline_every(self,k,string):
        words = string.split(" ")
        for i,w in enumerate(words):
            if i%k == 0 and i >0:
                words.insert(i,"\n")
        return " ".join(words)

###################################################################################################
def get_multi_sentence_distances(doc1,doc2,plot,normalized_dist_entries):
    emd_matrix = np.zeros((len(doc1), len(doc2)))
    (xticks, xtick_labels), (yticks,ytick_labels) = ([],[]),([],[])
    table = str.maketrans('', '', string.punctuation.replace("'",""))
    #pdb.set_trace()
    model, vocabulary = load_embddeing_model()
    #pdb.set_trace()
    emd = EMD()
    for i_x,sent1 in enumerate(doc1):
        print("sent1:",sent1)
        for i_y,sent2 in enumerate(doc2):
            print("sent2:",sent2)
            sents_tokens = emd.preprocess_text(sent1,sent2)
            sent1_tokens, sent2_tokens = emd.missing_words(sents_tokens,vocabulary) 
            dd, flow, dist_matrix = emd.get_emd(sent1_tokens, sent2_tokens,model,vocabulary,normalized_dist_entries)
            emd_matrix[i_y, i_x] = dd
            
            if plot == True:
                xticks.append(i_x); yticks.append(i_y)
                ##labels_X labels_Y
                sent11=[w.translate(table) for w in sent1.lower().split()]
                sent1_not_incl=""
                
                sent22=[w.translate(table) for w in sent2.lower().split()]
                sent2_not_incl=""
		for everyword in sent11:
                    if everyword not in sent1_tokens:
                        e = everyword.replace(everyword,'(' + everyword + ')')
                        sent1_not_incl=sent1_not_incl+e+" "
                    else:
                        sent1_not_incl=sent1_not_incl+everyword +" "
                    xtick_labels.append(sent1_not_incl)
                for everyword in sent22:
                    if everyword not in sent2_tokens:
                        e = everyword.replace(everyword,'(' + everyword + ')')
                        sent2_not_incl=sent2_not_incl+e+" "
                    else:
                        sent2_not_incl=sent2_not_incl+everyword +" "
                    ytick_labels.append(sent2_not_incl) 
                """    
                if ytick_labels[i_y] == "" and sent1 == sent2:
                    ytick_labels.append(sent1_not_incl)
                elif ytick_labels[i_y] == "" and sent1 != sent2:
		    ##labels_Y
                    sent22=[w.translate(table) for w in sent2.lower().split()]
                    sent2_not_incl=""
                    for everyword in sent22:
                        if everyword not in sent2_tokens:
                            e = everyword.replace(everyword,'(' + everyword + ')')
                            sent2_not_incl=sent2_not_incl+e+" "
                        else:
                            sent2_not_incl=sent2_not_incl+everyword +" "
                    ytick_labels.append(sent2_not_incl)                
                  """

    
    expected_emd_matrix = np.multiply(flow,dist_matrix)
    expected_emd = emd_matrix.sum()
    print("*expected_emd:\n",expected_emd)
    print("*emd_matrix:\n")
    #pd.DataFrame(emd_matrix).to_csv("/Users/danafaiez/Desktop/emd_matrix.csv")
    print("*flow:\n",flow)
    #pd.DataFrame(flow).to_csv("/Users/danafaiez/Desktop/flow.csv")
    print("*dist_matrix:\n", dist_matrix)
    #pd.DataFrame(dist_matrix).to_csv("/Users/danafaiez/Desktop/dist_matrix.csv")
    print("emd:\n",emd_matrix)
    #pd.DataFrame(emd_matrix).to_csv("/Users/danafaiez/Desktop/emd.csv")
    
    if plot == True:
        p = Plotter()
        title = "BMD distance values for various sentences"
        img_dicts = [{"title":title,"image":emd_matrix,"xticks":xticks,"xtick_labels":xtick_labels,"yticks":yticks,"ytick_labels":ytick_labels}]
        p.build_figure(img_dicts,1,1)
        """
        if return_labels == True:
            emd_matrix, (xticks, xtick_labels), (yticks,ytick_labels)
        elif return_labels == None:
            None
        else:
            emd_matrix
        """
    return emd_matrix

if __name__ == "__main__":
    get_multi_sentence_distances(doc1=examples.set7() ,doc2=examples.set8(),plot = False,normalized_dist_entries = True) 
    #get_multi_sentence_distances(np.array(["hello"]),np.array(["hi"]),plot = False,normalized_dist_entries = True) 
    #get_multi_sentence_distances(np.array(["hello","hi"]) ,np.array(["hello","hi"]),plot = False,normalized_dist_entries = False) 

