import sys
import math
sys.path.append('../src')
import numpy as np
import data_io, params, SIF_embedding
from scipy.spatial.distance import pdist
def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())
def cosine(vec1,vec2):
    distance = pdist(np.vstack([vec1, vec2]), 'cosine')[0]
    return distance
def get_embedding(sentence,words,weight4ind,params,We):
    # load sentences
    xx, mm =data_io.sentences2idx(sentence, words)
    ww = data_io.seq2weight(xx, mm, weight4ind)  # get word weights
    # get SIF embedding
    em = SIF_embedding.SIF_embedding(We, xx, ww, params)  # embedding[i,:] is the embedding for sentence i
    return em
# input
# wordfile = '../data/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website
if __name__ == '__main__':
    wordfile = '../data/glove.6B.50d.txt' # word vector file, can be downloaded from GloVe website
    weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
    weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 0 # number of principal components to remove in SIF weighting scheme
    sentences = ['I like to play football', 'Football is my favorite game']
    sentence1=['Football I like to play football']
    sentence2=['Football is my favorite game Football']
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
    # load sentences
    # x, m = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    # w = data_io.seq2weight(x, m, weight4ind) # get word weights
    # set parameters
    params = params.params()
    params.rmpc = rmpc
    #
    # # set parameters
    # params = params.params()
    # params.rmpc = rmpc
    # # get SIF embedding
    # embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i
    embedding1=get_embedding(sentence1,words,weight4ind,params,We)
    embedding2=get_embedding(sentence2,words,weight4ind,params,We)
    embedding3=get_embedding(sentences,words,weight4ind,params,We)
    ss="A fair number of brave souls who upgraded their SI clock oscillator haveshared their experiences for this poll. Please send a brief message detailingyour experiences with the procedure. Top speed attained, CPU rated speed,add on cards and adapters, heat sinks, hour of usage per day, floppy diskfunctionality with 800 and 1.4 m floppies are especially requested.I will be summarizing in the next two days, so please add to the networkknowledge base if you have done the clock upgrade and haven't answered thispoll. Thanks."
    sss=[]
    sss.append(ss)
    embedding4 = get_embedding(sss, words, weight4ind, params, We)
    print(embedding1)
    wwww=We[words["football"]]
    print(words["football"])
    print(We[words["football"]])
    print("senteces:")
    print(sentences)
    print("The distance between football and sentences:")
    print(cosine(embedding1,We[words["football"]]))
    print("The distance between football and soccer:")
    print(cosine(We[words["soccer"]],We[words["football"]]))
    print("The distance between sentences and soccer:")
    print(cosine(embedding1,We[words["soccer"]]))
    print("The distance between sentences and apple:")
    print(cosine(embedding1,We[words["apple"]]))
    print("The distance between sentence1 and sentence2:")
    print(cosine(embedding1,embedding2))
    print("The distance between sentences and football:")
    print(cosine(embedding3,We[words["football"]]))
    print(cosine(embedding3,embedding3))