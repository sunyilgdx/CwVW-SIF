#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2018/7/11
#CwVW-SIF情感分类
import os
from examples import sif_embedding
import sklearn.datasets
import data_io, params, SIF_embedding
import numpy
from sklearn import svm
from sklearn import model_selection
from sklearn import *
import sys
from importlib import reload
import _multibytecodec
from sklearn.preprocessing import Imputer
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import math
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# import gensim
# import gensim.models
from nltk.corpus import stopwords
import nltk
english_stopwords = stopwords.words('english')
english_punctuations = [
    ',',
    '.',
    ':',
    ';',
    '?',
    '(',
    ')',
    '[',
    ']',
    '&',
    '!',
    '*',
    '@',
    '#',
    '$',
    '%',
    '-',
    '_',
    "'",
    '"'
    ' ']
def cleanLines(line):
    line = line.lower()
    words = nltk.word_tokenize(line)
    # words=line.split(english_punctuations)
    # words = line.split()
    clean_words = [
        w for w in words if (
            # w not in english_stopwords and w not in english_punctuations)]
            w not in english_punctuations)]
    clean_line = ""
    num = len(clean_words)
    if(num >= 1):
        for i in range(0, num - 1):
            clean_line += clean_words[i] + " "
        clean_line += clean_words[num - 1]
        return clean_line
    else:
        return ""


def get_cleanwords(line):
    line = line.lower()
    words = nltk.word_tokenize(line)
    clean_words = [
        w for w in words if (
            w not in english_stopwords and w not in english_punctuations)]
    return clean_words


def get_filtedLines(line="", cut_words=set()):
    words = line.split(' ')
    filted_words = [w for w in words if (w not in cut_words)]
    filted_line = ""
    num = len(filted_words)
    if (num >= 1):
        for i in range(0, num - 1):
            filted_line += filted_words[i] + " "
        filted_line += filted_words[num - 1]
        return filted_line
    else:
        return ""

def get_sentences_embedding(dataset,words,weight4ind,params,We, cut_words):
    sentences_embedding=[]
    for d in dataset:
        s=d
        s=s.replace('\n','')
        s = cleanLines(s)
        s = get_filtedLines(s, cut_words)
        sent=[]
        sent.append(s)
        em = sif_embedding.get_embedding(sent, words, weight4ind, params, We)
        sentences_embedding.append(em)
    return sentences_embedding

def turn2std(old_x):
    n=len(old_x)
    new_x=[]
    for i in range(0,n):
        new_x.append(old_x[i][0])
    return numpy.array(new_x)

def get_sif(dataset):
    wordfile = '../data/glove.6B.50d.txt'  # word vector file, can be downloaded from GloVe website
    weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'  # each line is a word and its frequency
    weightpara = 2.7e-4  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 0  # number of principal components to remove in SIF weighting scheme
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
    param = params.params()
    param.rmpc = rmpc
    sentence_embedding_all = get_sentences_embedding(dataset, words, weight4ind, param, We)
    # sentence_embedding_all = turn2std(sentence_embedding_all)  # 将矩阵转换为标准矩阵
    return  sentence_embedding_all

def get_dataset(kind="train"):
    dataset=[]
    target=[]
    N=10000#训练
    J=1000#调参
    M=2000#测试
    if(kind=='train'):
        # files = ["../data/train_pos.txt","../data/train_neg.txt"]
        with open("../data/train_pos.txt","r") as f:
            n=0
            for sent in f.readlines():
                dataset.append(sent)
                target.append(1)
                n+=1
                if(n>=N):
                    break
        f.close()

        with open("../data/train_neg.txt", "r") as f:
            n = 0
            for sent in f.readlines():
                dataset.append(sent)
                target.append(0)
                n += 1
                if (n >=N):
                    break
        f.close()
        return dataset,numpy.array(target)

    if (kind == 'ajust'):
        # files = ["../data/train_pos.txt","../data/train_neg.txt"]
        with open("../data/train_pos.txt", "r") as f:
            n = 0
            for sent in f.readlines():
                if(n>=N):
                    dataset.append(sent)
                    target.append(1)
                n += 1
                if (n >= N+J):
                    break
        f.close()

        with open("../data/train_neg.txt", "r") as f:
            n = 0
            for sent in f.readlines():
                if (n >= N):
                    dataset.append(sent)
                    target.append(0)
                n += 1
                if (n >= N+J):
                    break
        f.close()
        return dataset, numpy.array(target)

    if(kind=="test"):
        with open("../data/test_pos.txt","r") as f:
            n=0
            for sent in f.readlines():
                dataset.append(sent)
                target.append(1)
                n+=1
                if (n >=M):
                    break
        f.close()
        with open("../data/test_neg.txt", "r") as f:
            n = 0
            for sent in f.readlines():
                dataset.append(sent)
                target.append(0)
                n += 1
                if (n >=M):
                    break
        f.close()
        return dataset,numpy.array(target)
def get_variance(fres=[]):
    # 计算列表中数据的方差
    n = len(fres)
    if(n <= 1):
        return 0.0
    sum = 0.0  # 总和
    average = 0.0  # 均值
    variance = 0.0  # 方差

    for i in range(0, n):
        sum += fres[i]
    if(sum<=0.0):
        return 0.0
    average = sum / (float)(n)
    temp = 0.0
    for i in range(0, n):
        temp += (fres[i] - average) * (fres[i] - average)
    variance = temp / (float)(n) / average / average
    return variance

def get_word_variance(dataset,dataset_target, words={}, word2weight={}):
    num_target =2
    word2times = {}  # 统计每个单词在各个文章类型中出现新的次数
    word2freq = {}  # 统计每个单词在各个文章类型中出现新的次数概率
    word2var = {}  # 统计每个单词的概率均方差
    symbol = '.;:,()? '
    for i in range(0, len(dataset)):
        # 统计每个词在各个类型中出现的次数
        s = dataset[i]
        s = s.replace('\n', '')
        ss = get_cleanwords(s)
        target = dataset_target[i]  # 判断这些单词属于什么类型
        for word in ss:
            word_exist = {}
            if(word in words and word not in  word_exist):#只有词频单词表里有并且本篇文章第一次出现的
                if (word in words):
                    word_exist[word]=1
                    if(word in word2times):
                        word2times[word][target] += 1
                    else:
                        times = [0 for n in range(num_target)]
                        word2times[word] = times
                        word2times[word][target] += 1

    target_times = [0 for n in range(num_target)]
    for i in range(0, len(dataset_target)):
        # 计算每个类别的总的个数
        target_times[dataset_target[i]] += 1
    for word in word2times:
        times = word2times[word]
        n_times = len(times)
        if(n_times <= 1):
            word2var[word] = 0
        else:
            fres = []  # 单词word在每个类型出现的概率
            for j in range(0, len(times)):
                fre = (float)(times[j]) / (float)(target_times[j])
                fres.append(fre)
            #此处使用整体方差的方法
            variance = get_variance(fres)
            #此处使用MAX_VARIANCE的方法
            # variance=get_max_variance(fres)
            word2var[word] = variance

    #乘以平滑反频率的权重a/a+f
    for word in word2var:
        if(word in word2weight):
            word2var[word] *= word2weight[word]
        else:
            word2var[word] *= 0

    return word2var

def sort_by_value(dict={}):
    list = sorted(dict.items(), key=lambda d: d[1], reverse=False)
    return list

def get_cut_words(cut_word_var=[]):
    cut_words = []
    for word_var in cut_word_var:
        cut_words.append(word_var[0])
    return cut_words

def list2set(list1=[]):
    s=set()
    for i in list1:
        s.add(i)
    return s


'''
情感二分类
'''
if __name__ == '__main__':
    train_dataset,train_target=get_dataset("train")
    # ajust_dataset, ajust_target = get_dataset("ajust")
    test_dataset, test_target = get_dataset("test")
    wordfile = '../data/glove.6B.50d.txt'  # word vector file, can be downloaded from GloVe website
    weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'  # each line is a word and its frequency
    weightpara = 2.7e-4  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 0  # number of principal components to remove in SIF weighting scheme
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
    params = params.params()
    params.rmpc = rmpc

    word2var = get_word_variance(train_dataset, train_target,words, word2weight)
    sorted_word_var = sort_by_value(word2var)
    rr=[0.43,0.44,0.45,0.47,0.48,0.49,0.51,0.52,0.53,0.55,0.56,0.57]
    with open("../result/sentiment_analysis.txt", 'w') as f:
        f.write("裁剪率\t准确率\t\n")
        for i in range( (int)(len(sorted_word_var)*0), (int)(len(sorted_word_var)),(int)(len(sorted_word_var)*0.02)):
        # for j in rr:
        #     i=(int)(len(sorted_word_var)*j)
            cut_word_var = sorted_word_var[:i]
            cut_words = get_cut_words(cut_word_var)
            cut_words2=list2set(cut_words)
            sentence_embedding_train = get_sentences_embedding(train_dataset, words, weight4ind, params, We, cut_words2)
            # sentence_embedding_ajust = get_sentences_embedding(ajust_dataset, words, weight4ind, params, We, cut_words2)
            sentence_embedding_test = get_sentences_embedding(test_dataset, words, weight4ind, params, We, cut_words2)
            x_train = turn2std(sentence_embedding_train)
            # x_ajust = turn2std(sentence_embedding_ajust)
            x_test = turn2std(sentence_embedding_test)
            ratio = 0
            if (i == 0):
                ratio = 0
            else:
                ratio = i / len(sorted_word_var)
            # print(len(train_dataset))
            clf = svm.SVC(C=3, kernel='rbf', gamma=20, decision_function_shape='ovr')
            print("裁剪率:%f" % (ratio))
            clf.fit(x_train, train_target)
            print("训练集")
            print(clf.score(x_train,train_target))  # 精度

            # ACU=clf.score(x_ajust, ajust_target)
            ACU = clf.score(x_test, test_target)
            print("调参集")
            print(ACU)  # 精度
            f.write("%s\t%s\t\n" % (str(ratio), str(ACU)))