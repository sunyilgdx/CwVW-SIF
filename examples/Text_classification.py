#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2018/6/28
#CwVW-SIF文本分类
import os
from examples import sif_embedding
import sklearn.datasets
import data_io
import params
import SIF_embedding
import numpy
from sklearn import svm
from sklearn import model_selection
from sklearn import *
import sys
from importlib import reload
import _multibytecodec
from sklearn.preprocessing import Imputer
import re
import string
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
            w not in english_stopwords and w not in english_punctuations)]
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


def get_sentences_embedding(dataset, words, weight4ind, params, We, cut_words):
    sentences_embedding = []
    blank = "BLANK"
    # i = 0
    for d in dataset.data:
        # print(i)
        s = d
        s = s.replace('\n', '')
        s = cleanLines(s)
        s = get_filtedLines(s, cut_words)
        sent = []
        sent.append(s)

        em = sif_embedding.get_embedding(sent, words, weight4ind, params, We)
        sentences_embedding.append(em)
        # i += 1
        # if(i==1000):
        #     break
    return sentences_embedding


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

def get_max_variance(fres=[]):
    #计算一组数据中两两方差中的最大数
    n = len(fres)
    if (n <= 1):
        return 0.0
    max=-99999.0
    for i in range(0,n):
        for j in range(i+1,n):
            # temp_list=[]
            # temp_list.append()
            temp_variance=get_variance([fres[i],fres[j]])
            if(temp_variance>max):
                max=temp_variance
    return max


def get_word_variance(dataset, words, word2weight={}):
    num_target = len(dataset.target_names)
    word2times = {}  # 统计每个单词在各个文章类型中出现新的次数
    word2freq = {}  # 统计每个单词在各个文章类型中出现新的次数概率
    word2var = {}  # 统计每个单词的概率均方差
    symbol = '.;:,()? '
    for i in range(0, len(dataset.data)):
        # 统计每个词在各个类型中出现的次数
        s = dataset.data[i]
        s = s.replace('\n', '')
        ss = get_cleanwords(s)
        target = dataset.target[i]  # 判断这些单词属于什么类型
        for word in ss:
            word_exist = {}
            if(word in words and word not in word_exist):#只有词频单词表里有并且本篇文章第一次出现的
                if (word in words):
                    # word_exist[word]=1
                    if(word in word2times):
                        word2times[word][target] += 1
                    else:
                        times = [0 for n in range(num_target)]
                        word2times[word] = times
                        word2times[word][target] += 1

    target_times = [0 for n in range(num_target)]
    for i in range(0, len(dataset.target)):
        # 计算每个类别的总的个数
        target_times[dataset.target[i]] += 1
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


def turn2std(old_x):
    n=len(old_x)
    new_x=[]
    for i in range(0,n):
        new_x.append(old_x[i][0])
    return numpy.array(new_x)

def check_nan(data):
    n = len(data)
    for i in range(0, n):
        m = len(data[i])
        if (not numpy.isfinite(data[i]).all()):
            print(i)
        for j in range(0, m):
            if(data[i][j] == numpy.nan):
                print(i, j)
            if (not numpy.isfinite(data[i][j]).all()):
                xxx = data[i][j]
                print(i, j)


def sort_by_value(dict={}):
    list = sorted(dict.items(), key=lambda d: d[1], reverse=False)
    return list

def sort_by_value2(dict={}):
    list = sorted(dict.items(), key=lambda d: d[1], reverse=True)
    return list

def get_cut_words(cut_word_var=[]):
    cut_words = []
    for word_var in cut_word_var:
        cut_words.append(word_var[0])
    return cut_words


def get_accuracy(y1, y2):
    count = 0.0
    for i in range(0, len(y1)):
        if(y1[i] == y2[i]):
            count += 1.0
    return count / float(len(y1))

def list2set(list1=[]):
    s=set()
    for i in list1:
        s.add(i)
    return s

def get_key_words(dataset,word2weight,word2var):
    key_words=[]
    key_words2 = []
    for data in dataset.data:
        words_fvw={}
        words_fvw2={}
        words_fre={}
        s = data
        s = s.replace('\n', ' ')
        ss = cleanLines(s)
        sss = get_cleanwords(ss)
        # words = s.split(' ')
        for word in sss:
            if(word in words_fre):
                words_fre[word] += 1
            else:
                words_fre[word] = 1
        for word in words_fre:
            if(word in word2weight and word in word2var):
                words_fvw[word] = words_fre[word] * word2weight[word] * word2var[word]
                words_fvw2[word] = words_fre[word] * word2weight[word]
        key_words.append(words_fvw)
        key_words2.append(words_fvw2)
    return key_words,key_words2

def sorted_key_words(key_words):
    sorted_key_words=[]
    for words in key_words:
        # sss = []
        key_value=sort_by_value2(words)
        # for k in key:
        #     t = []
        #     t.append(k)
        #     t.append(words[k])
        #     sss.append(t)
        # sorted_key_words.append(sss)
        sorted_key_words.append(key_value)
    return  sorted_key_words



if __name__ == '__main__':
    # 获得测试数据
    # 加载数据集
    # categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.baseball']
    train_dataset = sklearn.datasets.fetch_20newsgroups(
        subset='train', categories=categories, shuffle=True, random_state=42)
    test_dataset = sklearn.datasets.fetch_20newsgroups(
        subset='test', categories=categories, shuffle=True, random_state=42)
    # word vector file, can be downloaded from GloVe website
    wordfile = '../data/glove.6B.50d.txt'
    # each line is a word and its frequency
    weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'
    # the parameter in the SIF weighting scheme, usually in the range [3e-5,
    # 3e-3]
    weightpara = 2.7e-4
    rmpc = 0  # number of principal components to remove in SIF weighting scheme
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    # word2weight['str'] is the weight for the word 'str'
    word2weight = data_io.getWordWeight(weightfile, weightpara)
    # weight4ind[i] is the weight for the i-th word
    weight4ind = data_io.getWeight(words, word2weight)
    word2var = get_word_variance(train_dataset, words, word2weight)
    sorted_word_var = sort_by_value(word2var)
    params = params.params()
    params.rmpc = rmpc
    key_words,key_words2=get_key_words(train_dataset,word2weight,word2var)
    sorted=sorted_key_words(key_words)
    print(len(key_words))



    with open("../result/var.txt", 'w') as f:
        f.write("裁剪率\t准确率\t\n")

        for i in range(0, len(sorted_word_var), 200):
            cut_word_var = sorted_word_var[:i]
            cut_words = get_cut_words(cut_word_var)
            cut_words2 = list2set(cut_words)

            sentence_embedding_train = get_sentences_embedding(train_dataset, words, weight4ind, params, We, cut_words2)
            sentence_embedding_test = get_sentences_embedding(test_dataset, words, weight4ind, params, We, cut_words2)
            print(len(sentence_embedding_train))
            # 设置评估数据
            ACU = P = R = F1 = 0.0
            x_train = turn2std(sentence_embedding_train)
            x_test = turn2std(sentence_embedding_test)
            print(len(x_train))
            # print(len(y_train))
            clf = svm.SVC(C=5, kernel='rbf', gamma=3, decision_function_shape='ovr')
            clf.fit(x_train, train_dataset.target)
            ratio = 0
            if(i == 0):
                ratio = 0
            else:
                ratio = i / len(sorted_word_var)
            print("裁剪率:%f" % (ratio))
            print("训练集")
            train_score = clf.score(x_train, train_dataset.target)
            print(train_score)  # 精度
            print("测试集")
            test_score = clf.score(x_test, test_dataset.target)
            print(test_score)  # 精度
            x_test_predict = clf.predict(x_test)
            ACU = get_accuracy(x_test_predict, test_dataset.target)
            print("ACU is %f" % ACU)
            f.write("%s\t%s\t\n" % (str(ratio), str(ACU)))

