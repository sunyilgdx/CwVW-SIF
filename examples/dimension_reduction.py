#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2018/7/11
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from numpy import *
def get_sentences_embedding(dataset,words,weight4ind,params,We):
    #
    sentences_embedding=[]
    for d in dataset.data:
        s=d
        s=s.replace('\n','')
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
    weightpara = 2e-4  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 0  # number of principal components to remove in SIF weighting scheme
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
    param = params.params()
    param.rmpc = rmpc
    sentence_embedding_all = get_sentences_embedding(dataset, words, weight4ind, param, We)
    sentence_embedding_all = turn2std(sentence_embedding_all)  # 将矩阵转换为标准矩阵，需要时请将这一行的注释去掉
    return  sentence_embedding_all
def eigValPct(eigVals,percentage):
    sortArray=sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempsum=0
    num=0
    for i in sortArray:
        tempsum+=i
        num+=1
        if tempsum>=arraySum*percentage:
            return num
def pca(dataMat,division=2,percentage=0.7):
    meanVals = mean(dataMat, axis=0) #对每一列求平均值，因为协方差的计算中需要减去均值
    DataAdjust = dataMat - meanVals           #减去平均值
    covMat = cov(DataAdjust, rowvar=0)    #cov()计算方差
    eigVals,eigVects = linalg.eig(mat(covMat)) #计算特征值和特征向量
    #print eigVals
    eigValInd = argsort(eigVals)
    k = eigValPct(eigVals, percentage)  # 要达到方差的百分比percentage，需要前k个向量
    #二维损失60%，三维65%
    eigValInd = eigValInd[:-(division+1):-1]   #保留最大的前K个特征值
    redEigVects = eigVects[:,eigValInd]        #对应的特征向量
    lowDDataMat = DataAdjust * redEigVects     #将数据转换到低维新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals   #重构数据，用于调试
    return lowDDataMat, reconMat, eigValInd, redEigVects



if __name__ == '__main__':
    # 获得测试数据
    # 加载数据集，设置类别
    # categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

    category1 = ['comp.graphics']
    category2 = ['soc.religion.christian']
    category3 = ['alt.atheism']
    category4 = ['sci.med']
    train_dataset1 = sklearn.datasets.fetch_20newsgroups(
        subset='train', categories=category1, shuffle=True, random_state=42)
    train_dataset2 = sklearn.datasets.fetch_20newsgroups(
        subset='train', categories=category2, shuffle=True, random_state=42)
    train_dataset3 = sklearn.datasets.fetch_20newsgroups(
        subset='train', categories=category3, shuffle=True, random_state=42)
    train_dataset4 = sklearn.datasets.fetch_20newsgroups(
        subset='train', categories=category4, shuffle=True, random_state=42)
    test_dataset1 = sklearn.datasets.fetch_20newsgroups(
        subset='test', categories=category1, shuffle=True, random_state=42)
    test_dataset2 = sklearn.datasets.fetch_20newsgroups(
        subset='test', categories=category2, shuffle=True, random_state=42)
    # 获得文章向量sentence_embedding_all和真实标签labels_true
    sentence_embedding_train1=get_sif(train_dataset1)
    labels_true1 = train_dataset1.target
    sentence_embedding_train2 = get_sif(train_dataset2)
    labels_true2 = train_dataset2.target
    sentence_embedding_train3 = get_sif(train_dataset3)
    labels_true3 = train_dataset3.target
    sentence_embedding_train4 = get_sif(train_dataset4)
    labels_true4 = train_dataset4.target

    sentence_embedding_train1_ =mat(sentence_embedding_train1)
    sentence_embedding_train2_ = mat(sentence_embedding_train2)
    sentence_embedding_train3_ = mat(sentence_embedding_train3)
    sentence_embedding_train4_ = mat(sentence_embedding_train4)
    division = 3
    lowDDataMat1, reconMat1,eigValInd1, redEigVects1 = pca(sentence_embedding_train1_, division)
    lowDDataMat2, reconMat2,eigValInd2, redEigVects2 = pca(sentence_embedding_train2_, division)
    lowDDataMat3, reconMat3, eigValInd3, redEigVects3 = pca(sentence_embedding_train3_, division)
    lowDDataMat4, reconMat4, eigValInd4, redEigVects4 = pca(sentence_embedding_train4_, division)
    train_embedding1=sentence_embedding_train1* redEigVects1
    train_embedding2 = sentence_embedding_train2 * redEigVects2
    train_embedding3 = sentence_embedding_train3 * redEigVects3
    train_embedding4 = sentence_embedding_train4 * redEigVects4
    # fig,ax = plt.subplot(111,projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    # ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for data in train_embedding1:
        data = numpy.array(data)
        ax.scatter(data[0][0], data[0][1],data[0][2], s=10, color='r', marker='*')
    for data in train_embedding2:
        data = numpy.array(data)
        ax.scatter(data[0][0], data[0][1],data[0][2], s=10, color='b', marker='.')
    # for data in train_embedding3[:300]:
    #     data = numpy.array(data)
    #     ax.scatter(data[0][0], data[0][1],data[0][2], s=10, color='y', marker='o')
    # for data in train_embedding4[:300]:
    #     data = numpy.array(data)
    #     ax.scatter(data[0][0], data[0][1],data[0][2], s=10, color='g', marker='+')
    fig.autofmt_xdate()
    plt.show()

