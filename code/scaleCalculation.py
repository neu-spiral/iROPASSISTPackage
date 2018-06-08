#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:40:58 2017

@author: veysiyildiz
"""

import numpy as np
from PIL import Image



def neighDetect (point,data,w,K=10):
    '''
    Detects the closest neighbors of point in data. When Summation of the weights of these points gets larger than the K value, function stops.
    inputs:
        point- d*1 np.mat()
        data - a python list containing the points in the from as defined above
        w - a python list which contains the weight of each data point provided in the above list
        K - when to stop looking for neighbors.When Summation of the weights of these points gets larger than the K value, function stops
    output:
        neigh: a python list consists of tuples .It contains neighbors of the point as well as
                                the index of the neighbors in the data list :[(neigh1,indx1),(neigh2,idx2)]
    '''
    sortedMeansIdx= sorted( enumerate(data),key= lambda (idx,xj) :np.linalg.norm(point-xj))
    
    sumWj=0.
    neigh=[]
    for idx,xj in sortedMeansIdx:
        neigh.append((xj,idx))
        sumWj += w[idx]        
        if sumWj>K:
            return neigh
    return neigh   


def scaleCalc (point,data ,w , K , sigma=1., eps=1e-2): 
    '''
    for a point calculate the scale matrix using nearest neighbors
    inputs:
        point- d*1 np.mat()
        data - a python list containing the points in the from as defined above
        w - a python list which contains the weight of each data point provided in the above list
        K - when to stop looking for neighbors.When Summation of the weights of these points gets larger than the K value, function stops
    output:
        si: scale matrix, cov_mat= si^T * si        
    
    formulas:
        ci=ciH+deltai*identity 
        deltai= (1/n) * eps* trace(ciH)
        ciH= sum_(in neighbors) w_j * (x_j-x_i) * (x_j- x_i)^T
        
        c_i= Q_i * D_i * Q_i  (eigen Decomposition)
        s_i = sigma^-1 * D_i ^-1/2 * Q_i^T
    
    cov_mat= si^T * si    
    
    '''
    dim=point.shape[0]
    ciH=np.zeros((dim,dim))
    
    for (xj,idx) in neighDetect(point,data,w,K):
        ciH += (w[idx]/float(K)) * np.dot( (xj-point) , (xj-point).T )
        
    
    deltai= (1./dim) * eps * ciH.trace()
    ci=ciH + deltai * np.eye(dim)
    
    [eigVal,eigVec]=np.linalg.eig(ci)    
    si = (1./ sigma) * np.dot( np.sqrt(np.diag(1./eigVal)) , eigVec.T)
    return si


#
#
#manuelImage = np.array(Image.open("1_Manual_All.bmp"))[:,:,0]
##
##manuelImage = np.array(image)[:,:,1]
#
#manuelImage2=manuelImage#[300:400,400:500]
#manuelImage2[np.where(manuelImage2<255)]=0
#
#pixels=np.concatenate((np.mat(np.nonzero(manuelImage2))[0],np.mat(np.nonzero(manuelImage2)[1])),axis=0)
#
#means=[np.mat(i).T for i in pixels.T.tolist()]
#w=[1]*len(means)
#K=20.
#eps=1e-2
#sigma=1.
#
#
#point=means[10]
#neighDetect(point,means,w,K)
#
#data=means
#
#point=means[10]
#
#
#newS=[]
#for points in means:
#    newS.append(scaleCalc(points,data , w , K , sigma, eps))
#
#cov=[np.dot(s.T,s) for s in newS]






        