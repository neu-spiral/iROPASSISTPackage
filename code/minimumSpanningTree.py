#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:03:37 2017

@author: veysiyildiz
"""

'''
Minimum spanning tree for center lines


'''
import numpy as np
import networkx as nx
from operator import itemgetter



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi *180


def findDiscPoints(finalPointsList, cntr,radius):
    return [(idx,i) for idx,i in enumerate (finalPointsList) if np.abs(np.linalg.norm(i-cntr)-radius)<2.]
    
def findSeedPoints(finalPointsList, cntr,radius): 
    '''
    from the provided disc points measure the angle of vector between point and center and remove the points 
    which are almost on the same angle. remove the one which is further away from the disc center
    '''
    discPoints=findDiscPoints(finalPointsList, cntr,radius)
    angles=sorted([(i[0],angle_between(cntr-i[1], np.array([0.,1.]))) for i in discPoints], key=itemgetter(1))
    seedIdx=[i for i in angles]
    update=1
    while update==1:
        update=0
        for idx, i in enumerate (seedIdx):
            if idx+1 < len(seedIdx):
                if np.abs(seedIdx[idx+1][1]- i[1])  <1.5:
                    if np.linalg.norm(finalPointsList[i[0]]-cntr) < np.linalg.norm(finalPointsList[seedIdx[idx+1][0]]-cntr):
                        seedIdx.pop(idx+1)
                    else:
                        seedIdx.pop(idx)
                    update=1    
    seedIdx=[i[0] for i in seedIdx]
    return seedIdx    


def createGraph(finalPoints,cntr):   
    finalPointsList= list(finalPoints.T)    
    radius=35. 
    finalPointsList=[np.array(cntr)]+ finalPointsList
    seedIdx=findSeedPoints(finalPointsList, cntr,radius)

    graph=[]
    for i in seedIdx:
        graph.append(('cntr',i,radius))
    
    
    for idx,i in enumerate(finalPointsList[1:]):
        for idx2,i2 in enumerate(finalPointsList[idx:]):
            norm=np.linalg.norm(i-i2)
            if norm<3.:
                graph.append((idx,idx+idx2-1,np.linalg.norm(i-i2)))
    return graph   

def findBranchingPoints(edges):
    '''
    find the branching points of the givin edge list
        input: 
            edges a python list consist of tuples edges= [(node1,node2),(node2,node3), (node2,node4), (node3,node5), (node6,node5)]
        output: 
            a python list consist of brach points = [node2]
    '''

    
    allPoints= [i for item in edges for i in item]
    branchPoints = [k for k in set(allPoints) if allPoints.count(k)>2]
    return branchPoints

def branches(edges):
    '''
    find the bracnhes of a given graph
        input:
            edges a python list consist of tuples edges (makes sure that this list is coming from depth first traverse algovrithm)
                exp= [(node1,node2),(node2,node3), (node2,node4), (node3,node5), (node5,node6)]
            
        output:
            a python list consist of lists which contains the branches allBranches= [[(node1,node2)],[(node2,node3),(node3,node5),(node5,node6)],[node2,node4)]]
    '''
    waitList= [a for a in edges]
        
    allBrnc=[]
    branchPoints= findBranchingPoints(edges)

    while len( waitList)>0:
        brnc=[]
        pnt=waitList[0]
        brnc.append(pnt)
        waitList.remove(pnt)
        if len(waitList)==0:
            allBrnc.append(brnc)
            break    
        counter = 0
        while (pnt[1] not in branchPoints) :
            prevPnt= pnt
            pnt=waitList[0]
    
            if prevPnt[1] != pnt[0]:
                break
            if not (counter>0 and (pnt[0] in branchPoints)):
                counter +=1
                brnc.append(pnt)
                waitList.remove(pnt)
                if len(waitList)==0:
                    break              
            else:
                break
        allBrnc.append(brnc)
    return allBrnc

def removeOpticDiscConnections(allBranches):
    for idx2,branch in enumerate(allBranches):    
        counter=0
        temp=[m for m in branch]
        for idx,i in enumerate (branch):
            if i[0]=='cntr' or i[1]=='cntr':
                del temp[idx-counter] 
                counter+=1
        allBranches[idx2]=[m for m in temp]
    return allBranches
def removeEmptyBranches(allBranches):
    '''
    some branches are empty this function removes them from the list
    '''
    return [i for i in allBranches if len(i)>0]
    
def vesselTree(finalPoints,cntr):
    graph=createGraph(finalPoints,cntr)      
    G=nx.Graph()
    G.add_weighted_edges_from(graph) 
    
    T=nx.minimum_spanning_tree(G)    
    edgeList=list(nx.dfs_edges(T))
    allBranches= branches(edgeList)
    allBranches= removeOpticDiscConnections(allBranches)
    allBranches= removeEmptyBranches(allBranches)
    
    return allBranches



        
        
        
        
        
        
        
        
        
        
        
        
        
        