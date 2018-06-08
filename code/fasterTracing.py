
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:20:05 2017

@author: veysiyildiz

take the input as a segmented image interpolte with kernels starting from sample points climb the principle surface return the final points
"""

import numpy as np
from PIL import Image,ImageDraw
from skimage.morphology  import thin
import scaleCalculation as sc
import truncKDDE as kdde
import pickle

def climbPrincipalSurface(w,S,X_i,xStart, kernel_type='Gaussian',d=1,eps=0.1,maxIter=100):
    """ Starting from point xStart, climb up to d dimensional principal surface (using log function)  """
    newX=xStart.copy()
#    print 'newX is:' ,newX
    lastX=newX.copy()
#    print 'lastX is:' , lastX
    grad_newX=kdde.ln_gx(w,S,X_i,newX,  'Gaussian')
#    print 'grad is:', grad_newX
    hes_newX=kdde.ln_Hx(w,S,X_i,newX,  'Gaussian')
#    print 'hes is', hes_newX

    if ~np.isnan(hes_newX).any() and ~np.isnan(grad_newX).any():
        [eigVal,eigVec]=np.linalg.eig(hes_newX)        
        if eigVal[0]==eigVal[1]:
            return 
        #sort the eig_val-vec
        sortedEigVal, sortedEigVec=zip(*sorted(zip(eigVal,np.transpose(eigVec)),reverse=True))        

        #orthogonal ascent direction
        qOrt=Qort(sortedEigVec,d).reshape(newX.shape)
        ortDir=np.dot (np.dot(qOrt,qOrt.T),(grad_newX.T))
        #parallel subspc eigVectors
        qPar=Qpar(sortedEigVec,d)
        ## learning rate
        alpK=np.abs(1./max(sortedEigVal[d:]))
        ##initialize the iterations
        itNum=0
        while itNum<maxIter and np.abs(angle_between(grad_newX,qOrt)-90)>eps:
            itNum +=1
            lastX=newX.copy()
            newX=newX +  1.1 * alpK * ortDir 
            grad_newX= kdde.ln_gx(w,S,X_i,newX,  'Gaussian')
            hes_newX= kdde.ln_Hx(w,S,X_i,newX,  'Gaussian')

            if ~np.isnan(hes_newX).any() and ~np.isnan(grad_newX).any():
                [eigVal,eigVec]=np.linalg.eig(hes_newX)        
                if eigVal[0]==eigVal[1]:
                    return 
                #sort the eig_val-vec
                sortedEigVal, sortedEigVec=zip(*sorted(zip(eigVal,np.transpose(eigVec)),reverse=True))

                #parallel subspc eigVectors
                qPar=Qpar(sortedEigVec,d)
                #orthogonal ascent direction
                qOrt=Qort(sortedEigVec,d).reshape(newX.shape)
            
                #orthogonal ascent direction
                ortDir=np.dot ( np.dot(qOrt, qOrt.T), (grad_newX.T))
                ## learning rate
                alpK=np.abs(1./max(sortedEigVal[d:]))  
        return lastX #,path)
    return        
    
def Qort(sortedEigVec,d=1):
    """concatenation of eig vectors except first d. (orthogonal decompositon of hessian) """
    return np.concatenate(sortedEigVec[d:]).T
def Qpar(sortedEigVec,d=1):
    """concatenation of first d eig vectors .  (parallel decompositon of hessian)"""
    return np.concatenate(sortedEigVec[0:d]).T
    
#def isClose( w,S,X_i,x, kernel_type='Gaussian',d=1 ):   
#    """check if a point is close enough to ridge. if point worths itarating, returns true and derivatives of point"""
#    fX=kd.ln_fx(w,S,X_i,x,  'Gaussian')
#    gradX=kd.ln_gx(w,S,X_i,x,  'Gaussian')
#    hesX=kd.ln_Hx(w,S,X_i, x,  'Gaussian')
#    if ~np.isnan(hesX).any() and ~np.isnan(gradX).any():
#        [eigVal,eigVec]=np.linalg.eig(hesX)
#        if eigVal[0]==eigVal[1]:
#            return False
#        #sort the eig_val-vec
#        sortedEigVal, sortedEigVec=zip(*sorted(zip(eigVal,np.transpose(eigVec)),reverse=True))        
#        #orthogonal ascent direction
#        qOrt=Qort(sortedEigVec,d)
#        ortDir=qOrt*(qOrt.T)*(gradX.T)
#        #parallel subspc eigVectors
#        qPar=Qpar(sortedEigVec,d)
#        
#        #angle between gradient and parallel
#        ang=np.abs(angle_between(gradX,qPar))
#        
#        # second eigenvalue
#        lam2=sortedEigVal[1]
#        
#        if fX>np.log(0.01) and lam2<0 and True: #ang<50
#            ## learning rate
#            alpK=np.abs(1./max(sortedEigVal[d:]))
##            print ang
#            return (True) #, gradX, qOrt, ortDir, qPar, alpK)
#        else:
#            return False
#            
#    else:
#        return False
#        

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
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
def threshold(segmentedImage):
    if len(segmentedImage.shape)>2:
        segmentedImage= segmentedImage[:,:,1]      
    thr=threshold_otsu(segmentedImage)
    segmentedImage[np.where(segmentedImage<thr)]=0
    segmentedImage[np.where(segmentedImage>=thr)]=255    
    segmentedImage=segmentedImage.astype(bool)
    labeled = remove_small_objects(segmentedImage , 50) 
    return labeled
#    segmentedImage[np.where(segmentedImage<51)]=0
#    segmentedImage[np.where(segmentedImage>=51)]=1
#    
    
def sixDDRemove(segmentedImageName,cntr):
    '''
    remove the parts which are away from disc center more than 6dd (disc diameter is assumed as 60 pixels)
    '''
    
    segmentedImage = np.array(Image.open(segmentedImageName))
    segmentedImage = threshold(segmentedImage)


    image = Image.new('1', (segmentedImage.shape[1],segmentedImage.shape[0]))
    draw = ImageDraw.Draw(image)
    r=30 * 12
    x=cntr[1]
    y=cntr[0]
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255))
    
    return segmentedImage * np.array(image)

def discCenterRemove(segmentedImage,cntr):
    '''
    remove the disc center and interior parts (disc diameter is assumed as 60 pixels)
    '''
    image = Image.new('1', (segmentedImage.shape[1],segmentedImage.shape[0]),color=255)
    draw = ImageDraw.Draw(image)
    r=35. 
    x=cntr[1]
    y=cntr[0]
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(0))
    return segmentedImage * np.array(image)
from time import time 

def trace(segmentedImageName,cntr, saveDebug=1):
    
    sixDDImage = sixDDRemove(segmentedImageName,cntr)
    sixDDnoCenterImage = discCenterRemove(sixDDImage,cntr)
#    newImage=sixDDnoCenterImage[300:400,200:300].copy()
    thinImage= thin(sixDDnoCenterImage,max_iter=2)
    
    pixels=np.concatenate((np.mat(np.nonzero(sixDDnoCenterImage))[0],np.mat(np.nonzero(sixDDnoCenterImage)[1])),axis=0)    
    seeds=np.concatenate((np.mat(np.nonzero(thinImage))[0],np.mat(np.nonzero(thinImage)[1])),axis=0)    
    seeds=[np.array(i).reshape(2,1) for i in seeds.T.tolist()]
    means=[np.array(i).reshape(2,1) for i in pixels.T.tolist()]
    w=[1.]*len(means)
    time1=time()
    S=[0.5*np.eye(2)]* len(means)    
    K= np.sqrt(len(means))
    sigma=2.5
#    for point in means:        
#        S.append(sc.scaleCalc(point,means, w, K,sigma, eps =1e-2))
    time2=time()
#    print 'scale took' , time2-time1

    maxS=[i.max()+5. for i in S]
    finalPoints=np.empty((2,1))
    for i in seeds:
        xStart=i.copy()
        [neighList, neighS, neighW]= zip(*[ (mean,S[index], w[index])  for index,mean in enumerate(means) if np.all(np.abs((xStart-mean))<maxS[index])])#np.linalg.norm(np.dot(S[index],xStart-mean),np.inf)< 5.])#       
        finalX=climbPrincipalSurface(neighW,neighS,neighList,xStart, kernel_type='Gaussian',d=1,eps=1.,maxIter=50)
        if not (finalX is None): finalPoints=np.concatenate((finalPoints,finalX),axis=1)
    finalPoints=finalPoints[:,1:]
    time3=time()
    
    if saveDebug:        
        with open(segmentedImageName + 'FinalPointsofTracing.txt', 'wb') as f:
            pickle.dump(finalPoints,f)
    
    return finalPoints, time2-time1,time3-time2
    
    
#'''
#old code starts here
#'''    
#    pixels=np.concatenate((np.mat(np.nonzero(newImage))[0],np.mat(np.nonzero(newImage)[1])),axis=0)
#    
#    seeds=np.concatenate((np.mat(np.nonzero(thinImage))[0],np.mat(np.nonzero(thinImage)[1])),axis=0)
#    
#    seeds=[np.mat(i).T for i in seeds.T.tolist()]
#    
#    means=[np.mat(i).T for i in pixels.T.tolist()]
#    
#    w=[1]*len(means)
#    
#    S=[]
#    from time import time
#    time1=time()
#    for point in means:
#        S.append(scS.scaleCalc(point,means, w, K= np.sqrt(len(means)),sigma=2.5, eps =1e-2))
#    time2=time()
#    print 'scale calculation took', time2-time1
##    S=[np.eye(2)]*len(means)
#
#    maxS=[i.max()+3. for i in S]
#
#    finalPoints=seeds[0] 
#    counter=0
#    
#    for i in seeds:
#        counter +=1 
#        print counter 
#        xStart=np.mat(i)
#        [neighList, neighS, neighW]= zip(*[ (mean,S[index], w[index])  for index,mean in enumerate(means) if np.all(np.abs((xStart-mean))<maxS[index])])#np.linalg.norm(np.dot(S[index],xStart-mean),np.inf)< 5.])#
#        
#        (finalX,path, maxIt)=oldclimbPrincipalSurface(neighW,neighS,neighList,xStart, kernel_type='Gaussian',d=1,eps=1.,maxIter=50)
#
#        
#        if not (finalX is None): finalPoints=np.concatenate((finalPoints,finalX),axis=1)
#    time3=time()
#    print 'projecting the points took', time3-time2
#
#
#'''
#ends here
#'''
#fig,ax = plt.subplots()
#fig.set_size_inches(10.5, 10.5)
#plt.imshow(newImage.T, interpolation="nearest", origin="upper")
#plt.scatter(np.array(finalPoints[0,:]),np.array(finalPoints[1,:]),marker='.',color='b')
##plt.colorbar()
#
#plt.show()


#
#    import pickle
#    f=open(segmentedImageName + 'FinalPointsofTracing.txt', 'w')
#    pickle.dump(finalPoints,f)
#    f.close()
#    
#    return finalPoints








