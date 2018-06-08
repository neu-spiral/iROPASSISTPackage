
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:20:05 2017

@author: veysiyildiz

take the input as a segmented image interpolte with kernels starting from sample points climb the principle surface return the final points
"""

import numpy as np
import truncatedKernels as kd
from PIL import Image,ImageDraw
from skimage.morphology  import thin
import scaleCalculation as sc


def climbPrincipalSurface(w,S,X_i,xStart, kernel_type='Gaussian',d=1,eps=0.1,maxIter=100):
    """ Starting from point xStart, climb up to d dimensional principal surface (using log function)  """
    newX=xStart.copy()
    lastX=newX.copy()
#    path=newX.copy()
    grad_newX=kd.ln_gx(w,S,X_i,newX,  'Gaussian')
    hes_newX=kd.ln_Hx(w,S,X_i,newX,  'Gaussian')

    if ~np.isnan(hes_newX).any() and ~np.isnan(grad_newX).any():
        [eigVal,eigVec]=np.linalg.eig(hes_newX)        
        #plot_grad_eig(grad,eig_vec,xStart,mix_pdf)  
#        print eigVal
        if eigVal[0]==eigVal[1]:
            return (lastX,None,None)
        #sort the eig_val-vec
        sortedEigVal, sortedEigVec=zip(*sorted(zip(eigVal,np.transpose(eigVec)),reverse=True))        

        #orthogonal ascent direction
        qOrt=Qort(sortedEigVec,d)
        ortDir=qOrt*(qOrt.T)*(grad_newX.T)
        #parallel subspc eigVectors
        qPar=Qpar(sortedEigVec,d)
        ## learning rate
        alpK=np.abs(1./max(sortedEigVal[d:]))
        ##initialize the iterations
        itNum=0
        while itNum<maxIter and np.abs(angle_between(grad_newX,qOrt)-90)>eps:
            itNum +=1
#            print ortDir
#            print alpK
#            print 'angle at %d iteration is:' %(itNum),angle_between(grad_newX,qPar) 
            lastX=newX.copy()
            newX=newX + 0.5 * alpK * ortDir 
#            print newX
#            print 'increment at %d iteration is:' %(itNum), 0.05*alpK*ortDir 
#            path=np.concatenate((path,newX),axis=1)
            grad_newX= kd.ln_gx(w,S,X_i,newX,  'Gaussian')
            hes_newX= kd.ln_Hx(w,S,X_i,newX,  'Gaussian')

            if ~np.isnan(hes_newX).any() and ~np.isnan(grad_newX).any():
                [eigVal,eigVec]=np.linalg.eig(hes_newX)        
                #plot_grad_eig(grad,eig_vec,xStart,mix_pdf) 
#                print hes_newX, '\n'
#                print eigVal, 'inside while'
                if eigVal[0]==eigVal[1]:
                    return (lastX,None, None)
                #sort the eig_val-vec
                sortedEigVal, sortedEigVec=zip(*sorted(zip(eigVal,np.transpose(eigVec)),reverse=True))

                #parallel subspc eigVectors
#                qPar=Qpar(sortedEigVec,d)
                #orthogonal ascent direction
                qOrt=Qort(sortedEigVec,d)
                
                if itNum<maxIter and np.abs(angle_between(grad_newX,qOrt)-90)>eps: 
                    #orthogonal ascent direction
                    ortDir=qOrt*(qOrt.T)*(grad_newX.T)
#                    print 'final angle with orthogonal is: ', angle_between(grad_newX,qOrt)     
                    ## learning rate
                    alpK=np.abs(1./max(sortedEigVal[d:]))  
#                    print ortDir, 'sec'
#                    print alpK, 'sec' 
        if itNum==maxIter:
            if np.abs(angle_between(grad_newX,qOrt)-90) >5.:
                return None, None, None
            else:
                return None, None, lastX
            
#            while itNum<(maxIter+maxIter) and np.abs(angle_between(grad_newX,qOrt)-90)>10:
#                itNum +=1
#    #            print ortDir
#    #            print alpK
#    #            print 'angle at %d iteration is:' %(itNum),angle_between(grad_newX,qPar) 
#                lastX=newX.copy()
#                newX=newX + 100*ortDir
#    #            print newX
#    #            print 'increment at %d iteration is:' %(itNum), 20.*alpK*ortDir 
#                path=np.concatenate((path,newX),axis=1)
#                grad_newX= kd.ln_gx(w,S,X_i,newX,  'Gaussian')
#                hes_newX= kd.ln_Hx(w,S,X_i,newX,  'Gaussian')
#    
#                if ~np.isnan(hes_newX).any() and ~np.isnan(grad_newX).any():
#                    [eigVal,eigVec]=np.linalg.eig(hes_newX)        
#                    #plot_grad_eig(grad,eig_vec,xStart,mix_pdf) 
#    #                print hes_newX, '\n'
#    #                print eigVal, 'inside while'
#                    #sort the eig_val-vec
#                    sortedEigVal, sortedEigVec=zip(*sorted(zip(eigVal,np.transpose(eigVec)),reverse=True))
#    
#                    #parallel subspc eigVectors
#                    qPar=Qpar(sortedEigVec,d)
#                    #orthogonal ascent direction
#                    qOrt=Qort(sortedEigVec,d)
#                    
#                    if itNum<maxIter and np.abs(np.dot(grad_newX,qOrt))>eps: 
#                        #orthogonal ascent direction
#                        ortDir=qOrt*(qOrt.T)*(grad_newX.T)
#    
#                        ## learning rate
#                        alpK=np.abs(1./max(sortedEigVal[d:]))  
#    #                    print ortDir, 'sec'
#    #                    print alpK, 'sec'             
#        print itNum
#        print 'final angle with orthogonal is: ', angle_between(grad_newX,qOrt)
#        print 'final angle with parallel is: ', angle_between(grad_newX,qPar)
    return (lastX,None,None)
        

    
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

def sixDDRemove(segmentedImageName,cntr):
    '''
    remove the parts which are away from disc center more than 6dd (disc diameter is assumed as 60 pixels)
    '''
    
    segmentedImage = np.array(Image.open(segmentedImageName))
    if len(segmentedImage.shape)>2:
        segmentedImage= segmentedImage[:,:,1]  
    segmentedImage[np.where(segmentedImage<51)]=0
    segmentedImage[np.where(segmentedImage>=51)]=1

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

def trace(segmentedImageName,cntr):
    
    sixDDImage = sixDDRemove(segmentedImageName,cntr)
    sixDDnoCenterImage = discCenterRemove(sixDDImage,cntr)

    thinImage= thin(sixDDnoCenterImage,max_iter=2)
    
    pixels=np.concatenate((np.mat(np.nonzero(sixDDnoCenterImage))[0],np.mat(np.nonzero(sixDDnoCenterImage)[1])),axis=0)
    
    seeds=np.concatenate((np.mat(np.nonzero(thinImage))[0],np.mat(np.nonzero(thinImage)[1])),axis=0)
    
    seeds=[np.mat(i).T for i in seeds.T.tolist()]
    
    means=[np.mat(i).T for i in pixels.T.tolist()]
    
    w=[1]*len(means)
    
    S=[]
    from time import time
    time1=time()
    for point in means:
        S.append(sc.scaleCalc(point,means, w, K= np.sqrt(len(means)),sigma=2.5, eps =1e-2))
    time2=time()
    print 'scale calculation took', time2-time1
#    S=[np.eye(2)]*len(means)

    maxS=[i.max()+3. for i in S]

    finalPoints=seeds[0] 
    counter=0
    
    for i in seeds:
        counter +=1 
        print counter 
        xStart=np.mat(i)
        [neighList, neighS, neighW]= zip(*[ (mean,S[index], w[index])  for index,mean in enumerate(means) if np.all(np.abs((xStart-mean))<maxS[index])])#np.linalg.norm(np.dot(S[index],xStart-mean),np.inf)< 5.])#
        
        (finalX,path, maxIt)=climbPrincipalSurface(neighW,neighS,neighList,xStart, kernel_type='Gaussian',d=1,eps=1.,maxIter=50)

        
        if not (finalX is None): finalPoints=np.concatenate((finalPoints,finalX),axis=1)
    time3=time()
    print 'projecting the points took', time3-time2
    import pickle
    f=open(segmentedImageName + 'FinalPointsofTracing.txt', 'w')
    pickle.dump(finalPoints,f)
    f.close()
    
    return finalPoints











