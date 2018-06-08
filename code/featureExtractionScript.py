#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:04:04 2017

@author: veysiyildiz
"""

'''
extact features
'''

import scipy.integrate as integrate
import numpy as np
from PIL import Image
from skimage.morphology import binary_dilation 
from scipy.stats import moment
from sklearn.mixture import GMM

def  integralFunction(cs, a, b):
    '''
    find the curve length of cs between points a and b, this function is also called as g(t) fucntion 
    in the rest of the code(dg=first derivative, d2g=second derivative)
    '''
#    print integrate.quad(lambda t: np.linalg.norm(cs.derivative(1)(t)) , start , end )[0]
    return integrate.quad(lambda t: np.linalg.norm(cs.derivative(1)(t)) , a , b )[0]


#def sampleEqualDistPoints(cs):
#    sampleDist=0.5
#    s=0.
#    counter=1.
#    sList=[0.]
#    totalLenght= integralFunction(cs,0. ,cs.x[-1] )
#    while counter < totalLenght:
#        while np.abs(integralFunction(cs,sList[-1],sList[-1]+s)- sampleDist)> 0.05:
#            s+=0.05
#        sList.append(sList[-1]+s)

#        s=0.
#        counter+=sampleDist
#    return sList
def sampleEqualDistPoints(cs):
    sampleDist=1
    sList= [i for i in range(len(cs.x)/sampleDist) ]
#    s=0.
#    counter=1.
#    sList=[0.]
#    totalLenght= integralFunction(cs,0. ,cs.x[-1] )
#    while counter < totalLenght:
#        while np.abs(integralFunction(cs,sList[-1],sList[-1]+s)- sampleDist)> 0.05:
#            s+=0.05
#        sList.append(sList[-1]+s)
#        s=0.
#        counter+=sampleDist
    return sList



def derivative(f):
    """
    Computes the numerical derivative of a function.
    """
    def df(x, h=0.1e-5):
        return ( f(x+h/2) - f(x-h/2) )/h
    return df

def dg(cs,a):
    '''
    norm of the curve derivative at given point.
    s=integralFunction
    ds/dt= ||nabla_t C(t)||
    '''
    return np.linalg.norm(cs.derivative(1)(a))

def d2g(cs,a):
    '''
    derivative norm of the curve derivative at given point.
    s=integralFunction
    d^2s/dt^2= d||nabla_t C(t)||/dt
    '''    
    return derivative(lambda t: dg(cs,t))(a)


def velocity(cs,t):
    '''
    return the velocity vector of cubic spline at given point t.(derivative is with respecto the curle length parameter
      
    v(s)= dCs(s)/ds = dCs(g(t))/ g'(t)dt
    
    '''
    return cs.derivative(1)(t) / dg(cs,t)  



def acceleration(cs,t):
    '''
    return the acceleration vector of cubic spline at given point t.
      
    formula should be written here    
    '''
#    print 'cuvature funciton called'
    temp= dg(cs,t)
    firstTerm=cs.derivative(2)(t) / temp
    secondTerm= cs.derivative(1)(t) * d2g(cs,t) / temp**2
    return firstTerm - secondTerm  


'''
starting extracting features

'''

'''
point based features
'''

def curveLength(cs,start,end):
    '''
    find the length of the spline cs, starts at 'start' ends at 'end' according to formula:
        curveLength= intg_start^end ||dc(t)/dt|| dt
    '''
    return integralFunction(cs, start, end)

def chordLength(cs, a,b):
    '''
    find the euclidian distance between points a and b on cubic spline(a-b should be the spline parameter)
    
    ||cs(a)-cs(b)||
    
    '''
    return np.linalg.norm(cs(a)-cs(b))


def accelerationNorm (cs, t):
    '''
    Norm of Acceleration vector: rate of changing velocity with respect to the rate of changing point location

    '''
    return np.linalg.norm(acceleration(cs,t))


def curvature(cs,t):
    '''
    return the curvature value of cubic spline at given point t.
      
    cur =|| (nabla_t^T nabla_t C(t) *  dg(t) -   nabla_t C(t) * d^2g(t) )/ (dg(t))^2 ||
    '''
#    print 'cuvature funciton called'
    temp= dg(cs,t)
    firstTerm=cs.derivative(2)(t) / temp
    secondTerm= cs.derivative(1)(t) * d2g(cs,t) / temp**2
    return np.linalg.norm(firstTerm - secondTerm )  

def pointDiameter (segmentedImage, cs , t):
    '''
    At each point the width of vessel on the normal direction (the direction perpendicular to the velocity).
    visual problem needs to be solved
    
    '''
    loc= cs(t)
    if np.isnan(loc).any():
        return np.nan
    vel= velocity(cs,t)
    normal= np.array([-vel[1], vel[0]]) 
    upbound=loc.copy()
    shape= segmentedImage.shape
    while isInLimits(upbound, shape ) and segmentedImage[upbound.astype(int)[0],upbound.astype(int)[1]]!=0:
        upbound += 0.1 * normal
    lowbound = loc.copy()
    while  isInLimits(lowbound, shape ) and segmentedImage[lowbound.astype(int)[0],lowbound.astype(int)[1]]!=0:   
        lowbound -= 0.1 * normal
    pd=np.linalg.norm(lowbound-upbound)
    if pd!=0.:
        return pd
    else:
        return np.nan
    
def isInLimits(loc, shape):
    loc=loc.astype(int)
    if any(loc<0):
        return False
    if all(loc < shape):
        return True
    else:
        return False
    



'''
segment based features

'''



def cumTortuosityInd(cs, a, b,curvLeng=None,chLeng=None):
    '''
    Cumulative Tortuosity Index (CTI): For each segment, the curve length divided by chord length (The distance between the curve start and end points). 
	cti(x) = Lc(x)/Lx(x)

    '''
    if curvLeng is not None and chLeng is not  None:
        return 1.* curvLeng / chLeng
    elif curvLeng is not None and chLeng is None:
        return 1.* curvLeng / chordLength(cs, a,b) 
    elif curvLeng is None and chLeng is not None:
        return 1.* curveLength(cs,a,b) / chLeng
    else:
        return 1.* curveLength(cs,a,b) / chordLength(cs, a,b)      


def integratedCurvature(cs,a,b):
    '''
    Integrated Curvature (IC): For each segment, the sum of all the absolute value on point curvature.(a,b are curve parameters)
		ic(x)=intgrl_a^b |curvature(s)|ds
    '''
    
    return integrate.quad(lambda t: np.abs(curvature(cs,t)), a , b )[0]

def integratedSquaredCurvature(cs,a,b):
    '''
        Integrated Squared Curvature (ISC): For each segment, the sum of all the square of point curvature value.
		isc(x)=intgrl_a^b curvature(s)^2 ds
    '''
    
    return integrate.quad(lambda t: curvature(cs,t)**2, a , b )[0]

def icNormalizedByChordLength(cs,a,b,ic=None,chLeng=None):
    '''
        IC normalized by Chord Length: The IC value divided by the vessel chord length.
        		icLx(x) = ic(x)/Lx(x)
    '''
    if ic is not None and chLeng is not  None:
        return 1.* ic / chLeng
    elif ic is not None and chLeng is None:
        return 1.* ic / chordLength(cs, a,b) 
    elif ic is None and chLeng is not None:
        return 1.* integratedCurvature(cs,a,b) / chLeng
    else:
        return 1.* integratedCurvature(cs,a,b) / chordLength(cs, a,b)      
    
    


def icNormalizedByCurveLength(cs,a,b,ic=None,curvLeng=None):
    '''
        IC normalized by Curve Length: The IC value divided by the vessel curve length.
        		icLc(x) = ic(x)/Lc(x)
    '''
    if ic is not None and curvLeng is not  None:
        return 1.* ic / curvLeng
    elif ic is not None and curvLeng is None:
        return 1.* ic / curveLength(cs, a,b) 
    elif ic is None and curvLeng is not None:
        return 1.* integratedCurvature(cs,a,b) / curvLeng
    else:
        return 1.* integratedCurvature(cs,a,b) / curveLength(cs, a,b)      



        

def iscNormalizedByChordLength(cs,a,b,isc=None,chLeng=None):
    '''
        The ISC value divided by the vessel chord length.
		iscLx(x) = isc(x)/Lx(x)

    '''
    if isc is not None and chLeng is not  None:
        return 1.* isc / chLeng
    elif isc is not None and chLeng is None:
        return 1.* isc / chordLength(cs, a,b) 
    elif isc is None and chLeng is not None:
        return 1.* integratedSquaredCurvature(cs,a,b) / chLeng
    else:
        return 1.* integratedSquaredCurvature(cs,a,b) / chordLength(cs, a,b)      
    
    
def iscNormalizedByCurveLength(cs,a,b,isc=None,curvLeng=None):
    '''
        ISC normalized by Curve Length: The ISC value divided by the vessel curve length.
    		iscLc(x) = ic(x)/Lc(x)
    '''
    if isc is not None and curvLeng is not  None:
        return 1.* isc / curvLeng
    elif isc is not None and curvLeng is None:
        return 1.* isc / curveLength(cs, a,b) 
    elif isc is None and curvLeng is not None:
        return 1.* integratedSquaredCurvature(cs,a,b) / curvLeng
    else:
        return 1.* integratedSquaredCurvature(cs,a,b) / curveLength(cs, a,b)  
 
def checkLimits(cs, shape):
    'returns the indecies of the csx elements which stays in the limits of the image'
    list1= list(np.where(cs(cs.x).astype(int)[:,0]<shape[0])[0])
    list2=  list(np.where(cs(cs.x).astype(int)[:,1]<shape[1])[0])
    return list(set(list1).intersection(list2))

def averageSegmentDiameter (cs,a,b,segmentedImage,curvLeng=None):
    '''    
    Average Segment Diameter: The number of pixels on the vessel divided by the vessel curve length.
    		asd(x) = #pixels/Lc(x)
    possible very tick vessels should be checked
    '''
    newImage=np.zeros_like(segmentedImage)
    shape=segmentedImage.shape
    limits= checkLimits(cs,shape)    
    newImage[cs(cs.x).astype(int)[:,0][limits],cs(cs.x).astype(int)[:,1][limits]]=255        
#    newImage[cs(cs.x).astype(int)[:,0],cs(cs.x).astype(int)[:,1]]=255    
    tickImage= binary_dilation(newImage)
    tickImage= binary_dilation(tickImage)    
    intersection= tickImage * segmentedImage
    pixelNum = np.count_nonzero(intersection)
    tickPixels=np.count_nonzero(tickImage)
    if 1.* pixelNum/tickPixels> 0.7:
        tickImage= binary_dilation(tickImage)
        intersection= tickImage * segmentedImage
        pixelNum = np.count_nonzero(intersection)        
    if curvLeng is None:
        return 1.* pixelNum/ curveLength(cs,a,b)
    else: 
        return 1.* pixelNum/ curvLeng

#    tickerImage= binary_dilation(tickImage)
#    
#    fig = plt.figure(figsize=(10., 10.))
#    plt.imshow(newImage)
#    plt.imshow(tickImage)
#    plt.imshow(intersection)
#    #plt.imshow(segmentedImage)
#    
#    plt.legend(fontsize=12)
#    plt.show()
#        
#    
'''
   tree based features

''' 
    
def distToDiscCenter(cs, a,b, cntr):
    '''
    Distance to the Disc Center (DDC). For each vessel segment, DDC is the distance between the vessel ending point and the disc center.
		ddc(x) = ||c(a)- cntr||
        a,b are the start and end points of the segment
		Where cnrt is the coordinate of disc center.

    '''    
    return max(np.linalg.norm(cs(a)-cntr),np.linalg.norm(cs(b)-cntr))
    


'''
starting to extract all of the features
'''  


def addFeaturesTraditional(feats, newFeature):
    '''
    given 1d feature vector add new features from newFeature list(min,max, med, mean,moments)
    '''
    
    sortedFeature= sorted([i for i in newFeature if not np.isnan(i) ])
#    feats=np.array(sortedFeature[-1])
    feats=np.concatenate((feats,[sortedFeature[-1]]))
    feats=np.concatenate((feats,[sortedFeature[-2]]))
    feats=np.concatenate((feats,[sortedFeature[0]]))
    feats=np.concatenate((feats,[sortedFeature[1]]))
    feats=np.concatenate((feats,[np.mean(sortedFeature)]))
    feats=np.concatenate((feats,[np.median(sortedFeature)]))
    feats=np.concatenate((feats,[moment(sortedFeature,2)]))
    feats=np.concatenate((feats,[moment(sortedFeature,3)]))
    
    return feats
def addFeaturesGMM(feats, newFeature):
    '''
    given 1d feature vector add new GMM features from newFeature list(means coveriances,Pcomponentanalys)
    '''

    mix=GMM(n_components=2, random_state=1)
    sortedFeature= sorted([i for i in newFeature if not np.isnan(i) ])
    leng=len(sortedFeature)
    mix.fit(np.array(sortedFeature).reshape(leng,1))
    mean=mix.means_
    cov=mix.covars_
    rate= 1.*np.count_nonzero(mix.predict(np.array(sortedFeature).reshape(leng,1))==0)/leng
    feats=np.concatenate((feats,mean[0]))
    feats=np.concatenate((feats,mean[1]))
    feats=np.concatenate((feats,cov[0]))
    feats=np.concatenate((feats,cov[1]))
    feats=np.concatenate((feats,[rate]))    
    return feats

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

def extractFeatures(csList,segmentedImageName,cntr):  
    segmentedImage = np.array(Image.open(segmentedImageName))
    segmentedImage = threshold(segmentedImage)

#    if len(segmentedImage.shape)>2:
#        segmentedImage= segmentedImage[:,:,1]  
#    segmentedImage[np.where(segmentedImage<51)]=0
#    segmentedImage[np.where(segmentedImage>=51)]=255#    featureList=[]
#    time1=time() 
    featureDict={'cti':[],'ic':[], 'isc':[],'icLx':[],'icLc':[],'iscLx':[],'iscLc':[],'asd':[],'ddc':[],'curv':[],'pd':[]} #removed
#    ct ,ic ,isc,icLx, icLc, iscLx, iscLc, asd, dcd =[],[],[],[],[],[],[],[],[]
#    acc,curv,pd=[],[],[]
    for cs in csList:
        csx=cs.x
        csx=[item for item in csx if not any(np.isnan(cs(item)))]
#        segFeatures= []
        curvLeng=curveLength(cs,csx[0],csx[-1])
        chLeng=chordLength(cs,csx[0],csx[-1])   

        featureDict['cti'].append(cumTortuosityInd(cs, csx[0], csx[-1],curvLeng,chLeng))
        featureDict['ic'].append(integratedCurvature(cs, csx[0], csx[-1]))
#        print 'integratedCurvature done,s'
    
        featureDict['isc'].append(integratedSquaredCurvature(cs, csx[0], csx[-1]))
#        print 'integratedSquaredCurvature done'
        featureDict['icLx'].append(icNormalizedByChordLength(cs, csx[0], csx[-1],featureDict['ic'][-1],chLeng))
#        print 'icNormalizedByChordLength done'
        featureDict['icLc'].append(icNormalizedByCurveLength(cs, csx[0], csx[-1],featureDict['ic'][-1],curvLeng))
#        print 'icNormalizedByCurveLength done'
        featureDict['iscLx'].append(iscNormalizedByChordLength(cs, csx[0], csx[-1],featureDict['isc'][-1],chLeng))
#        print 'iscNormalizedByChordLength done'
        featureDict['iscLc'].append(iscNormalizedByCurveLength(cs, csx[0], csx[-1],featureDict['isc'][-1],curvLeng))
#        print 'iscNormalizedByCurveLength done'
        featureDict['asd'].append(averageSegmentDiameter (cs,csx[0], csx[-1],segmentedImage,curvLeng))
#        print 'averageSegmentDiameter done'
        featureDict['ddc'].append(distToDiscCenter(cs,csx[0],csx[-1],cntr))
#        print 'distToDiscCen done'
        
        
        equalS=sampleEqualDistPoints(cs)
        for t in equalS:
#            featureDict['acc'].append(accelerationNorm(cs,t)) #removed
#            print 'accelerationNorm done'
#            featureDict['curv'].append(curvature(cs,t))
            featureDict['curv'].append(curvature(cs,t)) ## we realized that acc is wrong in the previous code, after correcting it becomes the same a curv

#            print 'curvature done'        
            featureDict['pd'].append(pointDiameter(segmentedImage, cs,t))
#            print 'pointDiameter done'        


    feats=np.array([])
    feats= addFeaturesGMM(feats,featureDict['ddc'])
    feats= addFeaturesGMM(feats,featureDict['cti'])
    feats= addFeaturesGMM(feats,featureDict['ic'])
    feats= addFeaturesGMM(feats,featureDict['isc'])
    feats= addFeaturesGMM(feats,featureDict['icLc'])
    feats= addFeaturesGMM(feats,featureDict['iscLc'])
    feats= addFeaturesGMM(feats,featureDict['icLx'])
    feats= addFeaturesGMM(feats,featureDict['iscLx'])
#    feats= addFeaturesGMM(feats,featureDict['acc']) #removed
    feats= addFeaturesGMM(feats,featureDict['curv'])
    feats= addFeaturesGMM(feats,featureDict['asd'])
    feats= addFeaturesGMM(feats,featureDict['pd'])
    
    
    feats= addFeaturesTraditional(feats,featureDict['ddc'])
    feats= addFeaturesTraditional(feats,featureDict['cti'])
    feats= addFeaturesTraditional(feats,featureDict['ic'])
    feats= addFeaturesTraditional(feats,featureDict['isc'])
    feats= addFeaturesTraditional(feats,featureDict['icLc'])
    feats= addFeaturesTraditional(feats,featureDict['iscLc'])
    feats= addFeaturesTraditional(feats,featureDict['icLx'])
    feats= addFeaturesTraditional(feats,featureDict['iscLx'])
#    feats= addFeaturesTraditional(feats,featureDict['acc']) #removed
    feats= addFeaturesTraditional(feats,featureDict['curv'])
    feats= addFeaturesTraditional(feats,featureDict['asd'])
    feats= addFeaturesTraditional(feats,featureDict['pd'])
#    time2=time()
#    print 'time diff is' , time2-time1


#    a=np.array([])
    
#    a=np.concatenate((a,np.array([1])))
    return feats
























