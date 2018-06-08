
import numpy as np
import pickle

def sigmoid(x):
    """
    This function returns the output of sigmoid function.
    
    Input:
            - x : float number which is the input of sigmoid function.
    
    Output:
            - sig : sigmoid function value of input x
    """

    sig = 1.0/(1+np.exp(-x))
    return sig

def reOrganizeContribution(contribution):
    '''
    this funtion sums up the contributions of the same features.for example accelartion feature has min max 2ndmin 2nd max vs.. This funtion sums them up
    input: 
        -contribution : list with 156 elements
    output:
        -organized: a list with 12 elements
        
    '''
    def sumElements(lis,indx):
        sums=0.
        for i in indx:
            sums+=lis[i]
        return sums    
    ddc=[0,1,2,3,4,60,61,62,63,64,65,66,67]
    cti=[5,6,7,8,9,68,69,70,71,72,73,74,75]
    ic=[ 10, 11, 12, 13, 14,76, 77, 78, 79, 80, 81, 82, 83]
    isc=[15, 16, 17, 18, 19,84, 85, 86, 87, 88, 89, 90, 91]
    icLc=[20, 21, 22, 23, 24,92, 93, 94, 95, 96, 97, 98, 99]
    iscLc=[25, 26, 27, 28, 29,100, 101, 102, 103, 104, 105, 106, 107]
    icLx=[ 30, 31, 32, 33, 34,108, 109, 110, 111, 112, 113, 114, 115]
    iscLx=[35, 36, 37, 38, 39, 116, 117, 118, 119, 120, 121, 122, 123]
    acc=[40, 41, 42, 43, 44, 124, 125, 126, 127, 128, 129, 130, 131]
    curv=[45, 46, 47, 48, 49,132, 133, 134, 135, 136, 137, 138, 139]
    asd=[ 50, 51, 52, 53, 54,140, 141, 142, 143, 144, 145, 146, 147]
    pd=[55,56,57,58,59,148, 149, 150, 151, 152, 153, 154, 155]
    organized= [sumElements(contribution,ddc),sumElements(contribution,cti),sumElements(contribution,ic),\
                sumElements(contribution,isc),sumElements(contribution,icLc),sumElements(contribution,iscLc),\
                sumElements(contribution,icLx),sumElements(contribution,iscLx),sumElements(contribution,acc),\
                sumElements(contribution,curv),sumElements(contribution,asd),sumElements(contribution,pd)]
    return organized


def generateScore(feature,isPlus, vectFileName='beta.mat'):
    """
    This script is to generate score for iROP-ASSIST system.
    
    Input:
            - feature: 1 by 143 numpy array represents image features.
            - isPlus: binary 0 or 1. 0 predicts severity score in NOT normal class (normal vs not normal). 1 predicts severity score in Plus category (plus vs not plus).
            - vectFileName: string contains path to the .mat file which contains normVect, beta of RSD Plus and beta of RSD Pre-plus or higher.
    Output:
            - score: severity score in 0 to 100.
    """
    with open ('../parameters/feat_normalizer_parameters.txt', 'rb') as f:
        paramset = pickle.load(f)
        meanVec = paramset['norm_mean']
        stdVec = paramset['norm_std']
    with open ('../parameters/clasifier_params.txt', 'rb') as f:
        paramset = pickle.load(f)
        
    featureCenterlized = np.subtract(feature, meanVec)
    featureNormlized = np.divide(featureCenterlized, stdVec)
    if isPlus == 0:
        beta = paramset['beta_normal_5k_discDetected']
        const = paramset['c_normal_5k_discDetected']
    elif isPlus == 1:
        beta = paramset['beta_plus_5k_discDetected']
        const = paramset['c_plus_5k_discDetected']
    else:
        print 'isPlus should be either 0 or 1'
        
    scorePre = np.dot(featureNormlized, beta) + const
    # Map score into 0 to 100.
    score = 100.0*sigmoid(scorePre)
    return score 



