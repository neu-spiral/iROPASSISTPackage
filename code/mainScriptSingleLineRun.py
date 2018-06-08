#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:08:04 2018

@author: veysiyildiz
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:12:50 2017

@author: veysiyildiz
"""

import opticDiscDetector as odd
import fasterTracing as tc
import minimumSpanningTree as mst
import cubicSpline as cs
import featureExtractionScript as fes
import pandas as pd
import numpy as np
import argparse
import generateScore as gs
import os.path
#from time import time 

# python mainScript.py "example/" "imgNames.xlsx" "features.csv" "scores.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Retinal Image Analysis Code ',\
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('pathToFolder', help='Path to folder which contains images and xlsx file which contains names and disc centers')
    parser.add_argument('imageNamesXlsFile', help='Name of the xlsx file which contains names and disc centers') 
    parser.add_argument('scoreFileName', help='Name of the xlsx File in which output scores are stored')
    parser.add_argument('line', help='which line in the excel file to process')
    parser.add_argument('--saveDebug',default=1 ,help="If you want to save the segmented image, features, centerline points of the vessel, this should be 1.")
    parser.add_argument('--featureFileName',default='Features', help='Name of the xlsx File in which output features are stored')
    parser.add_argument('--predictPlus',default=0, help='The system would predict image severity score in Pre-plus or higher category by default. If you want to predict severity score in Plus category, this should be 1.')
    args = parser.parse_args()
    
    lineNumb=args.line
    path= args.pathToFolder
    if path[-1] != "/":
        path+= "/"        
    imageNames= args.imageNamesXlsFile
    if imageNames[-5:] != ".xlsx":
        imageNames=imageNames[:-5] + ".xlsx"
  
    if args.saveDebug:
        featureFileName=args.featureFileName
        if featureFileName[-4:] != ".xlsx":
            featureFileName+= ".xlsx"

    scoreFileName = args.scoreFileName
    if scoreFileName[-4:] != ".xlsx":
        scoreFileName+= ".xlsx"

    segmentationFileName= 'Segmented'    
    xl = pd.ExcelFile(path+imageNames)
    first_sheet = xl.parse(xl.sheet_names[0])

    
#
#    if args.saveDebug:            
#        outputFeatureDf = pd.DataFrame([],columns=[] )#'SegmentedImageName', 'Features'
#        featureWriter = pd.ExcelWriter(path+featureFileName, engine='xlsxwriter')
#        outputFeatureDf.to_excel(featureWriter, sheet_name='Sheet1')
   
    featureList=['DistanceToDiscCenter','CumulativeTortuosityIndex','IntegratedCurvature(IC)','IntegratedSquaredCurvature(ISC)'\
                 ,'ICNormalizedbyChordLength','ICNormalizedbyCurveLength','ISCNormalizedbyChordLength','ISCNormalizedbyCurveLength','NormofAcceleration',\
                 'Curvature','AverageSegmentDiameter','AveragePointDiameter']

#    outputScoreDf = pd.DataFrame([],columns=['SegmentedImageName','Score']+featureList )
#    scoreWriter = pd.ExcelWriter(path+scoreFileName, engine='xlsxwriter')
#    outputScoreDf.to_excel(scoreWriter, sheet_name='Sheet1')


    
    # Process CNN Segmentations
    nonSegmentedImageIdxs=[idx for idx,i in enumerate(first_sheet['SegmentationName']) if i!=i]
    imgCNNPathList=[path+str(first_sheet['ImageName'][i]) for i in nonSegmentedImageIdxs]        
    from time import time 
    ti=time()
#    cnnSegmentationsPath = path + segmentationFileName
#    cnnModel = 'unet'
#    unetCNN = SegmentUnet(cnnModel,cnnSegmentationsPath)
#    CNNSegmentedImagesList, imgNamesDoneList, imgNameFailedList = unetCNN.segment_batch(imgCNNPathList)
#    CNNSegmentedImages=[i for i in CNNSegmentedImagesList]
    allSegmentedImageNames= list(first_sheet['SegmentationName'])
#    allSegmentedImageNames=[i if i==i else CNNSegmentedImages[] for idx,i in enumerate(allSegmentedImageNames)]                [nonSegmentedImageIdxs]=CNNSegmentedImages
   
#
#    if CNNSegmentedImages:
#        for idx,i in enumerate(allSegmentedImageNames):
#            if i!=i:
#                allSegmentedImageNames[idx]=CNNSegmentedImages[0][len(path+segmentationFileName)+1:]
#                CNNSegmentedImages=CNNSegmentedImages[1:]
    allSegmentedImageNames=map(str,allSegmentedImageNames)
    imageToProcess= allSegmentedImageNames[eval(lineNumb)]
    import pickle
    featureList=[]    
    ti2=time()
    cnnSegmentationTime=ti2-ti
    timeList=[]
    for idx,segmentedImageName in enumerate([imageToProcess]):
        print 'Working on image: ', segmentedImageName
        segmentedImagePath=path+segmentationFileName +'/'+ segmentedImageName
        if os.path.isfile(segmentedImagePath):
#            print 'code is running'
#            features= np.array((1,2,3,5,67,56,6))
#            score= 25
            provided_center = np.array([first_sheet['CenterColumn'][eval(lineNumb)],first_sheet['CenterRow'][eval(lineNumb)]])
            if not all(np.isnan(provided_center)):
                cntr = provided_center
            else:
                cntr = odd.find_optic_disc_center(segmentedImageName,path+segmentationFileName +'/')[0]
            print 'Disc center found!, finding the vessel tree information..'
            finalPoints, scatime,protime = tc.trace(segmentedImagePath,cntr)
            branches=mst.vesselTree(finalPoints,cntr)
            splines= cs.fitSplines(branches,finalPoints)  
            print 'Starting to extract features..'            
            features=fes.extractFeatures(splines,segmentedImagePath,cntr)

            featureList+=[segmentedImageName,features]
#            score, contribution = gs.generateScore(features,args.predictPlus)
#            outputScoreDf=outputScoreDf.append(pd.DataFrame([[segmentedImageName,score]+contribution],columns=['SegmentedImageName','Score']+featureList ), ignore_index=True)
#        else:
#            print 'Segmented Image ' ,  segmentedImageName, ' could not be found'
#
#    if args.saveDebug: 
#        outputFeatureDf.to_excel(featureWriter, sheet_name='Sheet1',index=False,header=False)
#        featureWriter.save()
#    f=open('timeList'+imageNames+'.txt','wb')
#    pickle.dump(timeList,f)
#    pickle.dump(cnnSegmentationTime,f)
#    f.close()
    f2=open(path+str(lineNumb)+ featureFileName+'.txt','wb')
#    pickle.dump(timeList,f2)
    pickle.dump(featureList,f2)
    f2.close()    
#    outputScoreDf.to_excel(scoreWriter, sheet_name='Sheet1',index=False)
#    scoreWriter.sheets['Sheet1'].set_column('A:Z',20)
#    scoreWriter.save()
  

    


        
