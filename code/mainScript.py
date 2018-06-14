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
import sys
import pickle
sys.path.insert(0,'segmentation_package')
from segment_unet import SegmentUnet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Retinal Image Analysis Code ',\
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('pathToFolder', help='Path to folder which contains images and xlsx file which contains names and disc centers (this folder should be placed in data folder)')
    parser.add_argument('imageNamesXlsFile', help='Name of the xlsx file which contains names and disc centers') 
    parser.add_argument('scoreFileName', help='Name of the xlsx File in which output scores are stored')
    parser.add_argument('--saveDebug',default=1 ,help="If you want to save the segmented image, features, centerline points of the vessel, this should be 1.")
    parser.add_argument('--featureFileName',default='Features', help='Name of the xlsx File in which output features are stored')
    parser.add_argument('--predictPlus',default=1, help='The system would predict image severity score in Plus or higher category by default. If you want to predict severity score in Normal category, this should be 0.')
    args = parser.parse_args()
       
    path= args.pathToFolder
    if path[-1] != "/":
        path+= "/"     
    path = '../data/' + path
    
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

    isPlus = args.predictPlus 
    
    segmentationFileName= 'Segmented'    
    xl = pd.ExcelFile(path+imageNames)
    first_sheet = xl.parse(xl.sheet_names[0])

    with open('../parameters/featureList.txt','rb') as f:
        featNames= pickle.load(f)
        
    if args.saveDebug:            
        outputFeatureDf = pd.DataFrame([],columns=['Image Name']+ featNames )#'SegmentedImageName', 'Features'
        featureWriter = pd.ExcelWriter(path+featureFileName, engine='xlsxwriter')
        outputFeatureDf.to_excel(featureWriter, sheet_name='Sheet1')

#    featureList=['DistanceToDiscCenter','CumulativeTortuosityIndex','IntegratedCurvature(IC)','IntegratedSquaredCurvature(ISC)'\
#                 ,'ICNormalizedbyChordLength','ICNormalizedbyCurveLength','ISCNormalizedbyChordLength','ISCNormalizedbyCurveLength','NormofAcceleration',\
#                 'Curvature','AverageSegmentDiameter','AveragePointDiameter']

    outputScoreDf = pd.DataFrame([],columns=['SegmentedImageName','Score'])
    scoreWriter = pd.ExcelWriter(path+scoreFileName, engine='xlsxwriter')
    outputScoreDf.to_excel(scoreWriter, sheet_name='Sheet1')


    
    # Process CNN Segmentations
    nonSegmentedImageIdxs=[idx for idx,i in enumerate(first_sheet['SegmentationName']) if i!=i]
    providedSegmentationIdxs = [idx for idx,i in enumerate(first_sheet['SegmentationName']) if i==i]
    imgCNNPathList=[path+str(first_sheet['ImageName'][i]) for i in nonSegmentedImageIdxs]        
    cnnSegmentationsPath = path + segmentationFileName
    cnnModel = '../parameters/unet'
    unetCNN = SegmentUnet(cnnModel,cnnSegmentationsPath)
    CNNSegmentedImagesList, imgNamesDoneList, imgNameFailedList = unetCNN.segment_batch(imgCNNPathList)
    CNNSegmentedImages=[i for i in CNNSegmentedImagesList]
    fail_names = [i.split('/')[-1] for i in imgNameFailedList]
    allSegmentedImageNames= list(first_sheet['SegmentationName'])
#    allSegmentedImageNames=[i if i==i else CNNSegmentedImages[] for idx,i in enumerate(allSegmentedImageNames)]                [nonSegmentedImageIdxs]=CNNSegmentedImages
    updated_segmentation_list = []
    for idx, image in enumerate(first_sheet['ImageName']):
        if first_sheet['SegmentationName'][idx] == first_sheet['SegmentationName'][idx]:
            updated_segmentation_list += [str(first_sheet['SegmentationName'][idx])]
        elif not str(image)[:-3] + 'png' in fail_names:
            updated_segmentation_list += [str(image)[:-3] + 'png']
        else:
            updated_segmentation_list += [None]
        
        
    allSegmentedImageNames=map(str,updated_segmentation_list)
    for idx,segmentedImageName in enumerate(allSegmentedImageNames):
        print 'Working on image: ', segmentedImageName
        segmentedImagePath=path+segmentationFileName +'/'+ segmentedImageName
        if os.path.isfile(segmentedImagePath):
            provided_center = np.array([first_sheet['CenterColumn'][idx],first_sheet['CenterRow'][idx]])
            if not all(np.isnan(provided_center)):
                cntr = provided_center
            else:
                cntr = odd.find_optic_disc_center(segmentedImageName,path+segmentationFileName +'/')[0]
#            print cntr
            print '---> Disc center found!, finding the vessel tree information..'
            finalPoints, scatime,protime = tc.trace(segmentedImagePath,cntr, args.saveDebug)
            branches=mst.vesselTree(finalPoints,cntr)
            splines= cs.fitSplines(branches,finalPoints)  
            print '---> Starting to extract features..'            
            features=fes.extractFeatures(splines,segmentedImagePath,cntr)

            if args.saveDebug:
                tempDataFrame=pd.DataFrame([[segmentedImageName]+ list(features)], columns=['Image Name']+ featNames)
                outputFeatureDf=outputFeatureDf.append(tempDataFrame, ignore_index=True)
            score = gs.generateScore(features,isPlus)
            outputScoreDf=outputScoreDf.append(pd.DataFrame([[segmentedImageName,score]],columns=['SegmentedImageName','Score'] ), ignore_index=True)
        else:
            print ('Segmentation file of image "' + str(first_sheet['ImageName'][idx]) +
            '" could not be found! (Either segmentation code failed or segmanted image is not provided in the folder)')

    if args.saveDebug: 
        outputFeatureDf.to_excel(featureWriter, sheet_name='Sheet1',index=False)
        featureWriter.save()

    outputScoreDf.to_excel(scoreWriter, sheet_name='Sheet1',index=False)
    scoreWriter.sheets['Sheet1'].set_column('A:Z',20)
    scoreWriter.save()
  

    


        