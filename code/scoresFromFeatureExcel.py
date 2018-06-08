#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:49:15 2018

@author: veysiyildiz
"""

'''
calculate score from features excel

'''

import pandas as pd
import generateScore as gs
import numpy as np

feature_file_path = '../data/featsTreatedImages.xlsx'
output_excel_path = '../data/scores_of_featureFile.xlsx'

xl = pd.ExcelFile(feature_file_path)
first_sheet = xl.parse(xl.sheet_names[0])

imageNames = list(first_sheet['Image Name'])


outputScoreDf = pd.DataFrame([],columns= ['Image Name']+ ['Score'])
scoreWriter = pd.ExcelWriter(output_excel_path, engine='xlsxwriter')
outputScoreDf.to_excel(scoreWriter, sheet_name='Sheet1')
  
for idx,i in enumerate(imageNames):
    outputScoreDf=outputScoreDf.append(pd.DataFrame([[i] + [gs.generateScore(np.array(first_sheet.iloc[[idx]])[0,1:-1],1)] ],columns = ['Image Name'] + ['Score']), ignore_index=True)
    
outputScoreDf.to_excel(scoreWriter, sheet_name='Sheet1',index=False)
scoreWriter.sheets['Sheet1'].set_column('A:Z',20)
scoreWriter.save()
 








