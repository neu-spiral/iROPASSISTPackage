#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:44:49 2017

@author: veysiyildiz
"""


'''
for the giving branches create splines
'''

import numpy as np
from scipy import interpolate

def fitSplines(allBranches, finalPoints):
    finalPointsList= list(finalPoints.T)
    csList=[]
    for m in range(len(allBranches)):
        if len(allBranches[m])>5:
            branch=np.concatenate([finalPointsList[i[0]].reshape(2,1) for i in allBranches[m]],1)
            leng=branch.shape[1]
            cs=interpolate.CubicSpline(range(leng),np.array(branch).T,extrapolate=False)
            csList += [cs]
    return csList











