#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:35:26 2018

@author: veysiyildiz

detect the optic disc center of the input image 

"""

from keras.models import model_from_json
import loadImages

def find_optic_disc_center(image_name,image_folder):
    image_data=loadImages.readImages([image_name], image_folder)
         
    # load json and create model
    json_file = open('../parameters/unet_opticDisc/optic_disc_detector_model.json', 'rb')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../parameters/unet_opticDisc/optic_disc_detector_model.h5")
     #

    prediction =  loaded_model.predict(image_data)[:,0,0,:]
    
    return prediction
    
    


