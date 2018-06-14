# IROP-ASSIST-Package


***
![alt text](IROP-ASSIST-Package/data/example/Segmented/normal.png?raw=true "Title")

## Description
iROP ASSIST is a Python code package for computer-based ROP image analysis. It can be used to generate a severity score from an original image or from the manually segmented mask. Its main purpose is to test the computer-based system developed by iROP ASSIST team on the larger set of images.

This documentation contains the instructions on how to run the code 

#### Running The Code

### Example Usage 

Type following line in the 'Anaconda Prompt' or 'Terminal':
```
python mainScript.py "example/" "exampleImages.xlsx" "scoresOfExampleImages.xlsx" 
```
This code evaluates  the features for all images which are listed in the `exampleImages.xlsx`  file and located in the `example` folder which should be located in the 'data' folder.
At the end, the scores would be saved in :
```
../data/example/scoresOfExampleImages.xlsx
```
Detailed usage for the code is provided at the end of the document.

### Instructions

#### Package Requirements
+ Anaconda 4.4.0 Python 2.7 version 64-bit
+ Please create an environment using the provided 'environment.yml' file. Please check [this link](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for instructions about how to create an environment from an environment.yml file.

## Detailed Usage

The mainScript could produce the severity scores for multiple images in the same folder.

```
mainScript( pathToFolder, imageNamesXlsxFile, scoreFileName, 
saveDebug=1, featureFileName='Features', predictPlus=1)
```
Inputs to mainScript are as follows:

| Option | Description |
| ------ | ----------- |
| pathToFolder |  A string denoting the name of the folder (located under 'data' folder) containing all the images need to be processed and xlsx file which has the image information.  |
| imageNamesXlsxFile | A string denoting the '.xlsx' filename in the folder. In this Excel spreadsheet, first column is the image name ending with 'bmp','png',etc. Second column is the corresponding segmentation name. If segmentation images are provided, they should be located under 'Segmented' file under the folder where the color images and imageNames.xlsx file are located. The third and fourth column are the disc center for the image in (column, row) order. Note that imageNames column is required for code to run, the other columns are optional (if not provided, the system will run additional scripts to determine those information) |
| scoreFileName |  A string denoting the score file name , that would contain the image names and its corresponding severity scores. |
| saveDebug (optional) | If 1 (default) the system will save the debug files (features, vessel centerlines). If 0 is provided, debug files will not be saved.|
|featureFileName (optional) | A string denoting the '.xlsx' file name, that would contain the image names and its features.|
|predictPlus (optional) | If 1 (default) the system will create severity score from Plus vs Not Plus classifier. If 0 is provided, the system will create severity score from Normal vs Not Normal classifier (Please look at the refrences for details). |

## Feature and Score Generation

All the features generated are based on 6DD crop size. The severity score (0-100) is for predicting Reference Standard Diagnosis (RSD) in Plus vs Not Plus or Normal vs Not Normal Category. 

Detailed description of features can be found the paper. 

### Team:
+ Veysi Yildiz (yildiz@ece.neu.edu)
+ Peng Tian (pengtian@ece.neu.edu)
+ Ilkay Yildiz (yildizi@ece.neu.edu),
+ Yuan Guo (guo.yu@husky.neu.edu),
+ Stratis Ioannidis (ioannidis@ece.neu.edu),
+ Jennifer Dy (jdy@ece.neu.edu),
+ Deniz Erdogmus (erdogmus@ece.neu.edu)

# Reference
---------
--Ataer-Cansizoglu, Esra. "Retinal Image Analytics: A Complete Framework from Segmentation to Diagnosis." Order No. 3701264 Northeastern University, 2015. Ann Arbor: ProQuest. Web. 19 Dec. 2016.
--Bas E, Ataer-Cansizoglu E, Kalpathy-Cramer J, Erdogmus D. Retinal Vasculature Segmentation using Principal Spanning Forests. IEEE International Symposium on Biomedical Imaging (ISBI). 2012; 1792-1795.

