#################### External Data

import cv2
import numpy as np

#################### Subroutines

def CollectData(T_Group_1, T_Group_2, Group_1, Group_2):

    # Initialize Data
    TrainingRed = []
    TrainingWhite = []
    Red = []
    White = []

    # Store Data in Groups
    for imgIndex in range(T_Group_1):
        TrainingRed.append(cv2.imread("Data/Training_Red/T_Red_" + str(imgIndex) + ".jpg"))

    for imgIndex in range(T_Group_2):
        TrainingWhite.append(cv2.imread("Data/Training_White/T_White_" + str(imgIndex) + ".jpg"))

    for imgIndex in range(Group_1):
        Red.append(cv2.imread("Data/Red/Red_" + str(imgIndex) + ".jpg"))
        
    for imgIndex in range(Group_2):
        White.append(cv2.imread("Data/White/White_" + str(imgIndex) + ".jpg"))
   
    return TrainingRed, TrainingWhite, Red, White

def EnhanceData(): 
   return

def IsolateForegroundFromBackground(OriginalImage, Test=False):
    OriginalImage = ContrastStretch(OriginalImage); 
    ThresholdImage = GlobalThreshold(OriginalImage, 137); 
    ThresholdImage = BinInvert(ThresholdImage); 

    Contours = IsolateObjects(OriginalImage, ThresholdImage);

    FinalImage = [0, 0];
    for contourIndex in range(len(Contours)):
        if (GetBinaryTotal(Contours[contourIndex])[0] > FinalImage[1]):
            FinalImage[0] = Contours[contourIndex]
            FinalImage[1] = GetBinaryTotal(Contours[contourIndex])[0]

    ContourResult = BinInvert(FinalImage[0])
    Final = ApplyThresholdMask(OriginalImage, ContourResult)

    if (Test): 
        TestOutput = ConcatenateImages([OriginalImage, ThresholdImage, ContourResult, Final])
        cv2.imshow('image', TestOutput)
        cv2.waitKey(0)

        

    return Final

def ExtractFeaturesOfData():
   return

def RunDataThroughSVM():
   return

#################### Utility Methods

def IsolateObjects(Original, Thresh):
  cnt, grgb = cv2.findContours(Thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  isolatedObjects = []

  for contourIndex in range(len(cnt)):
    isolatedObjects.append(np.ones(Thresh.shape[:2], dtype="uint8") * 255)
    mask = isolatedObjects[contourIndex]
    cv2.drawContours(mask, cnt, contourIndex, 0, -1)
    Thresh = cv2.bitwise_and(Thresh, Thresh, mask=mask)
  
  return isolatedObjects

def RemoveColorSpace(OriginalImage, rR=False, rG=False, rB=False):
    B, G, R = cv2.split(OriginalImage)
    
    if (rR): R = R * 0
    if (rG): G = G * 0
    if (rB): B = B * 0

    NewImage = cv2.merge((B, G, R))
    return NewImage

def GlobalThreshold(OriginalImage, T = 255/2):
   OriginalImage = cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2GRAY)
   _, thresholded = cv2.threshold(OriginalImage, T, 255, cv2.THRESH_BINARY)
   return thresholded

def GetBinaryTotal(OriginalImage):
    White_Total = np.sum(OriginalImage == 255)
    Black_Total = np.sum(OriginalImage == 0)
    return [Black_Total, White_Total]

def ConcatenateImages(AllImages):
    if (len(AllImages) < 2): return None

    for ImageIndex in range(len(AllImages)):
        if (len(AllImages[ImageIndex].shape) < 3): 
            AllImages[ImageIndex] = cv2.cvtColor(AllImages[ImageIndex], cv2.COLOR_GRAY2BGR)
       
    Final = AllImages[0]

    for ImageIndex in range(1, len(AllImages)):
        Final = np.concatenate((Final, AllImages[ImageIndex]), axis=1)

    return Final

def ApplyThresholdMask(OriginalImage, BinaryImage):
    BinaryImage = cv2.cvtColor(BinaryImage, cv2.COLOR_GRAY2RGB)
    CutOut = cv2.bitwise_and(OriginalImage, BinaryImage)
    return CutOut


#################### Image Enhancement Methods

def ContrastStretch(OriginalImage):
    min_val = np.min(OriginalImage)
    max_val = np.max(OriginalImage)
    Stretched = cv2.normalize(OriginalImage, None, 0, 255, cv2.NORM_MINMAX)
    return Stretched

def GausssianBlur(OriginalImage, size=5):
   return cv2.GaussianBlur(OriginalImage, (size, size), 0)

#################### Filter Methods

def Invert(OriginalImage):
   inverted = np.abs(OriginalImage - 255)
   return inverted

def BinInvert(OriginalImage):
   inverted = cv2.bitwise_not(OriginalImage)
   return inverted

def Sharpen(OriginalImage):
    K = np.array(
       [
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]
    )

    Sharpened = cv2.filter2D(OriginalImage, -1, K)
    return Sharpened

def Blur(OriginalImage):
    K = np.array(
       [
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ]
    )

    Blurred = cv2.filter2D(OriginalImage, -1, K)
    return Blurred
