########################################
#################### External Data
########################################

import cv2, math, time
import numpy as np
from sklearn import svm
from skimage import feature

########################################
#################### External Data
########################################

class C:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[31m"
    MAG = "\33[35m"
    YELLOW = "\33[33m"
    END = "\033[0m"
    
FirstImage = True
StartTime = time.time()

########################################
#################### Subroutines
########################################

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

def EnhanceData(OriginalImage, EnhancementFlag): 
    global FirstImage
    Final = OriginalImage
    if (EnhancementFlag[0]):
        Final = GaussianBlur(Final)
        if FirstImage: print(C.MAG, "Enhancement Enabled:", C.END, "Gaussian Blur")
    if (EnhancementFlag[1]):
        Final = Sharpen(Final)
        if FirstImage: print(C.MAG, "Enhancement Enabled:", C.END, "Sharpening")
    if (EnhancementFlag[2]):
        Final = MedianFilter(Final, 3)
        if FirstImage: print(C.MAG, "Enhancement Enabled:", C.END, "Median Noise Filter")
    if (EnhancementFlag[3]):
        Final = ApplyColorWeight(Final, (.8, .3, .1))
        if FirstImage: print(C.MAG, "Enhancement Enabled:", C.END, "Color Weight (", C.RED, ".8,", C.GREEN, ".3,", C.BLUE, ".1", C.END, ")")

    return Final

def IsolateForegroundFromBackground(OriginalImage, Test=False):
    OriginalImage = ContrastStretch(OriginalImage); 
    ThresholdImage = GlobalThreshold(OriginalImage, 130); 
    ThresholdImage = BinInvert(ThresholdImage); 

    Contours = IsolateObjects(OriginalImage, ThresholdImage)

    FinalImage = [0, 0]
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

    return [OriginalImage, ThresholdImage, ContourResult, Final]

def ExtractFeaturesOfData(OriginalImage, FeatureFlag):
    global FirstImage
    FeatureList = []
    ForegroundImage = IsolateForegroundFromBackground(OriginalImage)[3]

    if FeatureFlag[0]: 
        FeatureList.append(Circularity(OriginalImage)[0])
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Circularity/Roundness")
    if FeatureFlag[1]: 
        FeatureList.append(Circularity(OriginalImage)[1])
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Area")
    if FeatureFlag[2]: 
        FeatureList.append(Circularity(OriginalImage)[2])
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Perimeter")
    if FeatureFlag[3]: 
        FeatureList += LBP(ForegroundImage)
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Local Binary Pattern")
    if FeatureFlag[4]: 
        FeatureList.append(HistogramFeatures(ForegroundImage)[1])
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Histogram Mean")
    if FeatureFlag[5]: 
        FeatureList.append(HistogramFeatures(ForegroundImage)[2])
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Histogram Standard Deviation")
    if FeatureFlag[6]: 
        FeatureList.append(HistogramFeatures(ForegroundImage)[3])
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Histogram Entropy")
    if FeatureFlag[7]: 
        FeatureList.append(ColorTotal(ForegroundImage, "R"))
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Color Total: Red")
    if FeatureFlag[8]: 
        FeatureList.append(ColorTotal(ForegroundImage, "G"))
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Color Total: Green")
    if FeatureFlag[9]: 
        FeatureList.append(ColorTotal(ForegroundImage, "B"))
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Color Total: Blue")
    if FeatureFlag[10]: 
        FeatureList.append(EulerNumber(OriginalImage))
        if FirstImage: print(C.CYAN, "Feature Enabled:", C.END, "Euler Number")

    FirstImage = False
    return FeatureList

def RunDataThroughSVM(train_feat, train_label, test_data, LowRange):
    results = []
    clf = svm.SVC(kernel='linear').fit(train_feat, train_label); results.append(clf.predict(test_data))
    clf = svm.SVC(kernel='poly').fit(train_feat, train_label); results.append(clf.predict(test_data))
    clf = svm.SVC(kernel='rbf').fit(train_feat, train_label); results.append(clf.predict(test_data))
    clf = svm.SVC(kernel='sigmoid').fit(train_feat, train_label); results.append(clf.predict(test_data))

    AdjustedResults = []
    for result in results:
        correct = 0
        for item in range(len(result)):
            if (result[item] == 0 and item < LowRange): correct += 1
            elif (result[item] == 1 and item >= LowRange): correct += 1
        AdjustedResults.append(correct)

    return AdjustedResults

def PrintResults(Results, Total):
    global StartTime
    print(C.GREEN, "\n Correct results of LINEAR Kernal:", C.END, str(round(Results[0] / Total * 100, 2)) + '%')
    print(C.GREEN, "Correct results of POLY Kernal:", C.END, str(round(Results[1] / Total * 100, 2)) + '%')
    print(C.GREEN, "Correct results of RBF Kernal:", C.END, str(round(Results[2] / Total * 100, 2)) + '%')
    print(C.GREEN, "Correct results of SIGMOID Kernal:", C.END, str(round(Results[3] / Total * 100, 2)) + '%')
    print(C.YELLOW, "\n] Application Finished Successfully with a duration of:", C.END, str(round(time.time() - StartTime, 2))+ "s\n")

########################################
#################### Feature Methods
########################################

def Circularity(OriginalImage):
    Foreground = IsolateForegroundFromBackground(OriginalImage)[2]
    Contours, _ = cv2.findContours(Foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Contour = Contours[0]
    A = cv2.contourArea(Contour)
    P = cv2.arcLength(Contour, True) + 1e-10
    C = 4 * np.pi * (A / (P * P))
    return [round(C, 4), round(A, 4), round(P, 1)]

def LBP(OriginalImage):
    OriginalImage = cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2GRAY)
    LBP = feature.local_binary_pattern(OriginalImage, 24, 3, method="uniform");
    hist, _ = np.histogram(LBP.ravel(), bins=range(0, 27), range=(0, 27))
    return hist.tolist();

def HistogramFeatures(OriginalImage):
    OriginalImage = cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2GRAY)
    Histogram = cv2.calcHist([OriginalImage], [0], None, [256], [0, 256])
    Histogram = OriginalImage.flatten()
    Mean = sum(Histogram / 256)
    NormalizedHistogram = Histogram / np.sum(Histogram)
    Entropy = -np.sum(NormalizedHistogram * np.log2(NormalizedHistogram + 1e-10))
    StandardDeviation = np.std(Histogram)
    return [Histogram, Mean, Entropy, StandardDeviation]

def ColorTotal(OriginalImage, Color="R"):
    B, G, R = cv2.split(OriginalImage)
    if (Color == "G"): return np.sum(G)
    if (Color == "B"): return np.sum(B)
    return np.sum(R)

def EulerNumber(OriginalImage):
    Thresh = IsolateForegroundFromBackground(OriginalImage)[1]
    _, Labels, Info, _ = cv2.connectedComponentsWithStats(Thresh, connectivity=8)
    ObjectTotal = len(Info) - 1
    HoleTotal = 0
    for Stat in Info[1:]:
        if Stat[cv2.CC_STAT_AREA] < 0: HoleTotal += 1
    return ObjectTotal - HoleTotal

########################################
#################### Utility Methods
########################################

def IsolateObjects(Original, Thresh):
  cnt, grgb = cv2.findContours(Thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  isolatedObjects = []

  for contourIndex in range(len(cnt)):
    isolatedObjects.append(np.ones(Thresh.shape[:2], dtype="uint8") * 255)
    mask = isolatedObjects[contourIndex]
    cv2.drawContours(mask, cnt, contourIndex, 0, -1)
    Thresh = cv2.bitwise_and(Thresh, Thresh, mask=mask)
  
  return isolatedObjects

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

########################################
#################### Enhancement Methods
########################################

def ContrastStretch(OriginalImage):
    min_val = np.min(OriginalImage)
    max_val = np.max(OriginalImage)
    Stretched = cv2.normalize(OriginalImage, None, 0, 255, cv2.NORM_MINMAX)
    return Stretched

def GaussianBlur(OriginalImage, size=5):
   return cv2.GaussianBlur(OriginalImage, (size, size), 0)

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

def Smooth(OriginalImage):
    K = np.array(
       [
            [0, 1, 0],
            [1, 4, 1],
            [0, 1, 0]
        ]
    )

    Smooth = cv2.filter2D(OriginalImage, -1, K)
    return Smooth

def MedianFilter(OriginalImage, Size):
    Filtered = cv2.medianBlur(OriginalImage, Size)
    return Filtered

def ApplyColorWeight(OriginalImage, W=(1, 1, 1)):
    B, G, R = cv2.split(OriginalImage)
    
    R = R * W[0]
    G = G * W[1]
    B = B * W[2]

    NewImage = cv2.merge((
        B.astype(np.uint8), 
        G.astype(np.uint8), 
        R.astype(np.uint8)))
    return NewImage

########################################
#################### Inversion Methods
########################################

def Invert(OriginalImage):
   inverted = np.abs(OriginalImage - 255)
   return inverted

def BinInvert(OriginalImage):
   inverted = cv2.bitwise_not(OriginalImage)
   return inverted