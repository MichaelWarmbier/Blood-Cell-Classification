########################################
#################### External Data
########################################

import ExternalMethods as M
from sklearn import svm
import cv2
import numpy as np
import sys

########################################
#################### Internal Data
########################################

class C:
    ERR = "\033[31m"
    INFO = "\33[33m"
    END = "\033[0m"

########################################
#################### Main Routine
########################################

if (len(sys.argv) > 1): OriginalImage = cv2.imread(sys.argv[1])
else: print(C.ERR, "ERROR:", C.END, "No image selected. Run the application with a location attached."); exit()

Total_T_White = 50
Total_T_Red = 100
Train_Feat = []; Train_Label = []; Test_Data = []
Train1, Train2, _, _ = M.CollectData(100, 50, 0, 0, False)
UsedEnhancements = [1, 1, 1, 0]
UsedFeatures = [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

OriginalImage = M.ContrastStretch(OriginalImage)
Objects = M.IsolateObjects(OriginalImage, 255 - M.GlobalThreshold(OriginalImage, 130))
print("\nTotal Objects:", len(Objects), "\n")

# Red
for Image in Train1:
    Image = M.EnhanceData(Image, UsedEnhancements)
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Train_Feat.append(Features)
    Train_Label.append(0)

# White
for Image in Train2:
    Image = M.EnhanceData(Image, UsedEnhancements)
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Train_Feat.append(Features)
    Train_Label.append(1)

# Test
for Object in Objects:
    Image = M.EnhanceData(Object, UsedEnhancements)
    Object = M.ApplyThresholdMask(OriginalImage, 255 - Object)
    Features = M.ExtractFeaturesOfData(Object, UsedFeatures)
    Test_Data.append(Features)

clf = svm.SVC(kernel='linear').fit(Train_Feat, Train_Label); Results = clf.predict(Test_Data)

ResultImage = 0
for V in range(len(Results)):
    if (Results[V]):    
        contours, _ = cv2.findContours(255 - Objects[V], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ResultImage = cv2.drawContours(OriginalImage, contours, -1, (0, 255, 0), 2)


cv2.imshow("", ResultImage)
cv2.waitKey(0)

