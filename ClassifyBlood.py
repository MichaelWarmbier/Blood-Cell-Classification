#################### External Data

import ExternalMethods as M
import cv2

#################### Internal Data

Total_T_White = 50
Total_T_Red = 100
Total_White = 11
Total_Red = 20

class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    END = '\033[0m'


#################### Main

TRed, TWhite, Red, White = M.CollectData(Total_T_Red, Total_T_White, Total_Red, Total_White)

# EnhanceData()

Train_Feat = []; Train_Label = []; Test_Data = []
UsedFeatures = [
    "Circularity" * 1, 
    "Area" * 1, 
    "Perimeter" * 1, 
    "Local Binary Pattern" * 1, 
    "Histogram Mean" * 1, 
    "Histogram Entropy" * 1, 
    "Histogram Standard Deviation" * 1,
    "Total Red Values" * 1,
    "Total Green Values" * 1,
    "Total Blue Values" * 1,
    "Euler Number" * 1
    ]

print(C.BLUE, "Activated Features:\n", UsedFeatures, C.END)

for Image in TRed:
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Train_Feat.append(Features)
    Train_Label.append(0)

for Image in TWhite:
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Train_Feat.append(Features)
    Train_Label.append(1)

for Image in Red:
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Test_Data.append(Features)

for Image in White:
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Test_Data.append(Features)


Results = M.RunDataThroughSVM(Train_Feat, Train_Label, Test_Data, Total_Red)
M.PrintResults(Results, Total_Red + Total_White, C)