#################### External Data

import ExternalMethods as M
import cv2

#################### Internal Data

Total_T_White = 50
Total_T_Red = 100
Total_White = 11
Total_Red = 20

#################### Main

TRed, TWhite, Red, White = M.CollectData(Total_T_Red, Total_T_White, Total_Red, Total_White)

# EnhanceData()

Train_Feat = []; Train_Label = []; Test_Data = []
UsedFeatures = [1, 1, 1, 1, 1, 1, 1]

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
M.PrintResults(Results, Total_Red + Total_White)