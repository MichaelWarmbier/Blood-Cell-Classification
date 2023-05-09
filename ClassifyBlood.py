########################################
#################### External Data
########################################

import ExternalMethods as M

########################################
#################### Internal Data
########################################

Total_T_White = 50
Total_T_Red = 100
Total_White = 170
Total_Red = 181

class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    END = '\033[0m'

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

UsedEnhancements = [
        "Blurring" * 0,
        "Sharpening" * 0,
        "Noise Reduction" * 0,
        "Color Weight" * 0
    ]

########################################
#################### Main
######################################## 
print(C.BLUE, "Activated Features:\n", UsedFeatures, C.END)

#################### Step #1: Organize Data Into Lists
TRed, TWhite, Red, White = M.CollectData(Total_T_Red, Total_T_White, Total_Red, Total_White)

#################### Step #2: Enhance/Improve Data
#################### Step #3: Collect Feature Information

for Image in TRed:
    Image = M.EnhanceData(Image, UsedEnhancements)
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Train_Feat.append(Features)
    Train_Label.append(0)

for Image in TWhite:
    Image = M.EnhanceData(Image, UsedEnhancements)
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Train_Feat.append(Features)
    Train_Label.append(1)

for Image in Red:
    Image = M.EnhanceData(Image, UsedEnhancements)
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Test_Data.append(Features)

for Image in White:
    Image = M.EnhanceData(Image, UsedEnhancements)
    Features = M.ExtractFeaturesOfData(Image, UsedFeatures)
    Test_Data.append(Features)

#################### Step #4: Apply Multiply SVM Kernals
Results = M.RunDataThroughSVM(Train_Feat, Train_Label, Test_Data, Total_Red)

#################### Step #5: Display Results
M.PrintResults(Results, Total_Red + Total_White, C)