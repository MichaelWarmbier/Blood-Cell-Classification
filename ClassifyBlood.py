#################### External Data

import ExternalMethods as M
import cv2
from skimage.filters import (threshold_niblack, threshold_sauvola)

#################### Internal Data

Total_T_White = 50
Total_T_Red = 100
Total_White = 11
Total_Red = 20

#################### Main

TRed, TWhite, Red, White = M.CollectData(Total_T_Red, Total_T_White, Total_Red, Total_White)
AllImages = Red

for Image in AllImages: M.IsolateForegroundFromBackground(Image, True);
