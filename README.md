# Blood-Cell-Classification

&nbsp;&nbsp;&nbsp;&nbsp;This project was created for the 2023 Spring Semester Image Processing course at the College of Staten Island (CUNY), lectured by Sos Agaian. Any updates made _after_ the date of May 19th, 2023 were made _after_ the project deadline. 

**Primary Goal**: to create an application for classifying a specific set of images into binary categories. The context of this specific project is _blood cells_. This project is created with several features meant to be tested in various orders in order to best optimize the algorithm for speed and accuracy.

**Secondary Goal**: to display the applications use by extending it to solve a problem. In the context of this specific project, that problem is _object detection_.

<br>

## <p align="center">Image Details</p>

<p align="center"><strong>All images are sourced from <a href="https://www.kaggle.com/datasets/paultimothymooney/blood-cells">Paul T. Mooney on Kaggle</a></strong></p>

&nbsp;&nbsp;&nbsp;&nbsp;A total of **761** images were used. Each image is sourced from the credited image database throguh manually isolation of cells from a large batch of cells. Of the 761 images:

<p align="center">50 were used for <em>training</em> white blood cells<br>
100 were usesd for <em>training</em> red blood cells<br>
210 were used for <em>classification testing</em> of white blood cells<br>
401 were used for <em>classification testing</em> of red blood cells<br></p>

## <p align="center">Installing and Running</p>

To run this application, you must have Python installed on your system as well as the following packages:

- `scikit-image`
- `scikit-learn`
- `opencv-python`
- `numpy`

You can run the application by going to the source folder, opening any **terminal** and running the following command:

`python ClassifyBlood.py`

You can specify features to utilize by adding all eleven flags at the end:

`python ClassifyBlood.py <flags>`

Example:

`python ClassifyBlood.py 0 0 0 1 1 1 1 1 1 1 1`

<br>


You can run the **demo app** by using the following command:

`python DemoApp.py <filename>`

## <p align="center">Application Structure</p>

<p align="center"><img src="https://cdn.discordapp.com/attachments/1065328426032058470/1106099311554613268/Untitled_Diagram.drawio.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;**Step #0**: which features are to be used are determined. Input is either **default**, in which case all features are used, or through **console input**.<br>
&nbsp;&nbsp;&nbsp;&nbsp;**Step #1**: image data is organized into four different lists.<br>
&nbsp;&nbsp;&nbsp;&nbsp;**Step #2**: image data is enhanced. Each enhancement is hard-coded and not modified through input, but may be modified through code.<br>
&nbsp;&nbsp;&nbsp;&nbsp;**Step #3**: image features are extracted based on the aforementioned input. Features are organized into lists. <br>
&nbsp;&nbsp;&nbsp;&nbsp;**Step #4**: features are inserted into SVM four SVM functions associated with different kernals. Results of each test are stored.<br>
&nbsp;&nbsp;&nbsp;&nbsp;**Step #5**: results of each test are converted into percents based on labels determined by order. Results are output onto console.

<br>

## Library Dependencies

**OpenCV**: cv2 image processing functions<br>
**Math**: complex mathematical computations<br>
**Time**: time storage used to determine runtime result<br>
**Numpy**: complex mathematical computations and array manipulation<br>
**Sklearn**: feature collection, SVM algorithm and kernals<br>

<br>

## Internal Data

`class C`: used for color labeling text when printing to onsole.<br>
`FirstImage`: flag used for printing feature inforamtion.<br>
`StartTime`: used to calculate runtime during **Step #5**.<br>

`Total_T_White`: constant used to store total amount of white training images<br>
`Total_T_Red`: constant used to store total amount of red training images<br>
`Total_White`: constant used to store total amount of white testing images<br>
`Total_Red`: constant used to store total amount of red testing images

`Train_Feat`: list used to store all training image features<br>
`Train_Label`: list used to store all trainingimage labels<br>
`Test_Data`: list used to store all test image features<br>

<br>

## File System

`/Blood-Cell-Classification`<br>
\---- `.gitignore`<br>
\---- `ClassifyBlood.py`<br>
\---- `ExternalMethods.py`<br>
\---- `README.md`<br>
\---- `DemoApp.py`<br>
\---- `RESULTS.txt`<br>
\---- `/Data`<br>
\----\---- `/Data_Examples`<br>
\----\---- `/Red`<br>
\----\---- `/Training_Red`<br>
\----\---- `/White`<br>
\----\---- `/Training_White`<br>

<br>

`ClassifyBlood.py`: primary application script and main routine.<br>
`ExternalMethods.py`: library of custom methods and other external libraries.<br>
`DemoApp.py`: Demostration of the applications use in object detection.<br>
`RESULTS.txt`: _manual_ entry of results from testing.<br>
`/Data`: collection of all source images organized into folders.<br>

<br>

## Code Documentation

<br>

### **Subroutines**

`CollectData(T_Group_1, T_Group_2, Group_1, Group_2)`<br>
**Description**: opens four groups of images and stores them lists.<br>
**Arguments**: `T_Group_1`: Number, `T_Group_2`: Number, `Group_1`: Number, `Group_2`: Number<br>
**Returns**: Image List, Image List, Image List, Image List<br>

`EnhanceData(OriginalImage, EnhancementFlag)`<br>
**Description**: enhances an individual image.<br>
**Arguments**: `OriginalImage`: Image, `EnhancementFlag`: Boolean List<br>
**Returns**: Image<br>

`IsolateForegroundFromBackground(OriginalImage, Test=False)`<br>
**Description**: isolates largest foreground object from the rest of an image.<br>
**Arguments**: `OriginalImage`: Image, `Test`: Boolean=True<br>
**Returns**: [Image, Image, Image, Image]

`ExtractFeaturesOfData(OriginalImage, FeatureFlag)`<br>
**Description**: extracts features and returns them in a list.<br>
**Arguments**: `OriginalImage`: Image, `FeatureFlag`: Boolean List<br>
**Returns**: Number List

`RunDataThroughSVM(train_feat, train_label, test_data, LowRange)`<br>
**Description**: runs image data through four SVMs and returns results.<br>
**Arguments**: `train_feat`: 2D Number List, `train_label`: Number List, `test_data`: 2D Number List, `LowRange`: Number<br>
**Returns**: Number List

`PrintResults(Results, Total)`<br>
**Description**: prints results of application.<br>
**Arguments**: `Results`: Number List, `Total`: Number<br>
**Returns**: NoneType

<br>

### **Feature Extraction Methods**

`Circularity(OriginalImage)`<br>
**Description**: calculates area, perimeter and circularity of an image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: [Number, Number, Number]

`LBP(OriginalImage)`<br>
**Description**: calculates local binary pattern of an image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: Number List

`HistogramFeatures(OriginalImage)`<br>
**Description**: calculates histogram, mean, standard deviation and entropy of an image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: [Number List, Number, Number, Number]

`ColorTotal(OriginalImage, Color="R")`<br>
**Description**: calculates total values of a color within an image.<br>
**Arguments**: `OriginalImage`: Image, `Color`: String="R"<br>
**Returns**: Number

`EulerNumber(OriginalImage)`<br>
**Description**: calculates number of holes in an image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: Number

<br>

### **Utility Methods**

`IsolateObjects(OriginalImage, Thresh)`<br>
**Description**: isolates objects within image into separate contour images.<br>
**Arguments**: `OriginalImage`: Image, `Thresh`: Image<br>
**Returns**: Image List

`GlobalThreshold(OriginalImage, T=255/2)`<br>
**Description**: thresholds an image globally.<br>
**Arguments**: `OriginalImage`: Image, `T`: Integer=127.5<br>
**Returns**: Image

`GetBinaryToatl(OriginalImage)`<br>
**Description**: returns total amount of black and white pixels in a binary image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: [Number, Number]

`ConcatenateImages(AllImages)`<br>
**Description**: concatenates images horizontally into one image.<br>
**Arguments**: `AllImages`: Image List<br>
**Returns**: Image

`ApplyThresholdMask(OriginalImage, BinaryImage)`<br>
**Description**: multiplies an image by a mask.<br>
**Arguments**: `OriginalImage`: Image, `BinaryImage`: Image<br>
**Returns**: Image

<br>

### **Enhancement Methods**

`ContrastStretch(OriginalImage)`<br>
**Description**: applies contrast stretching enhancement to image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: Image

`Sharpen(OriginalImage)`<br>
**Description**: applies sharpening enhancement to image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: Image

`Smooth(OriginalImage)`<br>
**Description**: applies smoothing enhancement to image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: Image

`MedianFilter(OriginalImage, Size)`<br>
**Description**: applies median noise filter enhancement to image.<br>
**Arguments**: `OriginalImage`: Image, `Size`: Number<br>
**Returns**: Image

`ApplyColorWeight(OriginalImage, W=(1, 1, 1))`<br>
**Description**: applies weighted percents to the colorspace of an image.<br>
**Arguments**: `OriginalImage`: Image, `W`: Number Tuple<br>
**Returns**: Image

<br>

### **Inversion Methods**

`Invert(OriginalImage)`<br>
**Description**: inverts a grayscale image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: Image

`BinInvert(OriginalImage)`<br>
**Description**: inverts a binary image.<br>
**Arguments**: `OriginalImage`: Image<br>
**Returns**: Image

## <p align="center">Final Results</p>

<p align="center">
    <img src="https://media.discordapp.net/attachments/1065328426032058470/1106128502488563773/image.png">
    <img src="https://cdn.discordapp.com/attachments/1065328426032058470/1106128502882836491/image.png">
    <img src="https://cdn.discordapp.com/attachments/1065328426032058470/1106128503247732806/image.png">
    <img src="https://cdn.discordapp.com/attachments/1065328426032058470/1106128604502433902/image.png">
    <img src="https://cdn.discordapp.com/attachments/1065328426032058470/1106128604766687254/image.png">
</p>