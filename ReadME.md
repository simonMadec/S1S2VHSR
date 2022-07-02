
Readme  
For Sentinel-1 Sentinel-2 Training

Input: 
A root folder 

With "Ground_truth", "Sentinel-1", "Sentinel-2", Folders

In each of this folder:
    "Training" "Validation" "Test" 

    In each of this folder:
        numpy fileS with the name : 
            Sentinel-2_Test_split_0.npy
        or  : Sentinel-2_Test_split_1.npy
        or in "Training" : Sentinel-2_Training_split_1.npy

        or in "Training" : Sentinel-2_Training_split_1.npy
        or in "Ground_truth" "Training" : Ground_truth_Training_split_1.npy

How to create this numpy files ? :

For S1 it is better to do a log normalisation 
ts_array = -10*np.log(ts_array)

Then apply a min max normalisation 
ts_array = (ts_array - min)/(max - min)

min and max should be save for the inferene .. 

For S2 we also apply a min max normalisation for each features

Size of the numpy :
For S1,S2
9*9*(numberOfDates*numberOfFeatures)
ex : 9x9x62 sor S1 (two features and 31 dates)
ex : 9x9x138 sor S2 (six features and 23 dates)

For Ground_Truth
path_size*x (x must be more than 1) the second column should be the id of the classes
recommanded path_size : 256
ex : 
nn.shape
(256, 4)
nn[0]
array([2.8096e+04, 6.0000e+00, 2.3232e+04, 2.4972e+04])
6 is the id ID or between 0 and number of (classes -1)

Then the numpy and before doing the inference
