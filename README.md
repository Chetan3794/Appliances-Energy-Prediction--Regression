# Appliances-Energy-Prediction--Regression


**Abstract:**

As the energy crisis increasing day by day causes various factors to think for in todays scenario. It not only affecting the world energy requirement issues but also affects the economic and social health of any country, mostly in the countries which are going through developing phase. To overcome several related issues, it is necessary to have the outages in order to compensate the load demand. And hence the prediction of energy used by appliances plays a vital role in the study.
	This article illustrates the energy consumed by house and the data has been collected with the help of sensors. The readings have been taken for each 10 min interval for consecutive 4.5 months. Saving of energy can be done by controlling the energy usage. And thus, prediction of usage comes into picture. This study can save the money of consumer as well as if extra energy is generated then it can also fed back to Grid(Also called as regeneration). The article mainly concentrates over the study with the help of machine learning regression technique to find out the energy consumption prediction.

**Keywords:**

Lasso, Ridge, RandomForest, Gradient boosting classifier, ExtraTree Regressor, MLPregressor. 

**Problem Statement:**

The dataset collected contain various information regarding features by which energy is being used. The features incorporated by temperature, humidity, wind speed, pressure, etc. Our main target is to analyze the data and predict the energy consumed by the house using the given dataset. For the same, we need to develop a supervised machine learning model based on a regression approach.  Illustration of the some major feature contained is given below:
date: time🡪 given date time month and day
lights : energy used by lights in Wh

T1 : Temperature given in kitchen area, in Celsius
T2 : Temperature given in living room area, in Celsius
T3 : Temperature mentioned in laundry room area
T4 : Temperature of office room, given in Celsius
T5 : Temperature recorded in  bathroom area, in Celsius
T6 : Temperature given outside the building area particularly (north side), in Celsius
T7 : Temperature provided in ironing room, in Celsius
T8 : Temperature in teenager room 2, in Celsius
T9 : Temperature in parents’ room, in Celsius
T_out : Outside temperature (from Chievres weather station), in °C
Tdewpoint : (from Chievres weather station), 
RH_1 : Kitchen area Humidity %
RH_2 : Living room area Humidity, in %
RH_3 : Laundry room area Humidity, in %
RH_4 : Office room Humidity, in %
RH_5 : Bathroom area Humidity, in %
RH_6 :Outside the building Humidity (north side), in %
RH_7 : Ironing room Humidity, in %
RH_8 : Teenager room 2  Humidity, in %
RH_9 : Parents’ room Humidity, in %
RH_out :Outside Humidity (from Chievres weather station), in %
Pressure : (from Chievres weather station), in mm Hg
Wind speed: (from Chievres weather station), in m/s
Visibility :(from Chievres weather station), in km
Rv1 :Random variable 1, non-dimensional[1]
Rv2 :Random variable 2, non-dimensional[1]
Appliances : Total energy used by appliances, in Wh[1]

**Introduction**

It is necessary have the understanding of energy usage in houses as it will influence the overall load demand and further economic concerns. Data required for the understanding can be captured by using the consumption of various household equipment such as washing machine, televisions, iron box, etc. 
	The usage of electricity in low energy houses can be determined by two important factors such as no of appliances consuming the energy and another is actual amount of usage of energy. Also, inside the house, the energy consumption is affected by various factors such as temperature, humidity, pressure, etc. For that, it is necessary to develop the model which can predict the energy usage as well as indicates the wastage of energy as well as abnormal usage of energy. Overall prediction will surely help in organizing the load demand as well as supply.
To predict the model, we will going to use machine learning regression based approach.

**Dataset division**

There are total 19735 no of rows available. We will go to divide this data into pre-training set, training set, validation set and testing set. The division of data is as follows:
75% of data will be put into training set whereas 25% of data will be put into testing set.
The pretrain set was used to find the best models for the given dataset. We have taken best 6 models using pretest set. Their performance will be compared based on their mean absolute errors.
Once the best 5 models will be obtained, hyperparameters for these models will be tuned and the best parameter will be selected.
4. Training Process:
We will use following 5 regression techniques to train the data:

**1] LASSO Regression:**

It is the regression which uses the shrinkage technique. Which means data values will be shrunk towards the central point. The Lasso regression is very useful when data parameters are few. The acronym “LASSO” stands for Least Absolute Shrinkage and Selection Operator.
Lasso solutions are quadratic programming problems, which are best solved with software (like Matlab). The goal of the algorithm is to minimize:
lasso regression

Which is the same as minimizing the sum of squares with constraint Σ |Bj≤ s (Σ = summation notation). Some of the βs are shrunk to exactly zero, resulting in a regression model that’s easier to interpret.

**2] RIDGE Regression:**

This regression method is mainly used when data having multi-collinearity. The method performs L2 regularization . Whenever multi-collnearity problem occurs, least-square are unbiased and variance are large. Because of it predicted value being far away from the actual values.

Fig. Ridge Regression

**3] Random Forest:**

Random Forest Regression method is a supervised learning algorithm which uses ensemble learning method for regression technique. Ensemble learning method is nothing but a technique which combines predictions from various machine learning algorithms to prepare a more accurate prediction as compare to the single model.

Fig. random Forest regression

**4] Gradient Boosting Classifier:**

This regression technique calculates the difference between the current predicted value and well known correct target value. This residual is then added to the existing model and this pushes the model towards correct values. To improve the performance of the model we can repeat the process again and again.

**5] ExtraTree-regressor:**

It is a type of ensemble learning technique of regression that adds the results of different de-correlated decision trees which are similar to Random Forest Classifier. Extra Tree can also achieve a good or better prediction than the random forest. The main difference between Random Forest and Extra Tree Classifier is as given below:
Extra Tree Classifier algorithm never performs bootstrap aggregation as in the random forest. This means, it takes a random subset of data without any replacement. Hence, nodes are always split on random splits but not on best splits.
In Extra Tree Classifier algorithm randomness doesn’t come from bootstrap aggregating but comes from the random splitting of the data.

Fig. ExtraRegressor

**Libraries used for Programming**

In this article the analysis has been done by using python language and Google-colab platform. Lets have brief introduction about these platforms.
PYTHON:
It is a high level language and widely used for the data indentation. The language supports many programming paradigms which includes functional, structured and object –oriented programming. It consist of various type of libraries and hence this language is also called “battery-included language”. 
	Python regulates dynamic typing as well as a combination of reference counting along with a cycle-detecting garbage collector for the memory management. It also uses dynamic name resolution (late binding), which can binds method and variable names during program execution over a platform.
Google Colab:
The google-colab is primarily a free Jupyter notebook which runs completely in the cloud. Moreover, Colab do not requires any type of setup. In addition to it, the notebooks which one will create can be simultaneously edited by the person’s team members just like in words. The biggest advantage is that the Colab botebook supports various types of machine learning libraries which could be easily accessible in your notebook.
As a programmer, you can perform the following using Google Colab.
Write and execute code in Python
Document your code that supports mathematical equations
Create/Upload/Share notebooks
Import/Save notebooks from/to Google Drive
Import/Publish notebooks from GitHub
Import external datasets e.g. from Kaggle
Integrate PyTorch, TensorFlow, Keras, OpenCV
Free Cloud service with free GPU

PYTHON Libraries:

In this article, we have explained the exploratory data analysis using various python libraries. Some of them are as follows:

**NumPy**

NumPy which is the abbreviation of Numerical Python is the fundamental package useful for numerical computation in Python. It incorporates a powerful N-dimensional array object. It has around 18,000 comments on GitHub along with an active community of 700 contributors. The NumPY is a general-purpose array-processing package which provides high-performance multidimensional objects which are also called as arrays. NumPy also shows the slowness issue partly by giving these multidimensional arrays and partly by providing functions as well as operators that operate satisfactorily on these arrays. 
Features:
Provides fast, precompiled functions for numerical routines
Array-oriented computing for better efficiency
Supports an object-oriented approach
Compact and faster computations with vectorization

**Pandas**

Python data analysis also known as Pandas are very useful for data analysis and visualization. It is the most powerful and efficiently used Python library for data science. It is been used along with NumPy in matplotlib. With large no of comments on GitHub and an active community of great contributors, it is heavily used for data analysis and cleaning. Pandas work fast and possess flexible data structures, like data frame CDs, which are well designed to work with structured data very effectively and intuitively. 
Features:
Eloquent syntax and rich functionalities that gives you the freedom to deal with missing data
Enables you to create your own function and run it across a series of data
High-level abstraction
Contains high-level data structures and manipulation tools

**Matplotlib**

Matplotlib is a powerful as well as beautiful visualization tool. It has more plotting library than pandas and numpy for Python on GitHub and a very active community of almost 700 contributors over globe. Because of the plots and graphs that it produces, it’s efficiently used for data visualization. In addition to it , Matplotlib also gives an object-oriented API, which can be used to attach and install those plots into virtual applications. 

Features:
Usable as a MATLAB replacement, with the advantage of being free and open source 
Supports dozens of backends and output types, which means you can use it regardless of which operating system you’re using or which output format you wish to use
Pandas itself can be used as wrappers around MATLAB API to drive MATLAB like a cleaner
Low memory consumption and better runtime behavior

**Scikit-learn**

It is the most useful library when the question comes about the machine learning. It provides all type of algorithms useful for the machine learning. It is design in such a way that it utilizes Pandas and NumPy
Applications:
clustering
classification
regression
model selection
dimensionality reduction

**Exploratory Data Analysis (EDA):**

The article incorporates various python libraries and functionalities in order have customer point of view visualization.
1] Raw Data Collection:
The data collected from the source contain various information useful for analysis as well as for visualization purpose. There two ways one can utilize the data. By importing the data file into the system and then accessing by writing code for it or by directly mounting the drive. Where it will directly fetch the data from the drive situated file.
2] Data Processing:
Generally speaking, data processing consists of gathering and manipulating data elements to return useful, potentially valuable information. The data processing also subdivided into various operations depending on the operation as follows:
Dealing with missing values:
It checks whether our data contains any missing value is there or not, then it will replace it with the zero. There are no such columns present in the database and hence no need of this operation.
Description of data:
In order to have the proper understanding of data, one can use describe and info function to get the detail idea about the acquired data.
From above processing we have following conclusion:
Feature ranges
Temperature range : -6 to 30 deg
Humidity range : 1 to 100 %
Windspeed range: 0 to 14 m/s
Visibility range: 1 to 66 km
Pressure range: 729 to 772 mm Hg
Appliance Energy Usage range: 10 to 1080 Wh
 
**3] Appliance vs. Energy consumption:**
When we will plot the appliance against the energy consumption graph, we can see that percentage of appliances consumption is less than 200Wh. The representation is as follows:
 
**4] Correlation plot:**

From the correlation plot, we can see that Temperature values T!-T9 have positive correlational values.
For the observation of indoor temperatures, the correlations are looking like high, since the ventilation is characterised by the HRV unit and that minimizes air temperature differences between the rooms. There are four columns which have a high degree of correlation with T9 - T3,T5,T7,T8 also T6 & T_Out has high correlation (both temperatures from outside) . The figure is shown in next page.

**4] Feature Importance:**

From feature importance graph we can analyzed that most important features are - 'RH_out', 'RH_8', 'RH_1', 'T3', 'RH_3'. Whereas least important features will be  - 'T7','Tdewpoint','Windspeed','T1','T5'.


**5] Train and testing procedure pipeline:**

Following methodology has been followed to train and test the model.
Storing of all the algorithm’s present in a list and then  Iterate over the list
The regressor’s random_state was initialized.
The regressor was design to fit on the test as well as training data
The properties of the regressor , Name, timining and score for training and testing set will be stored in a dictionary variable as key-value pairs.


**Results**

After performing various operations following results were obtained.


Name
Train_R2_Score
Test_R2_Score
Train_RMSE_Score
Test_RMSE_Score
0
Lasso
0.000000
-0.000517
1.000000e+00
0.968239
1
Ridge
0.140903
0.109054
9.268748e-01
0.913684
2
RandomForest
0.939403
0.552971
2.461646e-01
0.647200
3
GradientBoostingClassifier
0.329343
0.214281
8.189363e-01
0.858033
4
ExtraTreeRegressor 
1.000000
0.599255
1.326785e-15
0.612780
5
mlpregressor 
0.438139
0.346108
7.495736e-01
0.782750






**Conclusion**




It is clearly seen that best results for test set is being given by Extra Tree Regressor with R2 score of 0.599255
Least RMSE score is also given by Extra Tree Regressor which is 0.612780
Lasso regression model was not giving good result and hence proven to be the worst model.

Parameter Tuning and observation:
Depending on parameter tunning we can conclude that
Best possible parameter combination are - 'max_depth' is  100, max_features is 'sqrt', 'n_estimators' is 260 and random state is 40
Training set R2 score of 1.0 shows the overfitting on training set
Using hyperparameter tuning the R2 score can be improved from 0.59 to 0.60 of the Test set
Test set RMSE score is 0.60 which is get improved from 0.61 achieved using hyperparameter tuning.

**Reference:**
 
HackerRank
GeeksforGeeks
Analytics Vidhya
