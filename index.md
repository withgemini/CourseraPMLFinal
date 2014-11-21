Coursera Practical Machine Learning Final Project
========================================================

## Introduction

Using various devices it is now possible to collect a large amount of data about personal activity. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The goal of this project is to predict the manner in which they did the exercise. 

The data for this project come from http://groupware.les.inf.puc-rio.br/har. Data were collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. These six young healthy participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience.

## Data Pre-Processing

The dataset contains 19,622 observations of 160 variables. The outcome variable that the project aims to predict for the test set is "classe." Since the initial number of variables is too large, let's use 2 different functions to remove some variables. First, using nearZeroVar(), we identify the variables that have variance close to zero. Second, we use is.na() combined with other functions to identify the columns that contain more than 1,000 entries (almost 20% of the observations) with NAs. The resulting dataset contains 60 variables. 


```r
testing <- read.csv("pml-testing.csv", header=TRUE)

allData <- read.csv("pml-training.csv", header=TRUE)
removeNearZeroVarColumns <- nearZeroVar(allData)
allData <- subset(allData, select = -c(removeNearZeroVarColumns[-1]) )

removeNAMajorityColumns <- which(colSums(is.na(allData)) > 1000)
allData <- subset(allData, select = -c(removeNAMajorityColumns))
```

In order to perform cross validation to confirm that the model is chosen successfully, we split the resulting dataset into two, training and validation. In our initial attempts to create the model we used p=0.75, which meant that about 75% of data was in the training set. However, the dataset was too large to create a model and the algorithm ran for more than 2 hours. After trying various values we chose p=0.25 which still guaranteed the training set to be around 5,000 observations. 


```r
inTrain <- createDataPartition(y=allData$classe, p=0.25, list=FALSE)
training <- allData[inTrain,]
validation <- allData[-inTrain,]
```

## Model

For the final model we use Principal Components Analysis and Random Forests. Since initial data pre-processing, described earlier, did not eliminate enough variables, Principal Components Analysis enables to narrow down the number of variables further by using weighted combination of predictors instead. Random Forest seems to be the most appropriate method since it has high accuracy. The given dataset should not be affected by the negative side effects typical for Random Forests. While speed is an issue for this method, having training dataset of about 5,000 works well. Since the interpretability is not the goal of this project, we can ignore this side effect. The issue of overfitting is dealt with during cross validation. 


```r
modFit <- train(classe~., data=training, method="rf", prox=TRUE, preProcess="pca", na.omit=TRUE)
```

## Cross Validation

During data pre-processing the intial set was split into training and validation set, so that the model can be built on the training set and then evaluated against the validation set. The cross validation is especially important for the chosen model given the potential side effect of Random Forests overfitting. 

We use predict() funciton in order to predict the outcome variable classe for the validation set. Using confusionMatrix() function we compare the predicted outcome with the one that was actually obeserved in the validation set. 


```r
confusionMatrix(validation$classe,predict(modFit, validation))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4084   73   27    0    1
##          B  118 2644   84    1    0
##          C    0   88 2463   15    0
##          D    0    1  111 2278   22
##          E    0    0    2   62 2641
## 
## Overall Statistics
##                                         
##                Accuracy : 0.959         
##                  95% CI : (0.956, 0.962)
##     No Information Rate : 0.286         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.948         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.972    0.942    0.917    0.967    0.991
## Specificity             0.990    0.983    0.991    0.989    0.995
## Pos Pred Value          0.976    0.929    0.960    0.944    0.976
## Neg Pred Value          0.989    0.986    0.982    0.994    0.998
## Prevalence              0.286    0.191    0.183    0.160    0.181
## Detection Rate          0.278    0.180    0.167    0.155    0.179
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.981    0.963    0.954    0.978    0.993
```

The accuracy being 96% suggests that this is a good model. 


## Expected Out-Of-Sample Error

Based on the confusionMatrix() and Accuracy above, the expected out-of-sample error is about 4%.


