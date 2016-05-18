# Assignment: Prediction Assignment Writeup
Alice Sung  
May 17, 2016  


This is the course project of Coursera Procaticl Machine Learning of Data Science. 
The goal is to predict the manner in which people did the exersise using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 

# Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here:   http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).  

# Data Processing

## Data Download  
Download the data, read csv and handle the NA strings.  


```r
setwd("~/Documents/GitHub/CP8ML")
urltr <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlts <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(urltr, destfile="./training.csv", method="curl")
download.file(urlts, destfile="./testing.csv", method="curl")
```

Meaningless values were presented as "NA", "", or "#DIV/0!" in the file. Replace all of them by on single value "NA" so that they can be faster filtered-out later.  


```r
tr <- read.csv("training.csv",na.strings = c("NA", "", "#DIV/0!")) 
ts <- read.csv("testing.csv",na.strings = c("NA", "", "#DIV/0!"))
```


## Data Cleansing  

Observe the data. (results hided.)

```r
str(tr)
summary(tr)
names(tr)

# Here I choose not to print out the results, so that there's not too many lines taking all the space of this report.
```

Clear unuseful columns, such as name and timestamp. Get rid of NAs, for they contributes little to the prediction model model.   

```r
# clear unusful data, such as names, time...
tr <- tr[,8:160]
ts <- ts[,8:160]
# get rid of columns which have no values
feature <- colSums(is.na(tr))==0
tr <- tr[,which(feature)]
ts <- ts[,which(feature)]
```


# Modeling  

Here I choose 3 kinds of machine learning methods to apply to the data: Random Forest, Gradient Boosting Machine, Linear Discriminant Analysis.   

## Data Partitioning  
Split the data into training set and testing set by 70% and 30%. Training set is used to train the model and the other is used validate the models' accuracy.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.4
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.4
```

```r
library(gbm)
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
set.seed(0514)
inTraining <- createDataPartition(tr$classe,p=0.7,list=F)
training <- tr[inTraining,]
testing <- tr[-inTraining,]
```

## Model 1: Random Forest (RF)  
Use Random Forest to predict "classe" by all other 52 variables.  

```r
model_rf <- train(classe~.,data=training, method="rf")
model_rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.57%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3906    0    0    0    0 0.000000000
## B   13 2640    5    0    0 0.006772009
## C    0   18 2377    1    0 0.007929883
## D    0    0   32 2217    3 0.015541741
## E    0    0    1    5 2519 0.002376238
```

## Model 2: Gradient Boosting Machine (GBM)  
Use Gradient Boosting Machine to predict "classe" by all other 52 variables.   

```r
model_gbm <- train(classe~.,data=training, method="gbm")
```

```
## Loading required package: plyr
```

```r
model_gbm$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 52 predictors of which 43 had non-zero influence.
```

## Model 3: Linear Discriminant Analysis (LDA)   
Use Linear Discriminant Analysis to predict "classe" by all other 52 variables.  

```r
model_lda <- train(classe~.,data=training, method="lda")
```

```
## Loading required package: MASS
```

## Validation
Applying validation by using the 3 models to predict respectively, and compute the confusion matrix to check the accuracy.  

```r
pre_rf <- predict(model_rf,testing)
pre_gbm <- predict(model_gbm,testing)
pre_lda <- predict(model_lda,testing)
cfm_rf <- confusionMatrix(testing$classe,pre_rf); cfm_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    7 1128    4    0    0
##          C    0   10 1014    2    0
##          D    0    0   14  949    1
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9934         
##                  95% CI : (0.991, 0.9953)
##     No Information Rate : 0.2855         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9916         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9903   0.9826   0.9979   0.9991
## Specificity            0.9998   0.9977   0.9975   0.9970   1.0000
## Pos Pred Value         0.9994   0.9903   0.9883   0.9844   1.0000
## Neg Pred Value         0.9983   0.9977   0.9963   0.9996   0.9998
## Prevalence             0.2855   0.1935   0.1754   0.1616   0.1840
## Detection Rate         0.2843   0.1917   0.1723   0.1613   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9978   0.9940   0.9900   0.9974   0.9995
```

```r
cfm_gbm <- confusionMatrix(testing$classe,pre_gbm); cfm_gbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1648   19    3    2    2
##          B   29 1075   33    1    1
##          C    0   24  980   17    5
##          D    0    3   27  924   10
##          E    2   14    8    5 1053
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9652          
##                  95% CI : (0.9602, 0.9697)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9559          
##  Mcnemar's Test P-Value : 0.004632        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9815   0.9471   0.9324   0.9737   0.9832
## Specificity            0.9938   0.9865   0.9905   0.9919   0.9940
## Pos Pred Value         0.9845   0.9438   0.9552   0.9585   0.9732
## Neg Pred Value         0.9926   0.9874   0.9854   0.9949   0.9963
## Prevalence             0.2853   0.1929   0.1786   0.1613   0.1820
## Detection Rate         0.2800   0.1827   0.1665   0.1570   0.1789
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9877   0.9668   0.9615   0.9828   0.9886
```

```r
cfm_lda <- confusionMatrix(testing$classe,pre_lda); cfm_lda
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393   40  119  109   13
##          B  184  751  114   42   48
##          C  108   86  684  128   20
##          D   58   45  110  705   46
##          E   41  199   80   97  665
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7133          
##                  95% CI : (0.7016, 0.7249)
##     No Information Rate : 0.3031          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6368          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7808   0.6699   0.6179   0.6522   0.8396
## Specificity            0.9315   0.9186   0.9284   0.9461   0.9181
## Pos Pred Value         0.8321   0.6594   0.6667   0.7313   0.6146
## Neg Pred Value         0.9071   0.9220   0.9129   0.9236   0.9736
## Prevalence             0.3031   0.1905   0.1881   0.1837   0.1346
## Detection Rate         0.2367   0.1276   0.1162   0.1198   0.1130
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.8562   0.7942   0.7732   0.7991   0.8789
```


## Model Selection
Compare the model accuracy.

```r
cfm_rf$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9933730      0.9916163      0.9909517      0.9952834      0.2854715 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

```r
cfm_gbm$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##    0.965165675    0.955934236    0.960159558    0.969703465    0.285301614 
## AccuracyPValue  McnemarPValue 
##    0.000000000    0.004632177
```

```r
cfm_lda$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.133390e-01   6.368052e-01   7.015969e-01   7.248692e-01   3.031436e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   2.835910e-53
```

Random Forest has the best accuracy.  
- Overall Accuracy for rf : 0.9944  
- Overall Accuracy for gbm: 0.9635  
- Overall Accuracy for lda: 0.7133  

# Conclusion  
Since Random Forest has the best accuracy (99.44%), hence apply this model to predict the test set downloaded. By submitting the answers to week 4 quiz, they were all correct!!

```r
pre20 <- predict(model_rf,ts); pre20
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



