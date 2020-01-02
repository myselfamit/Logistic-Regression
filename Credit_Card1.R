Credit<-read.csv("F:/R and Data Science/Logistics Regression/Credit card case study/Credit_Card1.csv")

# Structure of data type
View(Credit)
str(Credit)

# Header and Univariate Analysis
names(Credit)
attach(Credit)
summary(Credit)

# Data Conversion
Credit$target<-as.factor(Credit$target)
table(Credit$target)
Credit$Dummy<-as.factor(ifelse(Credit$Gender=='M',1,0))
Credit$Gender<-NULL

str(Credit)

## simple Logistics Regression
logit <- glm(target ~ balance, data=Credit,family='binomial')
summary(logit)
anova(logit,test='Chisq')

###Prediction try with  2000
testing<-data.frame(balance=2000)
testing.probs <-predict(logit, testing, type='response')
testing.probs


###Multiple Logistic Regression
set.seed(45)
library(caret)
Train <- createDataPartition(Credit$target, p=0.7, list=FALSE)
training <- Credit[ Train, ]
testing <- Credit[ -Train, ]

# Data Pre-Processing 

# Missing Value 
sapply(training,function(x) sum(is.na(x)))

# For Outlier 

# For Income Variable
boxplot(training$income)

# treatment of outlier for balance
boxplot(training$balance)
summary(training$balance)
upper<-1162+1.5*IQR(training$balance);upper
training$balance[training$balance > upper]<-upper
boxplot(training$balance)
summary(training$balance)

logit <- glm(target ~ income + balance + Dummy,family='binomial',
             data=training)
summary(logit)

# model including all variable 
#relevel(Pclass,ref = 2)
logit2 <- step(glm(target ~ income + balance +Dummy,
                   family='binomial', data=training),direction = "both")
summary(logit2)
anova(logit2,test='Chisq')

# accuracy of model
Acc(logit2)

# odds Ratio
exp(coef(logit2))

# To get Coefficent of model(B0,B1)
logit2$coefficients

# Mathematical calculation check

# Manual Prediction for Male
y=-11.517966250+0.005726163*2113.01902+0.847245479*1
a<-exp(-y)
b<-1+a
c<-1/b
c

# Manual Prediction for Female
y=-11.517966250+0.005726163*2578.469+0.847245479*0
a<-exp(-y)
b<-1+a
c<-1/b
c

## Prediction on testing data
testing$probs <-predict(logit2, testing, type='response')
testing$Predict<-as.factor(ifelse(testing$probs>0.70,1,0))

# Accuracy of testing data
table(testing$Predict, testing$target)
confusionMatrix(testing$target,testing$Predict)

### Roc Curve
library(ROCR)
# Make predictions on training set
predictTrain = predict(logit2,testing, type="response")
# Prediction function
ROCRpred = prediction(predictTrain, testing$target)
# Performance function
ROCRperf = performance(ROCRpred, "tpr", "fpr")
# Plot ROC curve
plot(ROCRperf)
library(ROCR)
pred = prediction(testing$probs, testing$target)
as.numeric(performance(pred, "auc")@y.values)
