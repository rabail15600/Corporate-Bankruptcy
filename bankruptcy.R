# Reading the data
library(pROC)
library(caret)
library(rpart)
library(rpart.plot)
library(ranger)
library(vip)
library(xgboost)
library(Matrix)
library(glmnet)
library(ROSE)
library(ggplot2)
library(reshape2)
library(dplyr)
library(broom)
library(car)

setwd("F:/MSDS/Applied Statistics for Data Science")
co.data <- read.csv("Data/bankruptcy.csv")
str(co.data)

# Renaming the target
colnames(co.data)[1]="Bankruptcy"

# Encoding categorical variables
cat.cols <- c("Bankruptcy", "Liability.Assets.Flag", "Net.Income.Flag")
num.cols <- which((lapply(co.data, class))=="numeric")
co.data[,cat.cols] <- lapply(co.data[,cat.cols], factor)

# Dealing with missing values appearing as zero
table(is.na(co.data))
co.data[,num.cols][co.data[,num.cols]==0] <- NA
table(is.na(co.data))

# Remove columns that have NA's more than 50% and Tax because it can't be zero
which(colMeans(is.na(co.data))>0.1)
co.data.t <- subset(co.data, select=-c(Tax.rate..A.,
                                       Long.term.Liability.to.Current.Assets))

# Remove net income flag and Liability.Assets.Flag
lapply(co.data[,cat.cols], unique)
table(co.data$Liability.Assets.Flag)
bankrupt.data <- subset(co.data.t, select=-c(Net.Income.Flag, Liability.Assets.Flag))
num.cols2 <- which((lapply(bankrupt.data, class))=="numeric")
bankrupt.data[,num.cols2][is.na(bankrupt.data[,num.cols2])]=0
table(is.na(bankrupt.data))
dim(bankrupt.data)

# Proportion of bankrupt companies is rare
table(bankrupt.data$Bankruptcy)/nrow(bankrupt.data)
ggplot(bankrupt.data, aes(Bankruptcy,)) + geom_bar(fill=c("Yellow","Red")) + 
theme_bw()+ggtitle("Bankruptcy is a rare event (0 = No, 1 = Yes)")+
theme(plot.title = element_text(hjust = 0.5))

# Training and testing split in 80-20
set.seed(123457)
train.prop <- 0.80
strats <- bankrupt.data$Bankruptcy
rr <- split(1:length(strats), strats)
idx <- sort(as.numeric(unlist(sapply(rr, function(x) sample(x, length(x)*train.prop )))))
bankrupt.data.train <- bankrupt.data[idx, ]
bankrupt.data.test <- bankrupt.data[-idx, ]

# Check if the proportion of y levels is similar in training and testing
summary(bankrupt.data.train$Bankruptcy)/nrow(bankrupt.data.train)
summary(bankrupt.data.test$Bankruptcy)/nrow(bankrupt.data.test)
summary(bankrupt.data$Bankruptcy)/nrow(bankrupt.data)

# Bankrupt cases are really rate at about 3%
# Therefore, we will under-sample non-bankrupt cases to train the model
under_sample <- ovun.sample(Bankruptcy ~ ., data=bankrupt.data.train,
                            method='under', p=0.1)$data
summary(under_sample$Bankruptcy)/nrow(under_sample)


###############################################################################
# Logistic regression (All predictors)
###############################################################################
# Model using all predictors
model.full <- glm(Bankruptcy~ ., data=under_sample, family=binomial("probit"))
summary(model.full)

# Trying to filter variables manually to see if they converge
# First 7 converge
model.full <- glm(Bankruptcy~ ., data=under_sample[,1:7], family=binomial("probit"))

# Removing Pre.tax.net.Interest.Rate
under_sample1 <- subset(under_sample, select=-c(8))
model.full <- glm(Bankruptcy~ ., data=under_sample1[,1:8], family=binomial("probit"))

# Removing After.tax.net.Interest.Rate
under_sample2 <- subset(under_sample1, select=-c(8))
model.full <- glm(Bankruptcy~ ., data=under_sample2[,1:8], family=binomial("probit"))

# Removing Non.industry.income.and.expenditure.revenue
under_sample3 <- subset(under_sample2, select=-c(8))
model.full <- glm(Bankruptcy~ ., data=under_sample3[,1:8], family=binomial("probit"))

# Removing Continuous.interest.rate..after.tax.
under_sample4 <- subset(under_sample3, select=-c(8))
model.full <- glm(Bankruptcy~ ., data=under_sample4[,1:12], family=binomial("probit"))

# Removing Net.Value.Per.Share..B.
under_sample5 <- subset(under_sample4, select=-c(12))
model.full <- glm(Bankruptcy~ ., data=under_sample5[,1:12], family=binomial("probit"))

# Removing Net.Value.Per.Share..A.
under_sample6 <- subset(under_sample5, select=-c(12))
model.full <- glm(Bankruptcy~ ., data=under_sample6[,1:12], family=binomial("probit"))

# Removing Net.Value.Per.Share..C.
under_sample7 <- subset(under_sample6, select=-c(12))
model.full <- glm(Bankruptcy~ ., data=under_sample7[,1:12], family=binomial("probit"))

# Removing Persistent.EPS.in.the.Last.Four.Seasons
under_sample8 <- subset(under_sample7, select=-c(12))
model.full <- glm(Bankruptcy~ ., data=under_sample8[,1:14], family=binomial("probit"))

# Removing Operating.Profit.Per.Share..Yuan...
under_sample9 <- subset(under_sample8, select=-c(14))
model.full <- glm(Bankruptcy~ ., data=under_sample9[,1:14], family=binomial("probit"))

# Removing Per.Share.Net.profit.before.tax..Yuan...
under_sample10 <- subset(under_sample9, select=-c(14))
model.full <- glm(Bankruptcy~ ., data=under_sample10[,1:14], family=binomial("probit"))

# Removing "Realized.Sales.Gross.Profit.Growth.Rate
under_sample11 <- subset(under_sample10, select=-c(14))
model.full <- glm(Bankruptcy~ ., data=under_sample11[,1:21], family=binomial("probit"))

# Removing Cash.Reinvestment..
under_sample12 <- subset(under_sample11, select=-c(21))
model.full <- glm(Bankruptcy~ ., data=under_sample12[,1:21], family=binomial("probit"))

# Removing Current.Ratio
under_sample13 <- subset(under_sample12, select=-c(21))
model.full <- glm(Bankruptcy~ ., data=under_sample13[,1:30], family=binomial("probit"))

# Removing Net.profit.before.tax.Paid.in.capital
under_sample14 <- subset(under_sample13, select=-c(30))
model.full <- glm(Bankruptcy~ ., data=under_sample14[,1:30], family=binomial("probit"))

# Removing Inventory.and.accounts.receivable.Net.value
under_sample15 <- subset(under_sample14, select=-c(30))
model.full <- glm(Bankruptcy~ ., data=under_sample15[,1:32], family=binomial("probit"))

# Removing Average.Collection.Days
under_sample16 <- subset(under_sample15, select=-c(32))
model.full <- glm(Bankruptcy~ ., data=under_sample16[,1:33], family=binomial("probit"))

# Removing Fixed.Assets.Turnover.Frequency
under_sample17 <- subset(under_sample16, select=-c(33))
model.full <- glm(Bankruptcy~ ., data=under_sample17[,1:33], family=binomial("probit"))

# Removing Net.Worth.Turnover.Rate..times.
under_sample18 <- subset(under_sample17, select=-c(33))
model.full <- glm(Bankruptcy~ ., data=under_sample18[,1:33], family=binomial("probit"))

# Removing Revenue.per.person
under_sample19 <- subset(under_sample18, select=-c(33))
model.full <- glm(Bankruptcy~ ., data=under_sample19[,1:33], family=binomial("probit"))

# Removing Operating.profit.per.person
under_sample20 <- subset(under_sample19, select=-c(33))
model.full <- glm(Bankruptcy~ ., data=under_sample20[,1:34], family=binomial("probit"))

# Removing Working.Capital.to.Total.Assets
under_sample21 <- subset(under_sample20, select=-c(34))
model.full <- glm(Bankruptcy~ ., data=under_sample21[,1:36], family=binomial("probit"))

# Removing Cash.Total.Assets
under_sample22 <- subset(under_sample21, select=-c(36))
model.full <- glm(Bankruptcy~ ., data=under_sample22[,1:42], family=binomial("probit"))

# Removing Current.Liabilities.Liability
under_sample23 <- subset(under_sample22, select=-c(42))
model.full <- glm(Bankruptcy~ ., data=under_sample23[,1:43], family=binomial("probit"))

# Removing Current.Liabilities.Equity
under_sample24 <- subset(under_sample23, select=-c(43))
model.full <- glm(Bankruptcy~ ., data=under_sample24[,1:43], family=binomial("probit"))

# Removing Retained.Earnings.to.Total.Assets
under_sample25 <- subset(under_sample24, select=-c(43))
model.full <- glm(Bankruptcy~ ., data=under_sample25[,1:43], family=binomial("probit"))

# Removing Total.income.Total.expense
under_sample26 <- subset(under_sample25, select=-c(43))
model.full <- glm(Bankruptcy~ ., data=under_sample26[,1:46], family=binomial("probit"))

# Removing Working.capitcal.Turnover.Rate
under_sample27 <- subset(under_sample26, select=-c(46))
model.full <- glm(Bankruptcy~ ., data=under_sample27[,1:50], family=binomial("probit"))

# Removing Current.Liability.to.Equity
under_sample28 <- subset(under_sample27, select=-c(50))
model.full <- glm(Bankruptcy~ ., data=under_sample28[,1:56], family=binomial("probit"))

# Removing Net.Income.to.Total.Assets
under_sample29 <- subset(under_sample28, select=-c(56))
model.full <- glm(Bankruptcy~ ., data=under_sample29[,1:56], family=binomial("probit"))

# Removing Total.assets.to.GNP.price
under_sample30 <- subset(under_sample29, select=-c(56))
model.full <- glm(Bankruptcy~ ., data=under_sample30[,1:56], family=binomial("probit"))

# Removing No.credit.Interval
under_sample31 <- subset(under_sample30, select=-c(56))
model.full <- glm(Bankruptcy~ ., data=under_sample31[,1:58], family=binomial("probit"))

# Removing Liability.to.Equity
under_sample32 <- subset(under_sample31, select=-c(58))
model.full <- glm(Bankruptcy~ ., data=under_sample32, family=binomial("probit"))

# Removing Equity.to.Liability
under_sample33 <- subset(under_sample32, select=-c(60))

# Model without warning, using 59 out of 92 predictors
model.full <- glm(Bankruptcy~ ., data=under_sample33, family=binomial("probit"))
summary(model.full)

# Removing variables that have alias coefficients
attributes(alias(model.full)$Complete)$dimnames[[1]]
under_sample34 <- subset(under_sample33, select=-c(25)) # Removing Net.worth.Assets
model.full <- glm(Bankruptcy~ ., data=under_sample34, family=binomial("probit"))

# Check for multicollinearity
cor.pred <- cor(under_sample34[,2:58])
cor.pred[upper.tri(cor.pred)] <- 0
diag(cor.pred) <- 0
under_sample35 <- under_sample34[, !apply(cor.pred, 2, function(x) any(abs(x) > 0.95, na.rm = TRUE))]
under_sample36 <- cbind(Bankruptcy=under_sample34$Bankruptcy, under_sample35)

# Warning again
model.full <- glm(Bankruptcy~ ., data=under_sample36[,1:46],family=binomial("probit"))

# Removing Cash.Flow.to.Liability
under_sample37 <- subset(under_sample36, select=-c(46))
model.full <- glm(Bankruptcy~ ., data=under_sample37,family=binomial("probit"))

# Checking VIFs to make multicollinearity does not exist
vif(model.full)

# Drop Realized.Sales.Gross.Margin
under_sample38 <- subset(under_sample37, select=-c(Realized.Sales.Gross.Margin))
model.full <- glm(Bankruptcy~ ., data=under_sample38,family=binomial("probit"))
vif(model.full)

# Drop Allocation.rate.per.person
under_sample39 <- subset(under_sample38, select=-c(Allocation.rate.per.person))
model.full <- glm(Bankruptcy~ ., data=under_sample39,family=binomial("probit"))
vif(model.full)

# Drop Regular.Net.Profit.Growth.Rate
under_sample40 <- subset(under_sample39, select=-c(Regular.Net.Profit.Growth.Rate))
model.full <- glm(Bankruptcy~ ., data=under_sample40,family=binomial("probit"))
vif(model.full)

# Drop Current.Liability.to.Liability
under_sample41 <- subset(under_sample40, select=-c(Current.Liability.to.Liability))
model.full <- glm(Bankruptcy~ ., data=under_sample41,family=binomial("probit"))
vif(model.full)

# Drop Working.Capital.Equity
under_sample42 <- subset(under_sample41, select=-c(Working.Capital.Equity))

# Still warning!
model.full <- glm(Bankruptcy~ ., data=under_sample42[,1:42],family=binomial("probit"))

under_sample43 <- subset(under_sample42, select=-c(42))
model.full <- glm(Bankruptcy~ ., data=under_sample43[,1:43],family=binomial("probit"))

under_sample44 <- subset(under_sample43, select=-c(43))
model.full <- glm(Bankruptcy~ ., data=under_sample44[,1:43],family=binomial("probit"))

under_sample45 <- subset(under_sample44, select=-c(43))
model.full <- glm(Bankruptcy~ ., data=under_sample45[,1:43],family=binomial("probit"))

under_sample46 <- subset(under_sample45, select=-c(43))
model.full <- glm(Bankruptcy~ ., data=under_sample46,family=binomial("probit"))
vif(model.full)
summary(model.full)

# full model deviance
model.full$deviance

# Subset testing data
p.all <- which(colnames(bankrupt.data.test) %in% names(under_sample46))
bankrupt.data.test <- bankrupt.data.test[,p.all]


###############################################################################
# Logistic regression
# Using Stepwise selection
###############################################################################
model.null <- glm(Bankruptcy~1, data=under_sample46, family=binomial(link="probit"))
summary(model.null)

model.step <- step(model.null, list(lower=formula(model.null),
                                    upper=formula(model.full)),
                   direction="forward",trace=0)
summary(model.step) # Selects 12 predictors out of 59
vif_1 <- vif(model.step)
id <- which(colnames(under_sample46) %in% names(vif_1))
df.step <- under_sample46[,c(1, id)]

glm(formula = Bankruptcy ~ ., family = binomial(link = "probit"), 
    data = df.step)

df.step1 <- df.step[,1:12]

# Final model after feature selection
model.step <- glm(formula = Bankruptcy ~ ., family = binomial(link = "probit"), 
                  data = df.step1)

# Stepwise model evaluation

predict.train.step <- predict(model.step, newdata=df.step1, 
                              type="response")
predict.test.step <- predict(model.step, newdata=bankrupt.data.test, 
                             type="response")
(table.train.step <- table(df.step1$Bankruptcy, 
                           ifelse(predict.train.step>0.5,1,0)))
(table.test.step <- table(bankrupt.data.test$Bankruptcy, 
                          ifelse(predict.test.step>0.5,1,0)))

# Total accuracy rates
(accuracy.step.train <- round((sum(diag(table.train.step))/sum(table.train.step))*100,2))
(accuracy.step.test <- round((sum(diag(table.test.step))/sum(table.test.step))*100,2))

# Sensitivity, Specificity, and misclassification rate
# Training
sensitivity(table.train.step)
specificity(table.train.step)
100 - accuracy.step.train

#Testing
sensitivity(table.test.step)
specificity(table.test.step)
100 - accuracy.step.test

# Area under the curve
roc.train.step <- roc(df.step1$Bankruptcy, predict.train.step, levels=c(1,0))
auc(df.step1$Bankruptcy, predict.train.step)
roc.test <- roc(bankrupt.data.test$Bankruptcy, predict.test.step, levels=c(1,0))
auc(bankrupt.data.test$Bankruptcy, predict.test.step)

###############################################################################
# Decision tree
###############################################################################
# Growing the tree using training data
fit.allp <- rpart(Bankruptcy~.,method="class", data=under_sample46,
                  control=rpart.control(minsplit=1, cp=0.001))

## print cross-validation (cv) results
printcp(fit.allp)

# plot cross-validation (cv) results
plotcp(fit.allp) # visualize cross-validation results - see figure

# Finding the value of cp with smallest xerror
(cp= fit.allp$cptable[which.min(fit.allp$cptable[,"xerror"]),"CP"])
(xerr = fit.allp$cptable[which.min(fit.allp$cptable[,"xerror"]),"xerror"])

# plot of tree
rpart.plot(fit.allp, extra="auto")

# Evaluation metrics on training
train_df <- data.frame(actual=under_sample46$Bankruptcy, pred=NA)
train_df$pred <- predict(fit.allp, newdata=under_sample46, type="class")

# Confusion matrix
(train_conf_matrix_base <- table(train_df$actual,train_df$pred))

# Sensitivity, Specificity, misclassification rate, and accuracy rate
sensitivity(train_conf_matrix_base)
specificity(train_conf_matrix_base)
(train_conf_matrix_base[1,2] + train_conf_matrix_base[2,1])/sum(train_conf_matrix_base)
round((train_conf_matrix_base[1,1]+train_conf_matrix_base[2,2])/sum(train_conf_matrix_base)*100,2)

# Evaluation metrics on testing
test_df <- data.frame(actual=bankrupt.data.test$Bankruptcy, pred=NA)
test_df$pred <- predict(fit.allp, newdata=bankrupt.data.test, type="class")

# Confusion matrix
(test_conf_matrix_base <- table(test_df$actual,test_df$pred))

# Sensitivity, Specificity, misclassification rate, and accuracy rate
sensitivity(test_conf_matrix_base)
specificity(test_conf_matrix_base)
(test_conf_matrix_base[1,2] + test_conf_matrix_base[2,1])/sum(test_conf_matrix_base)
round((test_conf_matrix_base[1,1]+test_conf_matrix_base[2,2])/sum(test_conf_matrix_base)*100,2)

# AUC for training and test
roc.train1 <- roc(under_sample46$Bankruptcy, as.numeric(train_df$pred), levels=c(1,0))
auc(under_sample46$Bankruptcy, as.numeric(train_df$pred))
roc.test1 <- roc(bankrupt.data.test$Bankruptcy, as.numeric(test_df$pred), levels=c(1,0))
auc(bankrupt.data.test$Bankruptcy, as.numeric(test_df$pred))

###############################################################################
# Pruning the tree
###############################################################################
pfit.allp <- prune(fit.allp, cp=0.02272727)

# plot of the tree
rpart.plot(pfit.allp, extra = "auto")

# Evaluation metrics on training
train_df1 <- data.frame(actual=under_sample46$Bankruptcy, pred=NA)
train_df1$pred <- predict(pfit.allp, newdata=under_sample46, type="class")

# Confusion matrix
(train_conf_matrix_pruned <- table(train_df1$actual,train_df$pred))

# Sensitivity, Specificity, misclassification rate, and accuracy rate
sensitivity(train_conf_matrix_pruned)
specificity(train_conf_matrix_pruned)
(train_conf_matrix_pruned[1,2] + train_conf_matrix_pruned[2,1])/sum(train_conf_matrix_pruned)
round((train_conf_matrix_pruned[1,1]+train_conf_matrix_pruned[2,2])/sum(train_conf_matrix_pruned)*100,2)

# Evaluation metrics on testing
test_df1 <- data.frame(actual=bankrupt.data.test$Bankruptcy, pred=NA)
test_df1$pred <- predict(pfit.allp, newdata=bankrupt.data.test, type="class")

# Confusion matrix
(test_conf_matrix_pruned <- table(test_df1$actual,test_df1$pred))

# Sensitivity, Specificity, misclassification rate, and accuracy rate
sensitivity(test_conf_matrix_pruned)
specificity(test_conf_matrix_pruned)
(test_conf_matrix_pruned[1,2] + test_conf_matrix_pruned[2,1])/sum(test_conf_matrix_pruned)
round((test_conf_matrix_pruned[1,1]+test_conf_matrix_pruned[2,2])/sum(test_conf_matrix_pruned)*100,2)

# AUC for training and test
roc.train2 <- roc(under_sample46$Bankruptcy, as.numeric(train_df1$pred), levels=c(1,0))
auc(under_sample46$Bankruptcy, as.numeric(train_df1$pred))
roc.test2 <- roc(bankrupt.data.test$Bankruptcy, as.numeric(test_df1$pred), levels=c(1,0))
auc(bankrupt.data.test$Bankruptcy, as.numeric(test_df1$pred))

###############################################################################
# RandomForest using ranger
###############################################################################
num.pred <- 41
(mtry.1 <- floor(sqrt(num.pred)))
fit.rf.ranger <- ranger(Bankruptcy ~ ., data=under_sample46, 
                        importance='impurity', mtry=mtry.1)
print(fit.rf.ranger)

# Default OOB (Out of bag) RMSE - measure of how well this random forest with mtry = xx
(default_rmse <- sqrt(fit.rf.ranger$prediction.error))

# Variable importance plot
(v1 <- vi(fit.rf.ranger)) # print variable importance
vip(fit.rf.ranger)   # plot the variable importance as a bar graph

# Evaluation metrics on training
train_df2 <- data.frame(actual=under_sample46$Bankruptcy, pred=NA)
predict.train2 <- predict(fit.rf.ranger, data=under_sample46)
train_df2$pred <- predict.train2$predictions

# Confusion matrix
(train_conf_matrix_rf <- table(train_df2$actual,train_df2$pred))

# Sensitivity, Specificity, misclassification rate, and accuracy rate
sensitivity(train_conf_matrix_rf)
specificity(train_conf_matrix_rf)
(train_conf_matrix_rf[1,2] + train_conf_matrix_rf[2,1])/sum(train_conf_matrix_rf)
round((1 - (train_conf_matrix_rf[1,2] + train_conf_matrix_rf[2,1])/sum(train_conf_matrix_rf)),2)

# Evaluation metrics on testing
test_df2 <- data.frame(actual=bankrupt.data.test$Bankruptcy, pred=NA)
predict.test2 <- predict(fit.rf.ranger, data=bankrupt.data.test)
test_df2$pred <- predict.test2$predictions

# Confusion matrix
(test_conf_matrix_rf <- table(test_df2$actual, test_df2$pred))

# Sensitivity, Specificity, missclassification rate, and accuracy rate
sensitivity(test_conf_matrix_rf)
specificity(test_conf_matrix_rf)
(test_conf_matrix_rf[1,2] + test_conf_matrix_rf[2,1])/sum(test_conf_matrix_rf)
round((1 - (test_conf_matrix_rf[1,2] + test_conf_matrix_rf[2,1])/sum(test_conf_matrix_rf)),2)

# AUC of training and testing
roc.train3 <- roc(under_sample46$Bankruptcy, as.numeric(train_df2$pred),levels=c(1,0))
auc(under_sample46$Bankruptcy, as.numeric(train_df2$pred))
roc.test3 <- roc(bankrupt.data.test$Bankruptcy, as.numeric(test_df2$pred), levels=c(1,0))
auc(bankrupt.data.test$Bankruptcy, as.numeric(test_df2$pred))

###############################################################################
# Gradient boosting using  xgboost package
###############################################################################
# Transforming the predictors matrix using one-hot encoding
matrix_predictors.train <- as.matrix(sparse.model.matrix(Bankruptcy ~ .,
                                                         data=under_sample46))[,-1]
matrix_predictors.test <- as.matrix(sparse.model.matrix(Bankruptcy ~ .,
                                                        data = bankrupt.data.test))[,-1]

# Set up features and label in a Dmatrix form for xgboost
# Train dataset
pred.train.gbm <- data.matrix(matrix_predictors.train)

# Converting factor to numeric
bankrupt.data.train.gbm <- as.numeric(as.character(under_sample46$Bankruptcy))
dtrain <- xgb.DMatrix(data=pred.train.gbm, label=bankrupt.data.train.gbm)

# Test dataset
pred.test.gbm <- data.matrix(matrix_predictors.test)

# Converting factor to numeric
bankrupt.data.test.gbm <- as.numeric(as.character(bankrupt.data.test$Bankruptcy))
dtest <- xgb.DMatrix(data=pred.test.gbm, label=bankrupt.data.test.gbm)

# define watchlist
watchlist <- list(train=dtrain, test=dtest)

# define param
param <- list(max_depth = 2, eta = 1, verbose = 0, nthread = 2,
              objective = "binary:logistic", eval_metric = "auc")

# Fit XGBoost model and display training and testing data at each round
model.xgb <- xgb.train(param, dtrain, nrounds=4, watchlist, silent=T)

# Evaluation metrics on training
predict.bankruptcy.train <- predict(model.xgb, pred.train.gbm)
prediction.train.xgb <- as.numeric(predict.bankruptcy.train>0.5)

# Confusion matrix
train_conf_matrix_xgb <- table(bankrupt.data.train.gbm, prediction.train.xgb)

# Sensitivity, Specificity, misclassification rate, and accuracy rate
sensitivity(train_conf_matrix_xgb)
specificity(train_conf_matrix_xgb)
round(1 - ((sum(diag(train_conf_matrix_xgb)))/sum(train_conf_matrix_xgb)),3)
round((sum(diag(train_conf_matrix_xgb))/sum(train_conf_matrix_xgb)),2)

# Evaluation metrics on testing
predict.bankruptcy.test <- predict(model.xgb, pred.test.gbm)
prediction.test.xgb <- as.numeric(predict.bankruptcy.test>0.5)

# Confusion matrix
test_conf_matrix_xgb <- table(bankrupt.data.test.gbm, prediction.test.xgb)

# Sensitivity, Specificity, misclassification rate, and accuracy rate
sensitivity(test_conf_matrix_xgb)
specificity(test_conf_matrix_xgb)
round(1 - ((sum(diag(test_conf_matrix_xgb)))/sum(test_conf_matrix_xgb)),3)
round((sum(diag(test_conf_matrix_xgb))/sum(test_conf_matrix_xgb)),2)

# AUC of training and testing
roc.train4 <- roc(under_sample46$Bankruptcy, prediction.train.xgb,levels=c(1,0))
auc(under_sample46$Bankruptcy, prediction.train.xgb)
roc.test4 <- roc(bankrupt.data.test$Bankruptcy, prediction.test.xgb, levels=c(1,0))
auc(bankrupt.data.test$Bankruptcy, prediction.test.xgb)

# Feature importance
importance_matrix = xgb.importance(colnames(under_sample46[,2:42]), model=model.xgb)
importance_matrix
xgb.plot.importance(importance_matrix[1:10,])









