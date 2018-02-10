#Load packages
library(ggplot2)
library(plyr)
library(dplyr)
library(class)
library(tree)
library(randomForest)
library(ROCR)


#Read data
adult <- read.csv("C:/Users/Shon/Documents/College/PSTAT 131/Project/adult.data.csv")

#Add column labels
adult.names <- c("Age", 
                 "Workingclass", 
                 "Final_Weight", 
                 "Education", 
                 "Education_num", 
                 "Marital_Status", 
                 "Occupation", 
                 "Relationship", 
                 "Race", 
                 "Sex", 
                 "Capital_gain",
                 "Capital_loss",
                 "Hours_per_week",
                 "Native_country",
                 "Income")
colnames(adult) <- adult.names

#Show dataset
str(adult)

## Data Visualizations
ggplot(adult, aes(x=Sex y=Income)) + 
  geom_bar(aes(fill = Income), stat="Identity", position="dodge")

ggplot(adult, aes(x=Income)) + 
  geom_bar(stat="count")

#Recode variables
adult$Income <- ifelse(adult$Income == " <=50K", 0, 1)
adult$Income <- as.factor(adult$Income)


#Remove education_num and Native_country
adult <- subset(adult, select = -c(Education_num, Native_country))


#Count and remove missing values
adult[adult == " ?"] <- NA
sum(is.na(adult))

adult <- na.omit(adult)
adult <- data.frame(adult)

#Check for class imbalance
summary(adult$Income)

#Create training and test set
set.seed(1)
adult.GT50k <- subset(adult, adult$Income == 1)
adult.LT50k <- subset(adult, adult$Income == 0)

adult.GT50k.indices <- sample(1:nrow(adult.GT50k), 2000)
adult.LT50k.indices <- sample(1:nrow(adult.LT50k), 2000)

adult.GT50k.train <- adult.GT50k[adult.GT50k.indices,]
adult.LT50k.train <- adult.LT50k[adult.LT50k.indices,]
adult.train <- rbind(adult.GT50k.train, adult.LT50k.train)
adult.train <- adult.train[sample(nrow(adult.train)),]

GT50k.rows <- row.names(adult.GT50k.train)
LT50k.rows <- row.names(adult.LT50k.train)
GT50k.rows <- as.numeric(GT50k.rows)
LT50k.rows <- as.numeric(LT50k.rows)

adult.sub <- adult[-GT50k.rows,]
adult.sub <- adult.sub[-LT50k.rows,]

set.seed(1)
test.indices <- sample(1:nrow(adult.sub), 1000)
adult.test <- adult.sub[test.indices,]

summary(adult.train$Occupation)
summary(adult.test$Occupation)

Ytrain <- adult.train$Income
Xtrain <- adult.train %>% select(-Income)
Ytest <- adult.test$Income
Xtest <- adult.test %>% select(-Income)


########## Decision Trees #######################################################
erate <- function(predicted.value, true.value){
  return(mean(true.value!=predicted.value))
}

#Fit tree on entire dataset
tree.full <- tree(Income ~ ., data=adult)

#Plot tree
plot(tree.full)
text(tree.full, pretty = 0, cex = .8, col = "red")
title("Classification Tree Built on Full Adult Dataset")

#Fit model on training set
tree.adult <- tree(Income ~ ., data=adult.train)
tree.adult <- tree(Income ~ ., 
                   data = adult.train, 
                   control = tree.control(4000, 
                                          mincut = 5, 
                                          minsize = 10, 
                                          mindev = .003)) 

#Plot tree
plot(tree.adult)
text(tree.adult, pretty = 0, cex = .8, col = "red")
title("Unpruned Decision Tree of size 22")

#Predict on training and test set
tree.pred.train <- predict(tree.adult, adult.train, type="class")
tree.pred.test <- predict(tree.adult, adult.test, type="class")

#Calculate train and test error on tree
tree.errors <- data.frame(train.error = erate(tree.pred.train, Ytrain), 
                          test.error = erate(tree.pred.test, Ytest))
tree.errors

####Conduct 10-fold cross-validation to prune the tree
cv = cv.tree(tree.adult, FUN=prune.misclass, K=10)

# Best size
best.cv = cv$size[which.min(cv$dev)]
best.cv

#Prune tree
tree.adult.pruned <- prune.misclass(tree.adult, best=best.cv)

#Plot pruned tree
plot(tree.adult.pruned)
text(tree.adult.pruned, pretty=0, col = "blue", cex = .8)
title("Pruned Decision Tree of size 14")

#Predict pruned tree on training and test set
tree.pred.train.pruned <- predict(tree.adult.pruned, adult.train, type="class")
tree.pred.test.pruned <- predict(tree.adult.pruned, adult.test, type="class")

#Calculate train and test error on pruned tree
tree.errors.pruned <- data.frame(train.error = erate(tree.pred.train.pruned, Ytrain),
                                 test.error = erate(tree.pred.test.pruned, Ytest))
tree.errors.pruned

summary(tree.adult.pruned)

#Decision tree ROC
Tree.prob <- predict(tree.adult.pruned, adult.test)
Tree.prob2 <- data.frame(Tree.prob[,2])
Tree.pred <- prediction(Tree.prob2, adult.test$Income)
Tree.perf <- performance(Tree.pred, measure="tpr", x.measure="fpr")

#AUC
auc = performance(Tree.pred, "auc")@y.values
auc

plot(Tree.perf, col=2, lwd=3, main="Decision Tree ROC curve")
legend(.5,.4, "AUC = 0.8627769")
abline(0,1)


########## Logistic Regression ##############################################

nfold = 10
set.seed(1)
folds = seq.int(nrow(adult.train)) %>% ## sequential obs ids
  cut(breaks = nfold, labels=FALSE) %>% ## sequential fold ids
  sample ## random fold ids

do.chunk.glm <- function(chunkid, folddef, Xdat, threshold){
  train = (folddef!=chunkid)
  
  Xtr <- Xdat[train,]
  Xvl <- Xdat[!train,]
  
  glm.Xtr <- glm(Income ~., data=Xtr, family = binomial())
  
  probXtr <- predict(glm.Xtr, newdata=Xtr, type= "response")
  probXvl <- predict(glm.Xtr, newdata=Xvl, type= "response")
  
  train.error <- val.error <- rep(0, length(threshold))
  
  for (i in 1:length(threshold)){
    predY_Xtr <- ifelse(probXtr>threshold[i], 1, 0)
    predY_Xvl <- ifelse(probXvl>threshold[i], 1, 0)
    
    train.error[i] <- erate(Xtr$Income, predY_Xtr)
    val.error[i] <- erate(Xvl$Income, predY_Xvl)
  }
  
  data.frame(train.error = train.error, 
             val.error = val.error,
             threshold = threshold)
}

set.seed(1)
thresh <- seq(0.01, 0.99, by=0.01)

errors=NULL
errors = ldply(1:nfold, do.chunk.glm, 
               folddef=folds, 
               Xdat= adult.train, 
               threshold=thresh) %>% 
  group_by(threshold) %>%
  summarise_all(funs(mean)) 

errors

best.thr <- errors$threshold[which.min(errors$val.error)]
best.thr

#Generate model on training data
glm.train <- glm(Income ~., data=adult.train, family = binomial())

#Predict probabilites for training and test data
LogRegprob.train <- predict(glm.train, newdata=adult.train, type="response")
LogRegprob.test <- predict(glm.train, newdata=adult.test, type="response")

#Predict Income values for training and test data
LogRegpred.train <- ifelse(LogRegprob.train>best.thr, 1, 0)
LogRegpred.test <- ifelse(LogRegprob.test>best.thr, 1, 0)

#Calculate train and test error test data
LogReg.errors <- data.frame(train.error = erate(LogRegpred.train, Ytrain),
                            test.error = erate(LogRegpred.test, Ytest))
LogReg.errors


#ROC curve
LogReg.prob <- predict(glm.train, adult.test, type="response")
LogReg.pred <- prediction(LogReg.prob, adult.test$Income)
LogReg.perf <- performance(LogReg.pred, measure="tpr", x.measure="fpr")

#AUC
auc = performance(LogReg.pred, "auc")@y.values
auc

plot(LogReg.perf, col=2, lwd=3, main="Logistic Regresion ROC curve")
legend(.5,.4, "AUC = 0.9093078")

abline(0,1)

########## Random Forests ####################################################

train.error <- test.error <- rep(0, length(Xtrain))

for(i in 1:length(Xtrain)){
  bag.train <- randomForest(Income~., data = adult.train, mtry = i, ntree=2000, importance = TRUE)
  
  #Predict on training and test set
  Forest.pred.train <- predict(bag.train, type="class")
  Forest.pred.test <- predict(bag.train, adult.test, type="class")
  
  #Calculate train and test error
  train.error[i] <- erate(Forest.pred.train, Ytrain)
  test.error[i] <- erate(Forest.pred.test, Ytest)
}

Forest.errors <- data.frame(train.error = train.error,
                            test.error = test.error, 
                            mtry = 1:length(Xtrain))
Forest.errors

best.num.predictors <- Forest.errors[which.min(Forest.errors$test.error),]
best.num.predictors

best.bag.train <- randomForest(Income~., data = adult.train, mtry = best.num.predictors, ntree=2000, importance = TRUE)


importance(best.bag.train)
varImpPlot(best.bag.train)

#ROC curve
Forest.prob <- predict(best.bag.train, adult.test, type="prob")
Forest.prob2 <- data.frame(Forest.prob[,2])
Forest.pred <- prediction(Forest.prob2, adult.test$Income)
Forest.perf <- performance(Forest.pred, measure="tpr", x.measure="fpr")

#AUC
auc = performance(Forest.pred, "auc")@y.values
auc

plot(Forest.perf, col=2, lwd=3, main="Random Forest ROC curve")
legend(.59,.4, "AUC = 0.9319815")

abline(0,1)



