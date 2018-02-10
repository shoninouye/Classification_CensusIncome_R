# Classification of 1994 Census Income Data
### Classification of 1994 Census Income Data

## Abstract

In this project, my goal was to find the best model that would predict if an individual makes more than $50,000 a year. I also wanted to find out which predictor variables had the largest impact in determining this. I used Decision Trees, Logistic Regression, and Random Forests in this project. Random Forests ended up being the best model with the highest AUC and lowest misclassification error rate. Variables such as `Education`, `Relationship`, `Marital_status`, and `Occupation` were important predictors when predicting `Income`. In conclusion, this was a successful project that I would like to attempt again with more models and a larger dataset.

## Introduction

The focus of this project is on predicting if an individual makes more than $50,000 a year. The data used in this project was taken from the 1994 Census Database and was provided by the UCI Machine Learning Repository. Income has always been a popular topic and making a lot of money is something that almost everyone has dreamt of at some point in their lives. Growing up, many of my peers and I were told that if we wanted to be successful and get a job that made a lot of money, we would need to go to college. One of the big questions I wanted to answer with this project was how much of an impact does education play in determining an individual's annual income. In addition, I wanted to see what other factors play a role in determining income level and by how much.

This project is done in R and uses the packages `ggplot2`, `plyr`, `dplyr`, `class`, `tree`, `randomForest`, and `ROCR`. I used Decision Trees, Logistic Regression, and Random Forests to perform predictive modeling on the data. The best model was Random Forests, followed by Logistic Regression and then Decision Trees. The models showed that education was indeed a very important predictor in determining whether or not an individual made more than $50,000, as well as Capital Gain, Relationship, Age, and Occupation. Race, Sex, and Working Class were consisitently marked as predictors that had the least amount of impact on income. This report will consist of the step-by-step processes I took to find these conclusions and explanations on the concepts.

## Pre-processing

The following packages will be needed in order to perform our analysis:

    #install.packages("ggplot2")
    #install.packages("plyr")
    #install.packages("dplyr")
    #install.packages("class")
    #install.packages("tree")
    #install.packages("randomForest")
    #install.packages("ROCR")

    #Load packages
    library(ggplot2)
    library(plyr)
    library(dplyr)
    library(class)
    library(tree)
    library(randomForest)
    library(ROCR)

After downloading the dataset from the UCI Machine Learning Repository, read the data into R and check the structure of it.


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
    head(adult)


We can see that the `adult` dataset has 32560 observations and 15 variables. The variable `Income` is what we are predicting, so that leaves us with 14 predictor variables, 5 continuous and 9 categorical.


#### Variable recoding

In order to make working with the data easier in the long run, we will first recode `Income`, to be either 1 or 0 instead of ">50K" or "<=50K".


    #Recode variables
    adult$Income <- ifelse(adult$Income == " <=50K", 0, 1)
    adult$Income <- as.factor(adult$Income)

#### Removing variables

After inspecting the predictors `Education_num` and `Education`, we see that they are the portraying the same information. `Education_num` is just the numeric value of `Education`. We will keep `Education` because of its interpretability and remove `Education_num`.

We will also remove `Native_country` due to the fact that it will most likely not be a very meaningful predictor. Out of the 32560 observations, 90% have the value of "United States".

    ggplot(adult, aes(x=Native_country)) + geom_bar()

    #Remove education_num and Native_country
    adult <- subset(adult, select = -c(Education_num, Native_country))


#### Remove Missing Values

We also want to check how many missing values are in the dataset and then remove observations that have them.

    #Count missing values
    adult[adult == " ?"] <- NA
    sum(is.na(adult))

    #Remove missing values
    adult <- na.omit(adult)
    adult <- data.frame(adult)


#### Training and Test Set with Class-Imabalance

After looking closely at the `Income` variable, we can see that there is an evident class-imbalance problem.

    #Check for class imbalance
    ggplot(adult, aes(x=Income)) +
      geom_bar(stat="count")

Only about 1/4 of the observations have an `Income` value of ">$50K". To solve this problem, we will under-sample the data, taking 4000 observations for our training set with equal amounts of randomly selected values for`Income` and 1000 randomly selected observations from the remainder of the data for the test set.


    ##Create training and test set

    #Separate values of Income
    adult.GT50k <- subset(adult, adult$Income == 1)
    adult.LT50k <- subset(adult, adult$Income == 0)

    #Create inidces for 2000 observations
    set.seed(1)
    adult.GT50k.indices <- sample(1:nrow(adult.GT50k), 2000)
    adult.LT50k.indices <- sample(1:nrow(adult.LT50k), 2000)

    #Take 2000 random observations from both subsets of Income
    adult.GT50k.train <- adult.GT50k[adult.GT50k.indices,]
    adult.LT50k.train <- adult.LT50k[adult.LT50k.indices,]

    #Combine subsets and randomize
    adult.train <- rbind(adult.GT50k.train, adult.LT50k.train)
    adult.train <- adult.train[sample(nrow(adult.train)),]

    #Take row names from training observations
    GT50k.rows <- row.names(adult.GT50k.train)
    LT50k.rows <- row.names(adult.LT50k.train)
    GT50k.rows <- as.numeric(GT50k.rows)
    LT50k.rows <- as.numeric(LT50k.rows)

    #Create subset of adult dataset without training observations
    adult.sub <- adult[-GT50k.rows,]
    adult.sub <- adult.sub[-LT50k.rows,]

    #Take 1000 random observations for test set
    set.seed(1)
    test.indices <- sample(1:nrow(adult.sub), 1000)
    adult.test <- adult.sub[test.indices,]


For convenience purposes, we will create `Xtrain`, `Ytrain`, `Xtest`, and `Ytest` that containing the response and predictor variables for the training and test sets.

    Ytrain <- adult.train$Income
    Xtrain <- adult.train %>% select(-Income)
    Ytest <- adult.test$Income
    Xtest <- adult.test %>% select(-Income)


Now that we have concluded the pre-processing step, we can move on to creating models we will use to predict `Income`.

## Decision Trees

The first model that we will be using is a Decision Tree. Decision Trees can be used for either regression or classification, and in the context of this problem, we will be doing classification. A classification Decision Tree works by setting rules to predict that an observation belongs to the most commonly occurring class label in the dataset the model was trained on.


By using the `tree()` function, we are able to grow a tree on the training set, using `Income` as the response and all other variables as predictors.

    #Fit model on training set
    tree.adult <- tree(Income ~ ., data=adult.train)
    tree.adult <- tree(Income ~ .,
                      data = adult.train,
                      control = tree.control(4000,
                                             mincut = 5,
                                             minsize = 10,
                                             mindev = .003))


Now that the tree has been created, we can plot the tree to see what it looks like.

    #Plot tree
    plot(tree.adult)
    text(tree.adult, pretty = 0, cex = .8, col = "red")
    title("Unpruned Decision Tree of size 23")

Notice how at the split of each node, there is text describing the predictor variable and certain values within the variable. If an observation has these values, then it moves down the left side of the node. If it does not contain these values, it moves down the right side. The title of this tree is "Unpruned Decision Tree of size 23" because it is has 23 terminal nodes (the 1's and 0's at the bottom of the tree) and it is not pruned.

The next step is to prune our tree in order to find a better size and a better error rate. In order to prune the tree, we will perform a 10-fold cross-validation. This will allow us to find the best tree size that will minimize the error rate.

    #Conduct 10-fold cross-validation to prune the tree
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


Now that we have our pruned tree, we can do a little bit of analysis on what we can see. One of the advantages of decision trees is that it is easy to visually interpret. Right away, we see that the first node deals with the relationship predictor and shows that you move down the right side of the tree if you happen to be married. On that side of the tree, there is visibly more `Income` values of 1, indicating that more people make greater than $50K a year. Moving down a couple of nodes, we see `Education` on both sides of the tree. We can also see that if an individual has "some-college" level of education or less, they fall on the side of less `Income` values of 1.

With the pruned tree, we can apply the tree on the training and test sets in order to get our training and test error rates. We will create a function `erate()` that will calculate the misclassification error rate when given the predicted responses and actual responses as inputs. This function will be used to calculate the error rates for the rest of the models as well.


    #Create function to calculate misclassification error rate
    erate <- function(predicted.value, true.value){
      return(mean(true.value!=predicted.value))
    }

    #Predict pruned tree on training and test set
    tree.pred.train.pruned <- predict(tree.adult.pruned, adult.train, type="class")
    tree.pred.test.pruned <- predict(tree.adult.pruned, adult.test, type="class")

    #Calculate train and test error on pruned tree
    tree.errors.pruned <- data.frame(train.error = erate(tree.pred.train.pruned, Ytrain),
                                     test.error = erate(tree.pred.test.pruned, Ytest))
    tree.errors.pruned


Our pruned tree has a training error of 16.675% and a test error of 23%.

Now that we have our pruned decision tree, we can use the `summary()` function to see its inner workings.

    summary(tree.adult.pruned)

From here, we can see the predictor variables that went into the making of this pruned tree. This means that these were the most important predictors that influence `Income`.

Now that we have our ideal Decision Tree, we will use ROC (Receiver Operation Characteristic) curves to show the relationship between false positive (FP) and true positive (TP) rates.

And ideal ROC curve will be as close to the point (0,1) as possible.

    ## Decision tree ROC
    Tree.prob <- predict(tree.adult.pruned, adult.test)
    Tree.prob2 <- data.frame(Tree.prob[,2])
    Tree.pred <- prediction(Tree.prob2, adult.test$Income)
    Tree.perf <- performance(Tree.pred, measure="tpr", x.measure="fpr")

In order to determine the best model, we will be looking at the AUC (Area Under the Curve) of the ROC curve. The higher the AUC, the better the model is at predicting the response varible.


    #AUC
    auc = performance(Tree.pred, "auc")@y.values
    auc


Finally, we plot the ROC curve showing the AUC.


    plot(Tree.perf, col=2, lwd=3, main="Decision Tree ROC curve")
    legend(.5,.2, "AUC = 0.8627769")
    abline(0,1)


## Logistic Regression

The next model that we apply is Logistic Regression. In datasets where the response variable is binary, Logistic Regression works by modeling the probability that response variable $X$ belongs to a particular category instead of trying to model $X$ directly.

By using `glm` (general linear model), we can determine the likelihood of a specific observation having a particular class label. If we assign different thresholds to the probability that a certain class label is created, we can find the different error rates for those thresholds. Performing 10-fold cross-validation will allow us to choose the best threshold that minimizes the misclassification error rate.


    #Create variable to hold the number of cross-validations
    nfold = 10
    set.seed(1)

    #Create variable containing fold assignments for each observation
    folds = seq.int(nrow(adult.train)) %>%
      cut(breaks = nfold, labels=FALSE) %>%
      sample


We will create a function `do.chunk.glm` that will take in the different folds, the training data, and values of thresholds, and output a dataframe of training and test errors for each threshold.


    do.chunk.glm <- function(chunkid, folddef, Xdat, threshold){
      #Create training/validation indices
      train = (folddef!=chunkid)

      #Create training and validation sets
      Xtr <- Xdat[train,]
      Xvl <- Xdat[!train,]

      #Fit model on training set
      glm.Xtr <- glm(Income ~., data=Xtr, family = binomial())

      #Predict probabilites with model on training and validation set
      probXtr <- predict(glm.Xtr, newdata=Xtr, type= "response")
      probXvl <- predict(glm.Xtr, newdata=Xvl, type= "response")

      #Set list of errors (of length of number of thresholds) to 0
      train.error <- val.error <- rep(0, length(threshold))

      #Loop through all thresholds, calculating errors
      for (i in 1:length(threshold)){
        predY_Xtr <- ifelse(probXtr>threshold[i], 1, 0)
        predY_Xvl <- ifelse(probXvl>threshold[i], 1, 0)

        train.error[i] <- erate(Xtr$Income, predY_Xtr)
        val.error[i] <- erate(Xvl$Income, predY_Xvl)
      }

      #Create dataframe containing all training errors, validation errors, and threhsolds
      data.frame(train.error = train.error,
                 val.error = val.error,
                 threshold = threshold)
    }


Now that we have our `do.chunk.glm`, use `ldply` function to run the 10-fold cross-validation and output the best threshold correlated with lowest validation error.


    #Create list of threshold values
    thresh <- seq(0.01, 0.99, by=0.01)

    #Do 10-fold cross validation and store training error, validation error, and thresholds in 'errors'
    errors=NULL
    errors = ldply(1:nfold, do.chunk.glm,
                   folddef=folds,
                   Xdat= adult.train,
                   threshold=thresh) %>%
      group_by(threshold) %>%
      summarise_all(funs(mean))

    #Find best threhshold and output
    best.thr <- errors$threshold[which.min(errors$val.error)]
    best.thr


Now that we have the threshold value that will minimize validation error, we will create a new `glm` model that is trained on the entire training set.

    #Generate model on training data
    glm.train <- glm(Income ~., data=adult.train, family = binomial())

    summary(glm.train)

We can see the model coefficients from the summary of the model above. These are Logistic Regression coefficients, and for every one unit increase in that predictor, the log-odds of getting over $50K increases by that coefficient. So predictor values indicating marriage and a higher education (Relationship Wife, Marital_Status Married-civ-spouse, Education Bachelors, etc.) are more inclined to producing a response value of `Income` greater than 50K.


    #Predict probabilites for training and test data
    LogRegprob.train <- predict(glm.train, newdata=adult.train, type="response")
    LogRegprob.test <- predict(glm.train, newdata=adult.test, type="response")

    #Predict Income values for training and test data
    LogRegpred.train <- ifelse(LogRegprob.train>best.thr, 1, 0)
    LogRegpred.test <- ifelse(LogRegprob.test>best.thr, 1, 0)


Using the predicted response variables and true response variables, we can now calculate the training and test errors for Logistic Regression with the `erate` function.


    #Calculate train and test error test data
    LogReg.errors <- data.frame(train.error = erate(LogRegpred.train, Ytrain),
                                test.error = erate(LogRegpred.test, Ytest))
    LogReg.errors


As we can see, the Logistic Regression model has a training error of 17.05% and a test error of 20.6%.

We can see that the test error of Logistic Regression is lower than that of Decision Trees, but let's take a look at the ROC curve and AUC since that is what we are using for model validation.


    #ROC curve
    LogReg.prob <- predict(glm.train, adult.test, type="response")
    LogReg.pred <- prediction(LogReg.prob, adult.test$Income)
    LogReg.perf <- performance(LogReg.pred, measure="tpr", x.measure="fpr")

    #AUC
    auc = performance(LogReg.pred, "auc")@y.values
    auc



    Already, we can see that the AUC is higher than that of Decision Trees, which is a sign that the model is a better predictor as well.

    plot(LogReg.perf, col=2, lwd=3, main="Logistic Regresion ROC curve")
    legend(.5,.4, "AUC = 0.9093078")

    abline(0,1)


It might be hard to see, but the curve for the Logistic Regression ROC is slightly closer to the point (0,1) than the curve for Decision Trees as well.



## Random Forests

The last model that we will be using is Random Forests. In a sense, Random Forest is just an improved version of Decision Trees. Random Forests works by building decision trees on bootstraped training samples (like in the process known as bagging). In these trees, each time a split at a node is considered, a random sample of $X$ predictors is chosen to be split out of the full number of predictors. Using a smaller number predictors and a large number of trees is optimal when using Random Forests.


In order to find the optimal number of predictors, we will run a loop comparing different number of selected predictors and determine which gives the lowest misclassification error rate.


    #Set lists of errors to value 0 and list length eqaul to number of predictor values
    train.error <- test.error <- rep(0, length(Xtrain))

    #Run random forest model on different number of predictors and calculate training/test errors
    for(i in 1:length(Xtrain)){
      #Fit random forest model with 2000 trees and i predictors
      bag.train <- randomForest(Income~., data = adult.train, mtry = i, ntree=2000, importance = TRUE)

      #Predict on training and test set
      Forest.pred.train <- predict(bag.train, type="class")
      Forest.pred.test <- predict(bag.train, adult.test, type="class")

      #Calculate train and test error
      train.error[i] <- erate(Forest.pred.train, Ytrain)
      test.error[i] <- erate(Forest.pred.test, Ytest)
    }


Now we can create a dataframe containing all training and test errors for each set number of predictors used in the model.

    Forest.errors <- data.frame(train.error = train.error,
                                test.error = test.error,
                                mtry = 1:length(Xtrain))
    Forest.errors


    #Choose number of predictors that has the lowest test error
    best.num.predictors <- Forest.errors$mtry[which.min(Forest.errors$test.error)]

    #Show training error, test error, and number of predictors
    Forest.errors[best.num.predictors,]


After looking at the dataframe, we can see that the Random Forest model with the lowest test error used only 4 predictors. This makes sense because it is the default number of predictors used by Random Forests for classification and generally works pretty well.

Now that we have the best number of predictors, we will create the Random Forest model again, but only using 4 predictors and 2000 trees. We will also create a plot that shows the imporance of each variable.


    #Fit model with best number of predictors
    best.bag.train <- randomForest(Income~.,
                                   data = adult.train,
                                   mtry = best.num.predictors,
                                   ntree=2000,
                                   importance = TRUE)

    #Plot variable importance
    varImpPlot(best.bag.train)


MeanDecreaseAccuracy shows how worse the model does when specific predictors are taken out. The higher the value, the more the accuracy of the model predictions decreases. We will focus on this to rank the importance of our predictor variables. From this graph, we can see that `Capital_gain`, `Education`,`Age`, `Occupation`, and `Hours_per_week` made the most difference in determining `Income`.

MeanDecreaseGini essentially shows the purity of the nodes at the end of the tree. Gini impurity is a measure of how often a randomly chosen element in a set would be incorrectly labeled if labeled. In this case, the higher the MeanDecreaseGini, the less pure the nodes get and more important the predictors are. Some notable variables such as `Relationship`, `Occupation`, and `Marital_status` should also be taken into consideration due to their high MeanDecreaseGini value and prevelence in other models.


Finally, let's plot the ROC and compute the AUC to compare to the other two models.


    #ROC curve
    Forest.prob <- predict(best.bag.train, adult.test, type="prob")
    Forest.prob2 <- data.frame(Forest.prob[,2])
    Forest.pred <- prediction(Forest.prob2, adult.test$Income)
    Forest.perf <- performance(Forest.pred, measure="tpr", x.measure="fpr")

    #AUC
    auc = performance(Forest.pred, "auc")@y.values
    auc

    #Plot ROC with AUC
    plot(Forest.perf, col=2, lwd=3, main="Random Forest ROC curve")
    legend(.5,.4, "AUC = 0.9271246")
    abline(0,1)


## Conclusion

At the beginning of this project, the goal was to find a model that accurately predicts if an individual makes more than $50K a year and determine which factors had the largest impact on that response variable (while paying particular attention to Education). With an AUC higher and test misclassification error rate lower than the other two models, Random Forests wins as the best predictive model of the three. When looking at the variables, it was clear that some variables were prominent in having an effect on `Income`. `Education`, `Relationship`, `Marital_status`, and `Occupation` were some of those variables. To answer the question that I was most interested in, it was clear that any education level above a High School Diploma had a significant positive affect on determining if an individual made greater than 50K a year.

For further work, I intend on tinkering with the different values associated with Random Forests. I know there is much more to Random Forests than what I have accomplished with it in this project. I would also like to try different pruning methods with Decision Trees and attempt other models as well. Finally, I would like to attempt this whole project again, except with the full dataset. Instead of using 5000, I would want to use all 32560 observations.

## References

    - https://en.wikipedia.org/wiki/Random_forest
    - https://www.rdocumentation.org/packages/randomForest/versions/4.6-12/topics/randomForest
    - http://trevorstephens.com/kaggle-titanic-tutorial/r-part-5-random-forests/
    - https://cran.r-project.org/web/packages/randomForest/randomForest.pdf
    - https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    - https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    - http://www.listendata.com/2014/11/random-forest-with-r.html
    - https://stats.stackexchange.com/questions/164569/interpreting-output-of-importance-of-a-random-forest-object-in-r
    - http://blog.datadive.net/interpreting-random-forests/
    - https://archive.ics.uci.edu/ml/datasets/adult
    - Introduction to Data Mining - Tan, Steinbach and Kumar
    - A Introduction to Statistical Learning with Applications in R - James, Witten, Hastie and Tibshiran
