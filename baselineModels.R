# Load data
setwd("~/../ubuntu/flight-delays/newData")
library(data.table)
dat <- fread("ord-prp-2009.csv")
dat$ARR_DELAY <- as.factor(dat$ARR_DELAY)

# Clean up variable names
set.seed(2019)
newColNames <- colnames(dat)
for(i in 1:length(newColNames)) {
  if(substring(newColNames[i], 1, 1)=="+") {
    newColNames[i] <- paste0("plus", substring(newColNames[i], 2))
  } else if(substring(newColNames[i], 1, 1)=="-") {
    newColNames[i] <- paste0("minus", substring(newColNames[i], 2))
  }
}
colnames(dat) <- newColNames
dev <- sample(nrow(limited), 26209)
# Drop redundant/unneeded columns
devset <- dat[dev,-c("DEP_DELAY", "V1", "a_join_time",
                         "Number", "wxcodes", "id", 
                         "ORIGIN_AIRPORT_ID", "OP_CARRIER_FL_NUM",
                         "join_time", "year")]
training <- dat[-dev,-c("DEP_DELAY", "V1", "a_join_time",
                            "Number", "wxcodes", "id", 
                            "ORIGIN_AIRPORT_ID", "OP_CARRIER_FL_NUM",
                            "join_time", "year")]

# Gradient boosting machine
library(gbm)
gbm.model <- gbm((as.numeric(training$ARR_DELAY)-1) ~ .,
                 data=training, distribution="bernoulli",
                 n.trees=1000, shrinkage=0.1, verbose=TRUE)
summary(gbm.model)
ypred <- as.numeric(predict(gbm.model, devset, n.trees=1000)>0.5)
tab <- table(predict=ypred, truth=devset$ARR_DELAY); tab
accuracy <- (tab[1,1] + tab[2,2]) / nrow(devset); accuracy
recall <- tab[2,2] / (tab[2,2]+tab[1,2]); recall
precision <- tab[2,2] / (tab[2,2]+tab[2,1]); precision
f1 <- 2*recall*precision / (recall + precision); f1
gbm.tab <- tab

# Logistic regression
log.model <- glm(ARR_DELAY ~ .,
                 data=training, family=binomial, maxit=100)
ypred <- as.numeric(predict(log.model, devset, type="response")>0.5)
tab <- table(predict=ypred, truth=devset$ARR_DELAY); tab
accuracy <- (tab[1,1] + tab[2,2]) / nrow(devset); accuracy
recall <- tab[2,2] / (tab[2,2]+tab[1,2]); recall
precision <- tab[2,2] / (tab[2,2]+tab[2,1]); precision
f1 <- 2*recall*precision / (recall + precision); f1
log.tab <- tab

# Naive Bayes
library(e1071)
nb.model <- naiveBayes(ARR_DELAY ~ .,
                       data=training, laplace=0)
ypred <- predict(nb.model, devset, type="class")
tab <- table(predict=ypred, truth=devset$ARR_DELAY); tab
accuracy <- (tab[1,1] + tab[2,2]) / nrow(devset); accuracy
recall <- tab[2,2] / (tab[2,2]+tab[1,2]); recall
precision <- tab[2,2] / (tab[2,2]+tab[2,1]); precision
f1 <- 2*recall*precision / (recall + precision); f1
naiveBayes.tab <- tab