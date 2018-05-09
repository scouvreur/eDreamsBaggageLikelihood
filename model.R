# "It's difficult to make predictions, especially about the future."
# Niels Bohr, 1920

# Clear workspace variables
rm(list = ls())
cat("\014")

# Set working directory
setwd("~/Dropbox/Documents/Projects/DataScience/eDreamsBaggageLikelihood")

# Load libraries
library(haven, lmtest)

# Load in data
train <- read.csv("train.csv", header = TRUE, sep = ";", na.strings=c(""," ","NA"), stringsAsFactors = TRUE)
test <- read.csv("test.csv", header = TRUE, sep = ";", na.strings=c(""," ","NA"), stringsAsFactors = TRUE)

## Preliminary data investigation
round(prop.table(table(train$TRAIN))*100, digits = 1)
round(prop.table(table(train$PRODUCT))*100, digits = 1)

# Create a utility function to help with website extraction using UNIX grep
extractWebsite <- function(name) {
  name <- as.character(name)
  if (length(grep("ED", name)) > 0) {
    return("EDREAMS")
  } else if (length(grep("OP", name)) > 0) {
    return("OPODO")
  } else if (length(grep("GO", name)) > 0) {
    return("GO VOYAGE")
  } else {
    return("Other")
  }
}

# Run function and clear workspace
COMPANY <- NULL
for (i in 1:nrow(train)) {
  COMPANY <- c(COMPANY, extractWebsite(train[i,"WEBSITE"]))
}
train$COMPANY <- as.factor(COMPANY)

COMPANY <- NULL
for (i in 1:nrow(test)) {
  COMPANY <- c(COMPANY, extractWebsite(test[i,"WEBSITE"]))
}
test$COMPANY <- as.factor(COMPANY)

rm(COMPANY, i, extractWebsite)

# Variables not of interest removed
# Assuming there is no local variability between countries (UK, Italy, Spain etc.), big assumption though...
train <- subset(train, select = -c(TIMESTAMP, DEPARTURE:ARRIVAL, TRAIN, PRODUCT, GDS, NO_GDS, WEBSITE))
test <- subset(test, select = -c(TIMESTAMP, DEPARTURE:ARRIVAL, TRAIN, PRODUCT, GDS, NO_GDS, WEBSITE))

# Reformatting data structure for data
str(train)
str(test)
train$DISTANCE<- as.numeric(train$DISTANCE)
test$DISTANCE <- as.numeric(test$DISTANCE)

# It seems that there is not much interesting between different booking websites, they are all the same
round(prop.table(table(train$COMPANY, train$EXTRA_BAGGAGE), 1)*100, digits = 1)

# We therefore remove that website variable
train <- subset(train, select = -c(COMPANY))
test <- subset(test, select = -c(COMPANY))

# It seems that there is not much interesting with SMS confirmation or booking device either
round(prop.table(table(train$SMS, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$DEVICE, train$EXTRA_BAGGAGE), 1)*100, digits = 1)

# We therefore remove that SMS variable
train <- subset(train, select = -c(SMS, DEVICE))
test <- subset(test, select = -c(SMS, DEVICE))

# 
round(prop.table(table(train$HAUL_TYPE, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$TRIP_TYPE, train$EXTRA_BAGGAGE), 1)*100, digits = 1)

# We investigate a potential relation between infants and baggage
round(prop.table(table(train$ADULTS, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$CHILDREN, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$INFANTS, train$EXTRA_BAGGAGE), 1)*100, digits = 1)

# We could create a synthetic variable when at least on parent travels with one child or infant
extractWebsite <- function(name) {
  name <- as.character(name)
  if (length(grep("ED", name)) > 0) {
    return("EDREAMS")
  } else if (length(grep("OP", name)) > 0) {
    return("OPODO")
  } else if (length(grep("GO", name)) > 0) {
    return("GO VOYAGE")
  } else {
    return("Other")
  }
}

## Building the actual model

# 80/20 train/validation set split
validation <- train[40001:50000,]
train <- train[0:40000,]

model1 <- glm(factor(EXTRA_BAGGAGE) ~ DISTANCE + factor(HAUL_TYPE) + factor(TRIP_TYPE) + factor(DEVICE) + factor(SMS) + factor(COMPANY) + ADULTS + CHILDREN + INFANTS,
              data = train,
              family = binomial(link = "logit"))
summary(model1)
exp(cbind(odds=coef(model1), confint(model1)))

prediction <- predict(model1, validation, type="response")

library(pROC)
roc_obj <- roc(factor(validation$EXTRA_BAGGAGE), prediction)
auc(roc_obj)
