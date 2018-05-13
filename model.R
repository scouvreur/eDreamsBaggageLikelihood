# "It's difficult to make predictions, especially about the future."
# Niels Bohr, 1920

# Clear workspace variables
rm(list = ls())
cat("\014")

# Set working directory
setwd("~/Dropbox/Documents/Projects/DataScience/eDreamsBaggageLikelihood")

# Load libraries
library(lmtest, pROC, MLmetrics)

# Load in data
train <- read.csv("train.csv", header = TRUE, sep = ";", na.strings=c(""," ","NA"), stringsAsFactors = TRUE)
test <- read.csv("test.csv", header = TRUE, sep = ";", na.strings=c(""," ","NA"), stringsAsFactors = TRUE)

# Check for missing data
# length(train[!complete.cases(train),])

# Add a labels to the categories of variables
train$EXTRA_BAGGAGE <- factor(train$EXTRA_BAGGAGE, levels = c("False","True"), labels = c("No Extra Baggage", "Extra Baggage"))

# Reformatting data structure types
str(train)
str(test)
train$DISTANCE<- as.numeric(train$DISTANCE)
test$DISTANCE <- as.numeric(test$DISTANCE)

## Preliminary data investigation
# Severe class imbalance with these two variables
round(prop.table(table(train$TRAIN))*100, digits = 1)
round(prop.table(table(train$PRODUCT))*100, digits = 1)

# It seems that there is not much interesting with SMS confirmation
round(prop.table(table(train$SMS, train$EXTRA_BAGGAGE), 1)*100, digits = 1)

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
    return("OTHER")
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

# Create family size variable
FAMILY_SIZE = train$ADULTS + train$CHILDREN + train$INFANTS
train$FAMILY_SIZE <- FAMILY_SIZE
FAMILY_SIZE = test$ADULTS + test$CHILDREN + test$INFANTS
test$FAMILY_SIZE <- FAMILY_SIZE
rm(FAMILY_SIZE)

# Create a utility function to extract if adult is travelling alone
extractAlone <- function(familysize) {
  if (familysize > 1) {
    return(0)
  } else {
    return(1)
  }
}

# Create is alone variable
IS_ALONE <- NULL
for (i in 1:nrow(train)) {
  IS_ALONE <- c(IS_ALONE, extractAlone(train[i,"FAMILY_SIZE"]))
}
train$IS_ALONE <- as.factor(IS_ALONE)
train$IS_ALONE <- factor(train$IS_ALONE, levels = c(0,1), labels = c("Not alone", "Alone"))

IS_ALONE <- NULL
for (i in 1:nrow(train)) {
  IS_ALONE <- c(IS_ALONE, extractAlone(test[i,"FAMILY_SIZE"]))
}
test$IS_ALONE <- as.factor(IS_ALONE)
test$IS_ALONE <- factor(test$IS_ALONE, levels = c(0,1), labels = c("Not alone", "Alone"))

rm(IS_ALONE, i, extractAlone)

# Variables not of interest removed
# Assuming there is no local variability between countries (UK, Italy, Spain etc.), big assumption though...
train <- subset(train, select = -c(TIMESTAMP, DEPARTURE:ARRIVAL, TRAIN, PRODUCT, GDS, NO_GDS, WEBSITE, SMS))
test <- subset(test, select = -c(TIMESTAMP, DEPARTURE:ARRIVAL, TRAIN, PRODUCT, GDS, NO_GDS, WEBSITE, SMS))

# There is insteresting subtle variability between different booking websites, not all the same
round(prop.table(table(train$COMPANY, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$HAUL_TYPE, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$TRIP_TYPE, train$EXTRA_BAGGAGE), 1)*100, digits = 1)

# We investigate a potential relation between infants and baggage
round(prop.table(table(train$ADULTS, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$CHILDREN, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$INFANTS, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$FAMILY_SIZE, train$EXTRA_BAGGAGE), 1)*100, digits = 1)
round(prop.table(table(train$IS_ALONE, train$EXTRA_BAGGAGE), 1)*100, digits = 1)

# Export data for Python XGBoost Machine Learning model
# write.csv(train, file="train_.csv", row.names = FALSE)
# write.csv(test, file="test_.csv", row.names = FALSE)

## Building the actual model

# 80/20 train/validation set split
validation <- train[40001:50000,]
train <- train[0:40000,]

model <- glm(EXTRA_BAGGAGE ~ DISTANCE + factor(HAUL_TYPE) + factor(TRIP_TYPE) +
              factor(DEVICE) + factor(COMPANY) + FAMILY_SIZE +
              factor(IS_ALONE),
              data = train,
              family = binomial(link = "logit"))
summary(model)
exp(cbind(odds=coef(model), confint(model)))

prediction <- predict(model, validation, type="response")

roc_obj <- roc(factor(validation$EXTRA_BAGGAGE), prediction)
auc(roc_obj)
F1_Score(y_pred = prediction, y_true = validation$EXTRA_BAGGAGE)

plot(roc(validation$EXTRA_BAGGAGE, prediction, direction="<"),
     col="black",
     xlab="False Positive Rate",
     ylab="True Positive Rate",
     main="ROC Curve")

submission = data.frame(predict(model, test, type="response"))
write.csv(submission, file="submission_logreg.csv", row.names = TRUE)
