# "It's difficult to make predictions, especially about the future."
# Niels Bohr, 1920

# Clear workspace variables
rm(list = ls())
cat("\014")

# Set working directory
setwd("~/Dropbox/Documents/Projects/DataScience/eDreamsBaggageLikelihood")

# Load libraries
library(pROC)

# Load in data
train <- read.csv("train.csv", header = TRUE, sep = ";", na.strings=c(""," ","NA"), stringsAsFactors = TRUE)
test <- read.csv("test.csv", header = TRUE, sep = ";", na.strings=c(""," ","NA"), stringsAsFactors = TRUE)

# Fill in any missing values
train$DEVICE[is.na(train$DEVICE)] <- "OTHER"
test$DEVICE[is.na(test$DEVICE)] <- "OTHER"

# Reformatting data structure types
str(train)
str(test)
train$DISTANCE<- as.numeric(train$DISTANCE)
test$DISTANCE <- as.numeric(test$DISTANCE)

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

IS_ALONE <- NULL
for (i in 1:nrow(train)) {
  IS_ALONE <- c(IS_ALONE, extractAlone(test[i,"FAMILY_SIZE"]))
}
test$IS_ALONE <- as.factor(IS_ALONE)

rm(IS_ALONE, i, extractAlone)

# Create distance category
train$DISTANCE_CAT <- factor(cut(train$DISTANCE, c(-1,4000,100000), labels = FALSE))
test$DISTANCE_CAT <- factor(cut(test$DISTANCE, c(-1,4000,100000), labels = FALSE))

# Variables not of interest removed
# Assuming there is no local variability between countries (UK, Italy, Spain etc.), big assumption though...
train <- subset(train, select = -c(TIMESTAMP, DEPARTURE:ARRIVAL, TRAIN, PRODUCT, GDS, NO_GDS, WEBSITE, SMS))
test <- subset(test, select = -c(TIMESTAMP, DEPARTURE:ARRIVAL, TRAIN, PRODUCT, GDS, NO_GDS, WEBSITE, SMS))

# Check for missing data
length(train[!complete.cases(train),])
length(test[!complete.cases(test),])

# Export data for Python XGBoost Machine Learning model
write.csv(train, file="train_xgboost.csv", row.names = FALSE, quote = FALSE)
write.csv(test, file="test_xgboost.csv", row.names = FALSE, quote = FALSE)

# 80/20 train/validation set split
validation <- train[40001:50000,]
train <- train[0:40000,]

model <- glm(EXTRA_BAGGAGE ~ factor(HAUL_TYPE) + factor(TRIP_TYPE) +
             factor(DISTANCE_CAT) + factor(DEVICE) + factor(COMPANY) +
             FAMILY_SIZE + factor(IS_ALONE),
             data = train,
             family = binomial(link = "logit"))
summary(model)
exp(cbind(odds=coef(model), confint(model)))

prediction <- predict(model, validation, type="response")

roc_obj <- roc(factor(validation$EXTRA_BAGGAGE), prediction)
auc(roc_obj)

plot(roc(validation$EXTRA_BAGGAGE, prediction, direction="<"),
     col="black",
     xlab="False Positive Rate",
     ylab="True Positive Rate",
     main="ROC Curve")

test$EXTRA_BAGGAGE <- predict(model, test, type="response")
write.csv(test[,c("ID","EXTRA_BAGGAGE")], file="submission.csv", row.names = FALSE, quote = FALSE)
