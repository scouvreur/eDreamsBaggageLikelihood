# "It's difficult to make predictions, especially about the future."
# Niels Bohr, 1920

# Clear workspace variables
rm(list = ls())
cat("\014")

# Set working directory
setwd("~/Dropbox/Documents/Projects/DataScience/eDreamsBaggageLikelihood")

# Load libraries
library(pROC)
library(ggplot2)
library(caret)

# Load in data
train <- read.csv("train.csv", header = TRUE, sep = ";",
                  na.strings=c(""," ","NA"), stringsAsFactors = TRUE)
test <- read.csv("test.csv", header = TRUE, sep = ";",
                 na.strings=c(""," ","NA"), stringsAsFactors = TRUE)

# Fill in any missing values
train$DEVICE[is.na(train$DEVICE)] <- "OTHER"
test$DEVICE[is.na(test$DEVICE)] <- "OTHER"

# Check for missing data
length(train[!complete.cases(train),])
length(test[!complete.cases(test),])

# Reformatting data structure types
str(train)
str(test)
train$DISTANCE <- as.numeric(train$DISTANCE)
test$DISTANCE <- as.numeric(test$DISTANCE)

train$ARRIVAL <- as.Date(train$ARRIVAL, "%d/%b")
train$DEPARTURE <- as.Date(train$DEPARTURE, "%d/%b")
train$TRIP_LEN_DAYS <- as.integer(abs(difftime(train$ARRIVAL,
                                               train$DEPARTURE,
                                               units = "days")))

test$ARRIVAL <- as.Date(test$ARRIVAL, "%d/%b")
test$DEPARTURE <- as.Date(test$DEPARTURE, "%d/%b")
test$TRIP_LEN_DAYS <- as.integer(abs(difftime(test$ARRIVAL,
                                              test$DEPARTURE,
                                              units = "days")))

# Create a utility function to help with website extraction using UNIX grep
extractWebsite <- function(name) {
  name <- as.character(name)
  if (length(grep("ED", name)) > 0) {
    return("EDREAMS")
  } else if (length(grep("OP", name)) > 0) {
    return("OPODO")
  } else if (length(grep("GO", name)) > 0) {
    return("GO_VOYAGE")
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

# Adding labels to dataset
train$EXTRA_BAGGAGE <- factor(train$EXTRA_BAGGAGE,
                              levels = c("False","True"),
                              labels = c("NO_EXTRA_BAGGAGE", "EXTRA_BAGGAGE"))

train$IS_ALONE <- factor(train$IS_ALONE,
                         levels = c(0,1),
                         labels = c("NOT_ALONE", "ALONE"))

test$IS_ALONE <- factor(test$IS_ALONE,
                        levels = c(0,1),
                        labels = c("NOT_ALONE", "ALONE"))

# Variables not of interest removed
# Assuming there is no local variability between countries (UK, Italy, Spain etc.)
# big assumption though...
train <- subset(train, select = -c(TIMESTAMP, DEPARTURE:ARRIVAL,
                                   TRAIN, PRODUCT, GDS,
                                   NO_GDS, WEBSITE, SMS))
test <- subset(test, select = -c(TIMESTAMP, DEPARTURE:ARRIVAL,
                                 TRAIN, PRODUCT, GDS,
                                 NO_GDS, WEBSITE, SMS))

# Downsampling to balance both classes
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$EXTRA_BAGGAGE)
down_train <- subset(down_train, select = -c(Class))
# There is now no class imbalance in the subsample
table(down_train$EXTRA_BAGGAGE)

# Export data for Python XGBoost Machine Learning model
write.csv(down_train, file="train_xgboost.csv", row.names = FALSE, quote = FALSE)
write.csv(test, file="test_xgboost.csv", row.names = FALSE, quote = FALSE)

# 80/20 train/validation set split
validation <- train[40001:50000,]
train <- train[0:40000,]

model <- glm(EXTRA_BAGGAGE ~ factor(HAUL_TYPE) + factor(TRIP_TYPE) +
             DISTANCE + factor(DEVICE) + factor(COMPANY) +
             FAMILY_SIZE + factor(IS_ALONE) + TRIP_LEN_DAYS,
             data = down_train,
             family = binomial(link = "logit"))
summary(model)
exp(cbind(odds=coef(model), confint(model)))

validation$PREDICTION <- predict(model, validation, type="response")

ggplot(validation, aes(x = PREDICTION, fill = EXTRA_BAGGAGE))
       + geom_density(alpha = 0.5)

rocobj <- roc(factor(validation$EXTRA_BAGGAGE), validation$PREDICTION, ci=TRUE)

plot(roc(factor(validation$EXTRA_BAGGAGE),
     validation$PREDICTION,
     ci=TRUE, direction="<"),
     col="black",
     print.auc=TRUE,
     xlab="False Positive Rate",
     ylab="True Positive Rate",
     main="ROC Curve")

test$EXTRA_BAGGAGE <- predict(model, test, type="response")
write.csv(test[,c("ID","EXTRA_BAGGAGE")], file="submission.csv",
          row.names = FALSE, quote = FALSE)
