# "Ask not what your country can do for you, but ask what can you do for your country."
# John F. Kennedy, 1968

# "The simulacra is simply the truth that there is none."
# Jean Baudrillard, Simulacra and Simulation, 1988

# Clear workspace variables
rm(list = ls())
cat("\014")

# Set working directory
setwd("~/Dropbox/Documents/Projects/DataScience/eDreamsBaggageLikelihood")

# Load libraries
# library(haven, lmtest)

# Load in data
train <- read.csv("train.csv", header = TRUE, sep = ";", na.strings=c(""," ","NA"))
test <- read.csv("test.csv", header = TRUE, sep = ";", na.strings=c(""," ","NA"))

# Reformatting data structure for data
train$DISTANCE<- as.numeric(train$DISTANCE)
test$DISTANCE <- as.numeric(test$DISTANCE)

# 80/20 train/validation set split
validation <- train[40001:50000,]
train <- train[0:40000,]

str(train)


