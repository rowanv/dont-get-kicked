library(randomForest)
library(caTools) #for splitting dataset
library(Amelia) #dealing with NAs
library(ggplot2)
library(caret)

#Don't Get Kicked
if(FALSE){
data.dir <- "/Users/rowanvasquez/Documents/Personal\ Research\ Projects/Datasets/" 
ca.test.file <- paste0(data.dir, "dont-get-kicked-test.csv") 
ca.train.file <- paste0(data.dir, "dont-get-kicked-train.csv")

ca.test <- read.csv(ca.test.file)
ca.train <- read.csv(ca.train.file)


save(ca.train, file = "dont-get-kicked-train.Rda")
save(ca.test, file = "dont-get-kicked-test.Rda")
}
load("dont-get-kicked-train.Rda")
load("dont-get-kicked-test.Rda")

summary(ca.train)
str(ca.train)

table(ca.train$IsBadBuy)
table(ca.train$Auction)

#Data Exploration
#Vehicle Age
ggplot(ca.train, aes(x = VehicleAge)) + geom_line(stat = "density") + ggtitle("Vehicle Age Density Curve")
ggplot(ca.train, aes(x = VehicleAge)) + geom_histogram() + ggtitle("Vehicle Age Histogram")
boxplot(ca.train$VehicleAge, main = "Vehicle Age Boxplot")

ggplot(ca.train, aes(x=VehOdo)) + geom_line(stat = "density") + ggtitle("Vehicle Odometer Density Curve")

ggplot(ca.train, aes(x=WarrantyCost)) + geom_line(stat = "density") + ggtitle("Warranty Cost Density Curve")

#Acquisition cost paid for the vehicle at time of purchase
ggplot(ca.train, aes(x=VehBCost)) + geom_line(stat = "density") + ggtitle("Acquisition Cost at Time of Purchase Cost Density Curve")


ggplot(ca.train, aes(x = Auction)) + geom_histogram() + ggtitle("Auction Histogram")

ggplot(ca.train, aes(x = Transmission)) + geom_histogram() + ggtitle("Transmission Histogram")

#recode the auction price variables from factor to numeric
ca.train$MMRAcquisitionAuctionAveragePrice<-as.numeric(ca.train$MMRAcquisitionAuctionAveragePrice)
#Acquisition price for this vehicle in average condition at time of purchase

ca.train$MMRAcquisitionAuctionCleanPrice <- as.numeric(ca.train$MMRAcquisitionAuctionCleanPrice)
#Acquisition price for this vehicle in the above Average condition at time of purchase

ca.train$MMRAcquisitionRetailAveragePrice <- as.numeric(ca.train$MMRAcquisitionRetailAveragePrice)
#Acquisition price for this vehicle in the retail market in average condition at time of purchase
ca.train$MMRAcquisitonRetailCleanPrice <- as.numeric(ca.train$MMRAcquisitonRetailCleanPrice)
#Acquisition price for this vehicle in the retail market in above average condition at time of purchase
ca.train$MMRCurrentAuctionAveragePrice <- as.numeric(ca.train$MMRCurrentAuctionAveragePrice)
#Acquisition price for this vehicle in average condition as of current day
ca.train$MMRCurrentAuctionCleanPrice <- as.numeric(ca.train$MMRCurrentAuctionCleanPrice)
#Acquisition price for this vehicle in the above condition as of current day
ca.train$MMRCurrentRetailAveragePrice <- as.numeric(ca.train$MMRCurrentRetailAveragePrice)
#Acquisition price for this vehicle in the retail market in average condition as of current day
ca.train$MMRCurrentRetailCleanPrice <- as.numeric(ca.train$MMRCurrentRetailCleanPrice)
#Acquisition price for this vehicle in the retail market in above average condition as of current day

ggplot(ca.train, aes(x=MMRAcquisitionAuctionAveragePrice)) + geom_line(stat = "density")
ggplot(ca.train, aes(x=MMRAcquisitionAuctionCleanPrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRAcquisitionRetailAveragePrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRAcquisitonRetailCleanPrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRCurrentAuctionAveragePrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRCurrentAuctionCleanPrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRCurrentRetailAveragePrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRCurrentRetailCleanPrice)) + geom_line(stat = "density") 


#Cleaning the data set

#Redundant variable in Transmission
table(ca.train$Transmission)
ca.train$Transmission[ca.train$Transmission == "Manual"] <- "MANUAL"
table(ca.train$Transmission)

#not use purchase date, vehicle year is redundant, not use zip code

ca.train$PurchDate<-ca.train$VehYear<-ca.train$VNZIP1<-NULL
#getting rid of Make, Model, Trim, subModel, Color, Size, VNST temporarily since differs between training and testing set
ca.train$Make<-ca.train$Model<-ca.train$Trim<-ca.train$SubModel<-ca.train$Color<-ca.train$Size<-ca.train$VNST<-NULL
str(ca.train)


#ca.train <- na.omit(ca.train)
#For our first set of models, let's just get rid of the NAs. We'll deal with them more thoroughly when we refine our models

save(ca.train, file = "dont-get-kicked-train-clean.Rda") #saving the clean data set
#save(ca.test, file = "dont-get-kicked-test.Rda")



#########
#Split training set into sub-training and sub-testing
set.seed(1000)
ca.train.split = sample.split(ca.train$IsBadBuy, SplitRatio = 0.7)
ca.train.tr <- subset(ca.train, ca.train.split == TRUE)
ca.train.te <- subset(ca.train, ca.train.split == FALSE)

ca.train <- ca.train.tr


###########
#Baseline
##########

#Baseline is that no buy is a bad buy
accur.baseline = 64007/(8976 + 64007)
accur.baseline
#accur.baseline = 0.8770125

############
#Logit######
###########

#ca.train.cost<- str(ca.train[, c(2, 17:24,29, 31)])

logit1 <- glm(IsBadBuy~., data = ca.train, family = "binomial")
summary(logit1)

prob.logit1 <- predict(logit1, newdata = ca.train.te, type = "response")
summary(prob.logit1)
head(prob.logit1, n=10)
pred.logit1 <- prob.logit1 >= 0.50

#confusion marix
confus.logit1 <- table(pred.logit1, ca.train.te$IsBadBuy)
confus.logit1
accur.logit1 <- (confus.logit1[1,1]+confus.logit1[2,2])/(confus.logit1[1,1]+confus.logit1[2,2] + confus.logit1[1,2] + confus.logit1[2,1])
accur.logit1

#accuracy of logit1 is 89.42%
#just the price/cost variables
#logit2 <- glm(IsBadBuy~ c(17:24, 29, 31), data = ca.train, family = binomial)
#summary(logit2)


#Let's look at our missing values

#missmap(ca.train, main = "Missingness Map Train")
#missmap(ca.test, main = "Missingness Map Test")

str(ca.train)
