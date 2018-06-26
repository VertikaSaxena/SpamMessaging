#install.packages("caTools", "ROCR", "pROC", "lattice", "rpart.plot")

library(ggplot2)
library(readr)
library(caTools)
library(e1071)
library(rpart)
library(wordcloud)
library(tm)
library(SnowballC)
library(ROCR)
library(pROC)
library(RColorBrewer)
library(stringr)

rm(list=ls())
text= read.csv(choose.files())
text<- text[,1:2]

names(text) <- c("label","message")
levels(as.factor(text$label))
text$message<- as.character(text$message)
str(text)

##Find NULL Values and eliminate them##
which(!complete.cases(text))
table(text$label)
prop.table(table(text$label))
text$text_length<- nchar(text$message)

#Histogram text frequency and text length#
hist(text$text_length)

#ggplot#
ggplot(text,aes(text_length,fill=label)) + geom_histogram(binwidth = 6)+
  facet_wrap(~label)      #If you want to see ham and spam types seperately 

# Text Analysis
# Clean text for analysis and create bag of words from text using corpus and tokens

bag <- Corpus(VectorSource(text$message))
print(bag)
inspect(bag[1:3])
bag <- tm_map(bag, tolower)
bag <- tm_map(bag, removeNumbers)
bag <- tm_map(bag, stemDocument)
bag <- tm_map(bag, removePunctuation)
bag <- tm_map(bag, removeWords, c(stopwords("english")))
bag <- tm_map(bag, stripWhitespace)

print(bag)
inspect(bag[1:3])

graphics.off()
wordcloud(bag, max.words=150,scale=c(3,1),colors=brewer.pal(8,"Dark2"))

# Convert bag of words to data frame
frequencies <- DocumentTermMatrix(bag)
frequencies

# look at words that appear atleast 200 times
findFreqTerms(frequencies, lowfreq = 200)
sparseWords <- removeSparseTerms(frequencies, 0.995)
sparseWords

#organizing frequency of terms
freq <- colSums(as.matrix(sparseWords))
length(freq)
ord <- order(freq)
ord                

#find associated terms
findAssocs(sparseWords, c('call','get'), corlimit=0.10)

#create wordcloud
library(wordcloud)
set.seed(142)

# Let's visualize the data now. But first, we'll create a data frame.
#create a data frame for visualization
wf <- data.frame(word = names(freq), freq = freq)
head(wf)

#plot 5000 most used words
wordcloud(names(freq), freq, max.words = 2500, scale = c(5, .1), colors = brewer.pal(5, 'Dark2'))

##Let's create the word cloud for each ham and spam to understand difference
ham_cloud<- which(text$label=="spam")
spam_cloud<- which(text$label=="ham")

wordcloud(bag[ham_cloud],min.freq=100)
wordcloud(bag[spam_cloud],min.freq=100)

wordcloud(bag[ham_cloud],min.freq=50)
wordcloud(bag[spam_cloud],min.freq=50)


#plot words which appear atleast 10,000 times in the text
library(ggplot2)
chart <- ggplot(subset(wf, freq >100), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'white')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart

# convert the matrix of sparse words to data frame
sparseWords <- as.data.frame(as.matrix(sparseWords))

# rename column names to proper format in order to be used by R
colnames(sparseWords) <- make.names(colnames(sparseWords))
str(sparseWords)
sparseWords$label <- text$label
colnames(sparseWords)


# Predicting whether SMS is spam/non-spam by split data into 75:25 and assign to train and test.

set.seed(987)
split <- sample.split(sparseWords$label, SplitRatio = 0.75)
train <- subset(sparseWords, split == T)
test <- subset(sparseWords, split == F)

table(test$label)
print(paste("Predicting all messages as non-spam gives an accuracy of: ",
            100*round(table(test$label)[1]/nrow(test), 4), "%"))     ##shows the number of ham and spam 


text_classifier<- naiveBayes(label~.,train,laplace = 1)
text_test_pred<- predict(text_classifier,newdata = test)
table(test$label, text_test_pred)


#we fit the glm model to predict values
glm.model <- glm(label ~ ., data = train, family = "binomial")
glm.predict <- predict(glm.model, test, type = "response")

### AUC curve
glm.ROCR <- prediction(glm.predict, test$label)
print(glm.AUC <- as.numeric(performance(glm.ROCR,"auc")@y.values))

glm.prediction <- prediction(abs(glm.predict), test$label)
glm.performance <- performance(glm.prediction,"tpr","fpr")
plot(glm.performance)

### selecting threshold = 0.75 for spam filtering
table(test$label, glm.predict > 0.75)

# To show logistic model accuracy
glm.accuracy.table <- as.data.frame(table(test$label, glm.predict > 0.75))
print(paste("logistic model accuracy:",
            100*round(((glm.accuracy.table$Freq[1]+glm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))


# Support Vector Machine Model to show accuracy
svm.model <- svm(label ~ ., data = train, kernel = "linear", cost = 0.1, gamma = 0.1)
svm.predict <- predict(svm.model, test)
table(test$label, svm.predict)


svm.accuracy.table <- as.data.frame(table(test$label, svm.predict))
print(paste("SVM accuracy:",
            100*round(((svm.accuracy.table$Freq[1]+svm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))


