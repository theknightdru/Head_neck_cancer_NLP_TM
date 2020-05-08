##Does it appear that these automated machine learning methods understand well human language?


# yes AI  is very close to simulating human language ,todays technology on deep learning speech synthesis
#can very much replicate human language, some front runners like IBM's watson google analytics 
#where users can use natural language sentences to query their data instead of writing complicated query sentences well that can save time 
#An interesting use of NLP is Gmail’s Smart Reply feature. Google examines the content of an email and presents suggestions for answers.
#But just because a smart speaker or an AI assistant can respond to different ways of asking the weather, it doesn’t mean it is fully 
#understanding the human language,Current NLP is really only good at understanding sentences that have very clear meanings
# AI assistants are becoming better at carrying out basic commands, but if you think you can engage in meaningful conversations 
#and discuss abstract topics with them,they are a disaster;(, but all the progess thats made is very impressing
#the growth is tremendous and we have a long way to go before we have perfect cinematic Ai bots ;).








#load the dataset

hnmed <-
  read.csv("~/Downloads/CaseStudy14_HeadNeck_Cancer_Medication.csv")
View(hnmed)

#create a Vcorpus

hn_med_corpus <- Corpus(VectorSource(hnmed$MEDICATION_SUMMARY))
print(hn_med_corpus)


#inspecting the data
inspect(hn_med_corpus[1:3])

#cleaning the Vcorpus data
corpus_clean <- tm_map(hn_med_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
corpus_clean <- tm_map(corpus_clean, removeNumbers)

#re-checking the cleaned data
inspect(corpus_clean[1:3])


#Building a document term matrix (DTM)
hn_med_dtm <- DocumentTermMatrix(corpus_clean)

#splitting the data into train and test in 90-10
set.seed(12)
subset_int <-
  sample(nrow(hnmed), floor(nrow(hnmed) * 0.90))  # 80% training + 20% testing
hn_med_train <- hnmed[subset_int,]
hn_med_test <- hnmed[-subset_int,]
hn_med_dtm_train <- hn_med_dtm[subset_int,]
hn_med_dtm_test <- hn_med_dtm[-subset_int,]
corpus_train <- corpus_clean[subset_int]
corpus_test <- corpus_clean[-subset_int]


#get seer stage train and teat
prop.table(table(hn_med_train$seer_stage))
prop.table(table(hn_med_test$seer_stage))

#We can separate (dichotomize) the seer_stage into two categories:

#no stage or early stage cancer, and
#later stage cancer.

hn_med_train$stage <- hn_med_train$seer_stage %in% c(4, 5, 7)
hn_med_train$stage <-
  factor(
    hn_med_train$stage,
    levels = c(F, T),
    labels = c("early_stage", "later_stage")
  )
hn_med_test$stage <- hn_med_test$seer_stage %in% c(4, 5, 7)
hn_med_test$stage <-
  factor(
    hn_med_test$stage,
    levels = c(F, T),
    labels = c("early_stage", "later_stage")
  )
prop.table(table(hn_med_train$stage))

#creating the wordclouds for early later and total

wordcloud(corpus_train, min.freq = 40, random.order = FALSE)
#create a wordcloud for early with freq 20 cutoff

early <- subset(hn_med_train, stage == "early_stage")
later <- subset(hn_med_train, stage == "later_stage")
#early
wordcloud(early$MEDICATION_SUMMARY, max.words = 20)
#later

wordcloud(later$MEDICATION_SUMMARY, max.words = 20)

#interpretation of the three wordclouds
#all the three wordclouds have words which occur in all of them such as 'hours','dose', the early word
# wordcloud has words such as tablet,oral,oral,day indicating a starting stage of a medication
#the later wordcloud has words such as pain,for houns,units,blood indicating a later stage of treatment
# the overall wordcloud has words such as every,oral,pain,tablet,severe,oral. indication of a mix of
#both stages of treatment.
 
#Compute the TF-IDF(Term Frequency - Inverse Document Frequency).

dtm.tfidf<-DocumentTermMatrix(corpus_clean, control = list(weighting=weightTfIdf))
dtm.tfidf
dtm.tfidf$dimnames$Docs <- as.character(1:200)
inspect(dtm.tfidf[1:9, 1:10]) 
inspect(hn_med_dtm[1:9, 1:10]) 

set.seed(2)
fit1 <- cv.glmnet(x = as.matrix(dtm.tfidf), y = hnmed[['MEDICATION_SUMMARY']], 
                  family = 'binomial', 
                  # lasso penalty
                  alpha = 1, 
                  # interested in the area under ROC curve
                  type.measure = "auc", 
                  # 10-fold cross-validation
                  nfolds = 10, 
                  # high value is less accurate, but has faster training
                  thresh = 1e-3, 
                  # again lower number of iterations for faster training
                  maxit = 1e3)
plot(fit1)
#Cosine similarity

cos_dist = function(mat){
  numer = tcrossprod(mat)
  denom1 = sqrt(apply(mat, 1, crossprod))
  denom2 = sqrt(apply(mat, 1, crossprod))
  1 - numer / outer(denom1,denom2)
}

dist_cos = cos_dist(as.matrix(hn_med_dtm))

set.seed(2000)
fit_cos <- cv.glmnet(x = dist_cos, y = hnmed[['MEDICATION_SUMMARY']], 
                     family = 'multinomial', 
                     # lasso penalty
                     alpha = 1, 
                     # interested in the area under ROC curve
                     type.measure = "auc", 
                     # 10-fold cross-validation
                     nfolds = 10, 
                     # high value is less accurate, but has faster training
                     thresh = 1e-3, 
                     # again lower number of iterations for faster training
                     maxit = 1e3)
plot(fit_cos)

