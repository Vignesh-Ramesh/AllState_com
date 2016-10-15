install.packages("h2o")
library(h2o)

h2o.init(nthreads = 3)

# Read input data
train <- h2o.importFile("C:/Users/vrames1/Desktop/Kaggle/train/train.csv", destination_frame = "train.hex")
test <- h2o.importFile("C:/Users/vrames1/Desktop/Kaggle/test/test.csv", destination_frame = "test.hex")
train <- train[, -1]

#Removing these attributes because of correlation with other cont attributes

train<- train[,!names(train) %in% c("cont9","cont10","cont6","cont12")]
test<- test[,!names(test) %in% c("cont9","cont10","cont6","cont12")]

#Loss is a skewed distribution, taking log transform of loss for the model

train$loss <- log1p(train$loss)
splits <- h2o.splitFrame(
  data = train, 
  ratios = c(0.8),
  destination_frames = c("train.hex", "valid.hex"), seed = 1111
)
train <- splits[[1]]
valid <- splits[[2]]

submission <- test[, 1]
test <- test[, -1]

features <- colnames(train)[-131]
label <- "loss"

#attribute selection (grid search later)
ntrees <- c(100,200,300,400)
learn_rate <- c(1,0.5,0.1)

col_sample_rate<-c(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)

#The best ntrees was 100, max depth was 5 and learn rate was 0.1

for (c in col_sample_rate){
  for (t in col_sample_rate){
    if(t == 0.2 && c > 0.2)next  #Skip when n is 200,300,400
    gbm_model <- h2o.gbm(
      features, label, training_frame = train, validation_frame = valid,
      ntrees = 100,max_depth = 5,learn_rate = 0.1,col_sample_rate = c,col_sample_rate_per_tree = t
    )
    print(c(n,l,h2o.mse(h2o.performance(gbm_model, valid = TRUE))))
  }
}


#Try Deep learning  later
#dl_model <- h2o.deeplearning(features, label, training_frame = train, validation_frame = valid, model_id="dl_model_first", 
                             #activation="Rectifier",  ## default
#                             hidden=c(200,200),       ## default: 2 hidden layers with 200 neurons each
#                             epochs=50,
#                             variable_importances=T    ## not enabled by default
#)


#print (h2o.varimp(dl_model))

#View(h2o.varimp(dl_model))

#print(h2o.mse(h2o.performance(dl_model, valid = TRUE)))

submission$loss <- predict(gbm_model, newdata = test)

submission$loss <- expm1(submission$loss)

df<-as.data.frame(submission)

head(df)
#write.csv(df,'sub_7.csv',row.names = F)
