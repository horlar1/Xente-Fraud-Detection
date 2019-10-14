# Libaries
library(dplyr)
library(lubridate)
library(stringr)
library(data.table)
# Reading the data
### change this to yours
df = fread(paste0(data.dir,'/training.csv')) %>% as.data.frame()
test = fread(paste0(data.dir,'/test.csv')) %>% as.data.frame()

# Assigning Ids 
train.id = df$TransactionId
test.id = test$TransactionId
label = df$FraudResult

df = df %>%
  within(rm('FraudResult'))

df =rbind(df,test)

####### Data Cleaning and Feature Engineering ########
most_freq_hours <- c('17','16','9','13','18')
least_freq_hours <- c('0','1','23','2','3','22')
df = df%>%
  mutate(
TransactionId = as.numeric(str_extract(TransactionId,"[[:digit:]]+")),
BatchId = as.numeric(str_extract(BatchId,"[[:digit:]]+")),
AccountId = as.numeric(str_extract(AccountId,"[[:digit:]]+")),
SubscriptionId = as.numeric(str_extract(SubscriptionId,"[[:digit:]]+")),
CustomerId = as.numeric(str_extract(CustomerId,"[[:digit:]]+")),
ProviderId = as.numeric(str_extract(ProviderId,"[[:digit:]]+")),
ProductId = as.numeric(str_extract(ProductId,"[[:digit:]]+")),
ChannelId = as.numeric(str_extract(ChannelId,"[[:digit:]]+")),
TransactionStartTime = ymd_hms(df$TransactionStartTime),
CurrencyCode = NULL,
CountryCode = NULL,
ProductCategory = as.numeric(as.factor(ProductCategory)),
Amount_type = as.numeric(as.factor(ifelse(Amount>0 ,"Debit","Credit"))),
Amount2 = ifelse(df$Amount<0,df$Amount*-1,df$Amount),
Value_Amount_diff = Value-Amount2,
Amount2 = NULL,
hour = hour(TransactionStartTime),
week_day = wday(TransactionStartTime),
hour_test_bin = ifelse(hour %in% most_freq_hours,1,
                       ifelse(hour %in% least_freq_hours,2,3)),
date = NULL,
time = NULL,
TransactionId= NULL,
BatchId = NULL,
SubscriptionId = NULL,
CustomerId = NULL,
TransactionStartTime = NULL) %>%
 add_count(Value)%>%
  rename("Value_cnt" = n)%>%
  add_count(ProviderId,hour,ProductId)%>%
  rename("Prov_id_hr_prodid_cnt" = n)%>%
  add_count(ProviderId,hour,ChannelId)%>%
  rename("Prov_id_hr_cha_cnt" = n)%>%
  add_count(ProviderId, ChannelId,hour_test_bin)%>%
  rename("provid_chaid_hr_test_bin_cnt" = n)%>%
  add_count(ProductCategory, ProductId)%>%
  rename("prodcat_provid_cnt" = n)%>%
  add_count(ProductCategory, ProductId,hour_test_bin)%>%
  rename("prodcat_prodid_hr_test_bin_cnt" = n)%>%
  add_count(ProviderId, week_day,hour_test_bin)%>%
  rename("provid_wday_hr_test_bin_cnt" = n)%>%
  add_count(ProductId, week_day,hour_test_bin)%>%
  rename("prodid_wday_hr_test_bin_cnt" = n)%>%
  dplyr::select(-hour_test_bin)


## split the data
df_train = df[1:length(train.id),]
df_test = df[(length(train.id)+1):nrow(df),]


library(xgboost)

dtrain = xgb.DMatrix(as.matrix(df_train), label=label)
dtest = xgb.DMatrix(as.matrix(df_test[,colnames(df_train)]))

watchlist = list(train = dtrain)
#xgboost parameters
xgb_params <- list(colsample_bytree = 0.5, 
                   subsample = 0.5, 
                   booster = "gbtree",
                   max_depth = 3, 
                   min_child_weight = 0,
                   learning_rate = 0.03,
                 #  gamma = 0,
                   nthread = 8,
                   eval_metric = "auc", 
                   watchlist = watchlist,
                   objective = "binary:logistic")
#cross validation
set.seed(1235)
xgb_cv <- xgb.cv(xgb_params,
                 dtrain,
                 early_stopping_rounds = 100 ,
                 nfold = 5,
                 nrounds=5000,
                 print_every_n = 50)

# Training Xgb
set.seed(1235)
xgb_mod <- xgb.train(xgb_params,dtrain,nrounds = 222)

pred= predict(xgb_mod, dtest)
pred = ifelse(pred>0.51,1,0)

sub2 = data.frame(id=test.id,pred)
colnames(sub2) = c("TransactionId","FraudResult")
write.csv(sub2, file="sub.csv", row.names = F)



