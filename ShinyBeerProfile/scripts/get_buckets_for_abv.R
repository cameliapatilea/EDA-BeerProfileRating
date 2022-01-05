get_buckets_for_abv <- function(){
  data <- read.csv(paste0(path, "eda_dataset/beer_profile_and_ratings.csv"))
  data <- select(data, -Description)
  
  
  data$bucket_abv <- cut2(data$ABV, c(0,1,4,6,10,12,60))
  
  data <- data %>% count(bucket_abv) %>% group_by(bucket_abv)
  colnames(data) <- c("bucket", "count")
  
  data
}