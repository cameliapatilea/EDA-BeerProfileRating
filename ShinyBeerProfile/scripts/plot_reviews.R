plotReviews <- function(data = c()){
  
  channelPlot <- Highcharts$new()
  
  channelPlot$xAxis(categories = as.character(sort(unique(data$Overall_review_bucket))))
  
  
  df2 <- data.frame(bucket = character(0), aroma_review = numeric(0), taste_review = numeric(0), app_review = numeric(0), palate_review = numeric(0))
  
  a <- data %>% count(Aroma_review_bucket) %>% group_by(Aroma_review_bucket) %>% summarise(bucket = Aroma_review_bucket, count_aroma = n)
  a<- select(a, -"Aroma_review_bucket")
  
  b <-  data %>% count(Appearance_review_bucket) %>% group_by(Appearance_review_bucket) %>% summarise(bucket = Appearance_review_bucket, count_app = n)
  b <- select(b, -"Appearance_review_bucket")
  
  c <- data %>%  count(Taste_review_bucket) %>% group_by(Taste_review_bucket) %>% summarise(bucket = Taste_review_bucket, count_taste = n)
  c <- select(c, -"Taste_review_bucket")
  
  
  d <- data %>%  count(Palate_review_bucket) %>% group_by(Palate_review_bucket) %>% summarise(bucket = Palate_review_bucket, count_palate = n)
  d <- select(d, -"Palate_review_bucket")
  
  final <- data %>% count(Overall_review_bucket) %>% group_by(Overall_review_bucket) %>% summarise(bucket = Overall_review_bucket, count_overall = n) 
  final <- select(final, -"Overall_review_bucket")
  
  
  df2 <- merge(a,b, by="bucket")
  inter <- merge(c,d, by="bucket")
  df2 <- merge(df2, inter, by = "bucket")
  df2 <- merge(df2, final, by="bucket")
  
  df2$count_app <- round((df2$count_app / sum(df2$count_app))*100, 2)
  df2$count_taste <- round((df2$count_taste / sum(df2$count_taste))*100,2)
  df2$count_palate <- round((df2$count_palate / sum(df2$count_palate))*100,2)
  df2$count_aroma <- round((df2$count_aroma / sum(df2$count_aroma))*100,2)
  df2$count_overall <- round((df2$count_overall / sum(df2$count_overall))*100,2)
  
  
  channelPlot$series(name = 'Buckets', type = 'column', data = df2$count_overall)
  channelPlot$series(name = 'Aroma', type = 'line', data = df2$count_aroma, yAxis = 1, opposite = TRUE)
  channelPlot$series(name = 'Taste', type = 'line', data = df2$count_taste, yAxis = 1, opposite = TRUE)
  channelPlot$series(name = 'Appearance', type = 'line', data = df2$count_app, yAxis = 1, opposite = TRUE)
  channelPlot$series(name = 'Palate', type = 'line', data = df2$count_palate, yAxis = 1, opposite = TRUE)
  
  channelPlot$plotOptions(line = list(dataLabels = list(enabled = T, x = 4, y = 0)))
  
  channelPlot$title(text = paste0("Beer evaluation"))
  
  channelPlot$yAxis(list(list(title = list(text = "Percentage of reviews")
                              , alignTicks = TRUE
                              , max = 30)
                              , list(title = list(text = "Reviews")
                                , gridLineWidth = 0
                                , opposite = FALSE, min = 0, max = 30)))          
  
  channelPlot$chart(zoomType = "xy")
  
  channelPlot$set(width = 1500, height = 600)
  
  channelPlot$tooltip(shared = TRUE)
  
  list(channelPlot = channelPlot)
}