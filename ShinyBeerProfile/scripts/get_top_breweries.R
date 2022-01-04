library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)
library(ggplot2)

get_top_breweries <- function() {
  df <- read.csv(paste0(path, "eda_dataset/beer_profile_and_ratings.csv"))
  df <- select(df, -Description)
  
  brewery_counter <- df %>% count(Brewery)
  
  brewery_counter <- head(arrange(brewery_counter, desc(n)), n = 10)
  
  brewery_counter
}