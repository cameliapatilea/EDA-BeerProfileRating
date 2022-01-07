library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)
library(ggplot2)

get_top_styles <- function() {
  df <- read.csv(paste0(path, "eda_dataset/beer_profile_and_ratings.csv"))
  df <- select(df, -Description)
  
  style_counter <- df %>% count(Style)
  
  #style_counter <- head(arrange(style_counter, asc(n)), n = 10)
  style_counter <- head(style_counter[order(style_counter$n),], n = 10)
  
  style_counter
}