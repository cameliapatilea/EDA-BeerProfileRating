library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)
library(ggplot2)

get_beer_statistics <- function() {
  df <- read.csv(paste0(path, "eda_dataset/beer_profile_and_ratings.csv"))
  
  number_of_rows <- nrow(df)
  number_of_columns <- ncol(df)
  
  number_of_numeric_columns <- length(select_if(df, is.numeric))
  number_of_chr_columns <- length(select_if(df, is.character))
  
  
  result = list(number_of_rows = number_of_rows,
            number_of_columns = number_of_columns,
            number_of_numeric_columns = number_of_numeric_columns,
            number_of_chr_columns = number_of_chr_columns)
  
  result
  
}