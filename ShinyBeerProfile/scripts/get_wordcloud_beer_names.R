library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)

get_wordcloud_beer_names <- function() {
  df <- read.csv(paste0(path, "eda_dataset/beer_profile_and_ratings.csv"))
  df <- select(df, -Description)
  
  #Create a vector containing only the text
  text <- df$Name
  docs <- Corpus(VectorSource(text))
  
  docs <- docs %>% 
    tm_map(removeNumbers) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
  docs <- tm_map(docs, content_transformer(tolower))
  docs <- tm_map(docs, removeWords, stopwords("english"))
  
  dtm <- TermDocumentMatrix(docs)
  matrix <- as.matrix(dtm)
  words <- sort(rowSums(matrix), decreasing=TRUE)
  df_wordcloud_name <- data.frame(word = names(words), freq=words)
  
  set.seed(1234)
  
  df_wordcloud_name 
}