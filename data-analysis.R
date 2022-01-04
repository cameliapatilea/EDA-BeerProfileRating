require(wordcloud)
require(RColorBrewer)
require(wordcloud2)
library(tm)

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
wordcloud(words = df_wordcloud_name$word,
          freq = df_wordcloud_name$freq, 
          min.freq = 1,
          max.words = 200,
          random.order = FALSE,
          rot.per = 0.35,
          colors = brewer.pal(8, "Dark2"))

wordcloud2(data=df_wordcloud_name, size=1.6, color='random-dark')

# de facut in phostoshop
beer_1_path = system.file(paste0(path, "/media/beer-1.png"), package = "wordcloud2")
wordcloud2(df_wordcloud_name, figPath = paste0(path, "media/beer-1.png"), size = 1,color = "skyblue")