get_buckets_for_reviews <- function(){
  df <- read.csv(paste0(path, "eda_dataset/beer_profile_and_ratings.csv"))
  df <- select(df, -Description)
  
  df$Aroma_review_bucket <- cut2(df$review_aroma, c(1,2,3,3.25, 3.5,3.75, 4, 4.5, 5))
  df$Palate_review_bucket <- cut2(df$review_palate, c(1,2,3,3.25, 3.5,3.75, 4, 4.5, 5))
  df$Taste_review_bucket <- cut2(df$review_taste, c(1,2,3,3.25, 3.5,3.75, 4, 4.5, 5))
  df$Appearance_review_bucket <- cut2(df$review_appearance, c(1,2,3,3.25, 3.5,3.75, 4, 4.5, 5))
  df$Overall_review_bucket <- cut2(df$review_overall, c(1,2,3,3.25, 3.5,3.75, 4, 4.5, 5))
  
  df
}