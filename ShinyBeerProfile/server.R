library(shiny)
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)
library(DT)

source(paste0(path, "ShinyBeerProfile/scripts/get_wordcloud_beer_names.R"))
source(paste0(path, "ShinyBeerProfile/scripts/get_top_breweries.R"))
source(paste0(path, "ShinyBeerProfile/scripts/get_top_styles.R"))
source(paste0(path, "ShinyBeerProfile/scripts/get_beer_statistics.R"))
source(paste0(path,'ShinyBeerProfile/scripts/get_buckets_for_reviews.R'))
source(paste0(path,'ShinyBeerProfile/scripts/get_buckets_for_abv.R'))
source(paste0(path,'ShinyBeerProfile/scripts/plot_reviews.R'))
source(paste0(path,'ShinyBeerProfile/scripts/plot_abv.R'))

pkg <- c("log4r", "bizdays", "data.table", "dplyr", "lubridate", "RPostgres", "xlsx", "rCharts", "shiny", "shinydashboard", "reshape2", "reshape", "optiRum", "plotly")
lapply(pkg, require, character.only = TRUE)

shinyServer(function(input, output) {
    
    dataset_stats <- get_beer_statistics()
    
    output$number_of_rows <- renderValueBox({
        valueBox(
            dataset_stats$number_of_rows, "Number of Rows", icon = icon("arrows-alt-v"), color = "yellow")
    })
    
    output$number_of_columns <- renderValueBox({
        valueBox(
            dataset_stats$number_of_columns, "Number of Columns", icon = icon("arrows-alt-h"), color = "purple")
    })
    
    output$number_of_numeric_columns <- renderValueBox({
        valueBox(
            dataset_stats$number_of_numeric_columns, "Number of Numeric Columns", icon = icon("dice-two"), color = "green")
    })
    
    output$number_of_chr_columns <- renderValueBox({
        valueBox(
            dataset_stats$number_of_chr_columns, "Number of Character Columns", icon = icon("cuttlefish"), color = "red")
    })
    
    output$beer_table <- DT::renderDataTable({
        df <- read.csv(paste0(path, "eda_dataset/beer_profile_and_ratings.csv"))
        df <- select(df, -Description)
        df <- df[,1:7]

        DT::datatable(df, options = list(lengthMenu = c(10, 20, 50), pageLength = 10))
    })
    
    # Beer Names Wordcloud
    output$wordcloud_beer_names <- renderWordcloud2({
        wordcloud2(data=get_wordcloud_beer_names(), size = 3, backgroundColor="#ffffff")
    })
    
    # Top Breweries
    output$top_breweries <- renderPlot({
        
        brewery_counter <- get_top_breweries()
        
        ggplot() +
            geom_col(data = brewery_counter,
                     mapping = aes(x = reorder(Brewery, n),
                                   y = n),
                     alpha = 0.8,
                     fill = "#FF6666") +
            geom_label(data = brewery_counter,
                       aes(x = reorder(Brewery, n),
                           y = n,
                           label = scales::comma(n)),
                       size = 5,
                       nudge_y = 0) +
            scale_y_continuous(labels = scales::comma) +
            scale_fill_manual(values = c("gray75", "#E83536")) +
            coord_flip() +
            labs(title = "") +
            guides(fill = "none") +
            xlab("Brewery Name") +
            ylab("Number of Reviewed Beers") +
            theme(plot.background = element_rect(fill = "#ecf0f5", colour = "#ecf0f5"),
                  panel.background = element_rect(fill = "#ecf0f5", colour = "#ecf0f5"),
                  axis.text=element_text(size=14),
                  axis.title = element_text(size=20))
    })
    
    # Top Styles
    output$top_styles <- renderPlot({
        
        styles_counter <- get_top_styles()

        g <- ggplot() +
            geom_bar(data = styles_counter,
                     aes(x = reorder(Style, n), y = n),
                     stat = "identity",
                     alpha = 0.8,
                     fill = "#6ba4fa") +
            scale_y_continuous(labels = scales::comma) +
            scale_fill_manual(values = c("gray75", "#E83536")) +
            labs(title = "") +
            guides(fill = "none") +
            xlab("Style") +
            ylab("Number of Reviewed Beers") +
            theme(plot.background = element_rect(fill = "#ecf0f5", colour = "#ecf0f5"),
                  panel.background = element_rect(fill = "#ecf0f5", colour = "#ecf0f5"),
                  axis.text=element_text(size=14),
                  axis.title = element_text(size=20),
                  axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) 
        show(g)
    })
    
    
    output$plotReviews<- renderChart2({
        data <- get_buckets_for_reviews()
        plot <- plotReviews(data = data)
        plot$channelPlot
    })
    
    
    output$plotABV<- renderChart2({
        data <- get_buckets_for_abv()
        plot <- plot_abv(data = data)
        plot$channelPlot
    })

})
