library(shiny)
library(shinythemes)
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)
library(DT)

sidebar <- dashboardSidebar(
    sidebarMenu(
        menuItem("Beers In Numbers", tabName = "beers-in-numbers", icon = icon("cash-register")),
        menuItem("Word My Beer", tabName = "word-my-beer", icon = icon("beer")),
        menuItem("Find Your Poison", tabName = "find-your-posion", icon = icon("flask"), badgeLabel = "new", badgeColor = "red")
    )
)

body <- dashboardBody(
    tabItems(
        tabItem(tabName = "beers-in-numbers",
                fluidPage(
               
        
                h2("Beers In Numbers"),
                br(),
                br(),
                br(),
                h2("1. Some Quick Stats", style = "color: crimson"),
                br(),
                br(),
                fluidRow(
                    column(
                        DT::dataTableOutput("beer_table"),
                        width = 8
                    ),
                    column(
                        fluidRow(
                            column(valueBoxOutput("number_of_rows"), width = 12),
                            column(valueBoxOutput("number_of_columns"), width = 12),
                            column(valueBoxOutput("number_of_numeric_columns"), width = 12),
                            column(valueBoxOutput("number_of_chr_columns"), width = 12)
                        ),
                        width = 4
                    )
                ),
                br(),
                br(),
                fluidRow(
                    column(
                        width = 12,
                        showOutput("plotReviews", "highcharts"),
                       
                        
                    )
                )
                
                
        )),
        
        tabItem(tabName = "word-my-beer",
                h1("Word My Beer!!!"),
                br(),
                h2("1. Beer Naming", style = "color: crimson"),
                wordcloud2Output("wordcloud_beer_names", height = "500px", width = "100%"),
                br(),
                br(),
                br(),
                h2("2. Top 10 Breweries", style = "color: crimson"),
                plotOutput("top_breweries", height = "600px", width = "100%"),
                br(),
                br(),
                h2("3. Top 20 Styles", style = "color: crimson"),
                plotOutput("top_styles", height = "600px", width = "100%"),
                br(),
        ),
        
        tabItem(tabName = "find-your-poison",
                h2("Find Your Poison")
        )
    )
    
)

shinyUI(dashboardPage(
    skin = "purple",
    dashboardHeader(title = "Beer Master"),
    sidebar,
    body
))
