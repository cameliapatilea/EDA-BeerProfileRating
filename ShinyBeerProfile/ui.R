library(shiny)
library(shinythemes)
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)
library(DT)
require(highcharter)

sidebar <- dashboardSidebar(
    sidebarMenu(
        menuItem("Beers In Numbers", tabName = "beers-in-numbers", icon = icon("cash-register")),
        menuItem("Word My Beer", tabName = "word-my-beer", icon = icon("beer")),
        menuItem("Find Your Poison", tabName = "find-your-poison", icon = icon("flask"), badgeLabel = "new", badgeColor = "red")
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
                        width = 12
                    )
                ),
                br(),
                br(),
                fluidRow(
                    column(width = 2),
                    column(width = 10,
                           valueBoxOutput("number_of_rows"),
                           valueBoxOutput("number_of_columns"))
                ),
                fluidRow(),
                fluidRow(
                    column(width = 2),
                    column(width = 10,
                           valueBoxOutput("number_of_numeric_columns"),
                           valueBoxOutput("number_of_chr_columns"))
                ),
                fluidRow(),
                br(),
                br(),
                h2("2. Beer evaluation based on reviews", style = "color: crimson"),
                br(),
                br(),
                fluidRow(
                    column(
                        width = 12,
                        showOutput("plotReviews", "highcharts"),
                    )
                ),
                br(),
                br(),
                br(),
                h2("3. Alcohol by volume evaluation", style = "color: crimson"),
                h3("ABV or Alcohol by volume is a standard measure of how much alcohol (ethanol) is contained in a given volume of an alcoholic beverage (expressed as a volume percent). It is defined as the number of millilitres (mL) of pure ethanol present in 100 ml of solution at 20 °C.", style = "color: black"),
                br(),
                fluidRow(
                    column(
                        width = 8,
                        showOutput("plotABV", "highcharts"),
                    ),
                    column(
                        width = 3,
                        fluidRow()
                    )
                ),
                br(),
                fluidRow(
                    column(width = 4),
                    column(width = 4,
                           box(
                               title = h2("ABV (Alcohol By Volume)"),
                               width = NULL,
                               background = "teal",
                               tags$ul(
                                   tags$li(h4("[0, 1) - Considered to be alcohol free beer")),
                                   tags$li(h4("[1, 4) - Low alcohol beer")),
                                   tags$li(h4("[4, 6) - Considered to be normal beer")),
                                   tags$li(h4("[6, 10) - Average alcohol for cidre")),
                                   tags$li(h4("[10, 12) - Strong beers")),
                                   tags$li(h4("[12, 60] - Strong beers and outliers")),
                               )
                           ),
                    column(width = 4))
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
