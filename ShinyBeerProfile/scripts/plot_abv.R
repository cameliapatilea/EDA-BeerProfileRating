plot_abv <- function(data = c()){
  data_table <- data
  data_plot <- data
  data_plot$count <-  round((data_plot$count / sum(data_plot$count))*100,2)
  
  
  channelPlot <- Highcharts$new()
  
  channelPlot$xAxis(categories = as.character(sort(unique(data_plot$bucket))))
  channelPlot$series(name = 'Buckets', type = 'column', data = data_plot$count)
  
  channelPlot$plotOptions(line = list(dataLabels = list(enabled = T, x = 4, y = -10, format = "{point.y:.2f} %")))
  
  channelPlot$title(text = paste0("Alcohol by volume"))
  
  channelPlot$yAxis(list(list(title = list(text = "Percentage of how many beers are in a certain bucket")
                              , alignTicks = TRUE
                              , max = 50)
                         ))          
  
  channelPlot$chart(zoomType = "xy")
  
  channelPlot$set(width = 1000, height = 600)
  
  channelPlot$tooltip(shared = TRUE)
  
  list(channelPlot = channelPlot)
}