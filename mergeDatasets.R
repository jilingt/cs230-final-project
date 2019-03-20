# Libaries
library(data.table)
library(stringr)

# Working directory
setwd("output")

# Loop through years
for(year in 2008:2011) {
  
  # Read and format weather data
  weatherDat <- fread(paste0("output222-",year,".txt"), sep=",")
  tmp <- as.data.table(str_split_fixed(weatherDat$valid, " ", 2))
  colnames(tmp) <- c("date", "time")
  timeDat <- as.numeric(str_remove(tmp$time, ":"))
  timeDat <- timeDat - timeDat%%100 + timeDat%%100*10./6
  timeDat <- as.data.table(timeDat)
  names(timeDat) <- c("time")
  tmp2 <- as.data.table(str_split_fixed(tmp$date, "-", 3))
  colnames(tmp2) <- c("year", "month", "day")
  tmp2 <- transform(tmp2, year=as.numeric(year), month=as.numeric(month), day=as.numeric(day))
  newCols <- cbind(tmp2, timeDat)
  finalWeather <- cbind(newCols, weatherDat[, -"valid"])
  
  # Read and format flight data
  airplaneDat <- fread(paste0("full-",year,".txt"))
  airplaneDat <- airplaneDat[, -"V15"]
  CRS_DEP_TIME <- airplaneDat$CRS_DEP_TIME
  CRS_DEP_TIME <- CRS_DEP_TIME - CRS_DEP_TIME%%100 + CRS_DEP_TIME%%100*10/6
  CRS_DEP_TIME <- as.data.table(CRS_DEP_TIME)
  airplaneDatNew <- cbind(airplaneDat[, -"CRS_DEP_TIME"], CRS_DEP_TIME)
  rm(CRS_DEP_TIME)
  
  # Merge weather and flight data
  setkey(airplaneDatNew, YEAR, MONTH, DAY_OF_MONTH, ORIGIN, CRS_DEP_TIME)
  setkey(finalWeather, year, month, day, station, time)
  finalWeather[, join_time:=time]
  merged <- finalWeather[airplaneDatNew, roll="nearest"]
  fwrite(merged, paste0("full-merged22-",year,".csv"))
}



