# Libaries
library(data.table)
library(stringr)

# Working directory
setwd("output")

# Loop through years
for(year in 2008:2011) {
  
  # Read and format weather data
  print(paste0("Reading weather data for year ",year))
  weatherDat <- fread(paste0("output222-",year,".txt"), sep=",")
  print("Done reading.")
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
  names(finalWeather) <- c("year","month","day","a_time","a_station",
                           "a_tmpc","a_dwpc","a_relh","a_feel","a_drct",
                           "a_sped","a_alti","a_mslp","a_p01m","a_vsby","a_gust_mph",
                           "a_skyc1","a_skyl1","a_wxcodes")
  print(paste0("Done formatting weather data for year ",year))
  
  # Read and format flight data
  print(paste0("Reading merged data for year ",year))
  existingDat <- fread(paste0("full-merged22-",year,".csv"))
  print("Done reading.")
  CRS_ARR_TIME <- existingDat$CRS_ARR_TIME
  CRS_ARR_TIME <- CRS_ARR_TIME - CRS_ARR_TIME%%100 + CRS_ARR_TIME%%100*10/6
  CRS_ARR_TIME <- as.data.table(CRS_ARR_TIME)
  existingDatNew <- cbind(existingDat[, -"CRS_ARR_TIME"], CRS_ARR_TIME)
  rm(CRS_ARR_TIME)
  print(paste0("Done formatting merged data for year ",year))
  
  # Add in arrival airport codes
  print("Reading airport codes")
  airportCodes <- fread("all_airports.txt", col.names=c("Number", "arrAirport"),
                        header=FALSE)
  setkey(existingDatNew, DEST_AIRPORT_ID)
  setkey(airportCodes, Number)
  print("Merging in airport codes.")
  finalPreDat <- airportCodes[existingDatNew]
  print("...done.")
  
  # Merge weather and flight data
  print("Merging merged data and arrival weather.")
  setkey(finalPreDat, year, month, day, arrAirport, CRS_ARR_TIME)
  setkey(finalWeather, year, month, day, a_station, a_time)
  finalWeather[, a_join_time:=a_time]
  merged <- finalWeather[finalPreDat, roll="nearest"]
  print("Done...writing.")
  fwrite(merged, paste0("finalmerge-",year,".csv"))
  print(paste0("Finished with year ", year))
}