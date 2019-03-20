# Libaries
library(data.table)

# Work directory
setwd("output")

# Loop through years
for(year in 2008:2011) {
  print(paste("reading airports list for", year))
  allDat <- fread(paste0("finalmerge-",year,".csv"))
  airports <- unique(allDat$station)
  
  setwd("../newData")
  for(airport in airports) {
    print(paste0("reading data for ", airport, " in ", year))
    airDat <- fread(paste0("finalmerge-", airport, "-", year, ".csv"))
    
    # Split weather codes
    print(paste("splitting weather codes for departure"))
    codes <- strsplit(as.character(airDat$wxcodes), ' ')
    airDat$id <- seq_len(nrow(airDat))
    tmp1 <- data.frame(id = factor(rep(airDat$id, times = lengths(codes)),
                                   levels = airDat$id),
                       amenities = unlist(codes))
    tmp2 <- as.data.table(cbind(id = airDat$id,
                                table(tmp1$id, tmp1$amenities)))
    setkey(airDat, id)
    setkey(tmp2, id)
    mergedDat <- airDat[tmp2]
    mergedDat <- mergedDat[, -"wxcodes"]
    
    # Split weather codes
    print(paste("splitting weather codes for arrival"))
    codes <- strsplit(as.character(airDat$a_wxcodes), ' ')
    tmp1 <- data.frame(id = factor(rep(airDat$id, times = lengths(codes)),
                                   levels = airDat$id),
                       amenities = unlist(codes))
    tmp2 <- as.data.table(cbind(id = airDat$id,
                                table(tmp1$id, tmp1$amenities)))
    setkey(airDat, id)
    setkey(tmp2, id)
    mergedDat <- airDat[tmp2]
    mergedDat <- mergedDat[, -"a_wxcodes"]
    
    # Categorical variables
    categories = c("month", "OP_UNIQUE_CARRIER", "station",
                   "CANCELLATION_CODE", "skyc1", "a_station", "a_skyc1")
    oneHotted <- mergedDat
    
    for(cat in categories) {
      catCol <- oneHotted[[cat]]
      values <- unique(catCol)
      toBeAdded <- data.table()
      initial <- 0
      print(paste0("trying to one-hot variable ", cat))
      
      # One-hotting
      for(val in values) {
        oneHot <- as.data.table(as.numeric(catCol==val))
        names(oneHot) <- paste0(cat,".",val)
        if(initial==0) {
          toBeAdded <- oneHot
          initial <- 1
        } else {
          toBeAdded <- cbind(toBeAdded, oneHot)
        }
      }
      
      # Adding to datatable
      print(paste0("trying to cbind variable ", cat))
      write(paste0("oneHotted <- cbind(oneHotted[,-\"", cat, "\"], toBeAdded)"), "tmp.R")
      source("tmp.R")
      file.remove("tmp.R")
    }
    
    # Writing data
    print(paste("writing data for",airport, "in", year))
    fwrite(oneHotted, paste0("onehot-finalmerge-",airport,"-",year,".csv"))
  }
  setwd("../output")
}
