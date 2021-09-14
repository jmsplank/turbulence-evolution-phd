#! /usr/local/bin/Rscript

thisFile <- function() {
        cmdArgs <- commandArgs(trailingOnly = FALSE)
        needle <- "--file="
        match <- grep(needle, cmdArgs)
        if (length(match) > 0) {
                # Rscript
                return(normalizePath(sub(needle, "", cmdArgs[match])))
        } else {
                # 'source'd via R console
                return(normalizePath(sys.frames()[[1]]$ofile))
        }
}
wd <- dirname(thisFile())
print(wd)
data <- read.csv(file = paste(wd, "raw_r.csv", sep="/"))
library(earth)
mars <- earth(x = data['x'], y = data['y'], keepxy = TRUE)
prediction <- predict(mars)
write.csv(prediction, paste(wd, 'mars.csv', sep='/'), row.names = FALSE)
