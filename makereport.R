# Set working directory to current folder
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Load relevant libraries
library(knitr)

knit2html("report.Rmd")