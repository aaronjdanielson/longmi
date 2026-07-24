# R side: mice(method="logreg") on the same data/mask -> geeglm binomial
# -> mice::pool. CROSS-METHOD STATISTICAL comparison, not backend parity.
suppressPackageStartupMessages({library(mice); library(geepack)})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("--file=", "", grep("--file=", args, value = TRUE)))
f <- read.csv(file.path(here, "shared_binary.csv"))
f$y[f$observed == 0] <- NA
wide <- reshape(f[, c("pid","wave","y")], idvar = "pid",
                timevar = "wave", direction = "wide")
names(wide) <- sub("^y\\.", "y", names(wide))
wide <- merge(wide, unique(f[, c("pid","treat")]), by = "pid")
wide$y1 <- factor(wide$y1); wide$y2 <- factor(wide$y2); wide$y3 <- factor(wide$y3)
imp <- mice(wide[, -1], m = 20, method = "logreg", seed = 20260725,
            printFlag = FALSE)
fits <- lapply(seq_len(imp$m), function(k) {
  comp <- complete(imp, k); comp$pid <- wide$pid
  long <- reshape(comp, varying = c("y1","y2","y3"), v.names = "y",
                  timevar = "wave", times = 1:3, direction = "long")
  long$y <- as.integer(as.character(long$y))
  long <- long[order(long$pid, long$wave), ]
  geeglm(y ~ treat * factor(wave), id = pid, data = long,
         family = binomial, corstr = "exchangeable", std.err = "san.se")
})
pl <- summary(pool(as.mira(fits)))
write.csv(data.frame(term = as.character(pl$term), estimate = pl$estimate,
                     se = pl$std.error),
          file.path(here, "results_mice.csv"), row.names = FALSE)
print(pl[, c("term","estimate","std.error")], digits = 3)
