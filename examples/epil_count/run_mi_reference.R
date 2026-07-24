# R-side MI for the epil example — a CROSS-METHOD STATISTICAL COMPARISON.
#
# Same source data (MASS::epil, hash-verified), same shared MAR mask, same
# substantive GEE and estimand as run_python.py's MI arm — but a different
# imputation model: wide-format mice predictive mean matching, versus
# longmi's negative-binomial random-intercept model. This is deliberately
# NOT backend parity and NOT numerical parity: agreement is expected at
# the level of pooled estimates and standard errors (statistical
# agreement), never imputed values.
#
# Run: Rscript examples/epil_count/run_mi_reference.R

suppressPackageStartupMessages({library(mice); library(geepack)})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
here <- normalizePath(dirname(script))
repo <- normalizePath(file.path(here, "..", ".."))

if (packageVersion("MASS") < "7.3-65") stop("MASS >= 7.3-65 required")
data("epil", package = "MASS")
frame <- data.frame(
  subject = as.integer(epil$subject), period = as.integer(epil$period),
  y = as.integer(epil$y),
  treat = as.integer(epil$trt == "progabide"),
  base = as.integer(epil$base), age = as.integer(epil$age)
)
frame$lbase1 <- log1p(frame$base)
mask <- read.csv(file.path(repo, "validation", "masks",
                           "epil_mar_seed_20260723.csv"))
frame <- merge(frame, mask, by = c("subject", "period"), sort = FALSE)
frame$y[frame$observed == 0] <- NA
frame <- frame[order(frame$subject, frame$period), ]

# wide format: one outcome column per period, baseline covariates once
wide <- reshape(
  frame[, c("subject", "period", "y", "treat", "lbase1", "age")],
  idvar = "subject", timevar = "period", direction = "wide"
)
names(wide) <- sub("^y\\.", "y", names(wide))
wide <- wide[, c("subject", "treat.1", "lbase1.1", "age.1",
                 "y1", "y2", "y3", "y4")]
names(wide)[2:4] <- c("treat", "lbase1", "age")

imp <- mice(wide[, -1], m = 20, method = "pmm", seed = 20260723,
            printFlag = FALSE)

terms_map <- c(
  "(Intercept)" = "intercept", "treat" = "treat", "period" = "period",
  "lbase1" = "lbase1", "age" = "age", "treat:period" = "treat_period"
)

fits <- vector("list", imp$m)
for (m in seq_len(imp$m)) {
  comp <- complete(imp, m)
  comp$subject <- wide$subject
  long <- reshape(
    comp, varying = c("y1", "y2", "y3", "y4"), v.names = "y",
    timevar = "period", times = 1:4, direction = "long"
  )
  long <- long[order(long$subject, long$period), ]
  fits[[m]] <- geeglm(
    y ~ treat * period + lbase1 + age,
    id = subject, data = long,
    family = poisson, corstr = "exchangeable", std.err = "san.se"
  )
}

pooled <- summary(pool(as.mira(fits)))
stopifnot(all(as.character(pooled$term) %in% names(terms_map)))
out_frame <- data.frame(
  analysis = "mi_rubin_r_pmm",
  term = unname(terms_map[as.character(pooled$term)]),
  estimate = pooled$estimate,
  robust_se = pooled$std.error
)
out <- file.path(here, "results_r_mi.csv")
write.csv(format(out_frame, digits = 17, trim = TRUE), out,
          row.names = FALSE, quote = FALSE)
cat("wrote", out, "\n")
print(out_frame, digits = 5)
