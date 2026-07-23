# R side of the epil first validation task (see run_python.py).
#
# Loads MASS::epil from the locally installed MASS package, normalizes it to
# the shared schema, verifies the SHA-256 of the normalized integer core
# against epil_provenance.json's value, applies the shared MAR mask, and
# fits the same Poisson GEE (exchangeable, clustered by subject) with
# geepack, writing results_r.csv with language-neutral term names.
#
# Run from anywhere:
#     Rscript examples/epil_count/run_reference.R

suppressPackageStartupMessages(library(geepack))

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
here <- normalizePath(dirname(script))
repo <- normalizePath(file.path(here, "..", ".."))

# -- load and normalize ----------------------------------------------------

# MASS 7.3-65 corrected y in row 31 (subject 8, period 3) from 21 to 23;
# older MASS copies silently disagree with the Rdatasets mirror Python uses.
if (packageVersion("MASS") < "7.3-65") {
  stop("MASS >= 7.3-65 required (epil row-31 erratum); found ",
       packageVersion("MASS"))
}

data("epil", package = "MASS")
frame <- data.frame(
  subject = as.integer(epil$subject),
  period = as.integer(epil$period),
  y = as.integer(epil$y),
  treat = as.integer(epil$trt == "progabide"),
  base = as.integer(epil$base),
  age = as.integer(epil$age)
)
frame <- frame[order(frame$subject, frame$period), ]
rownames(frame) <- NULL
stopifnot(nrow(frame) == 236, length(unique(frame$subject)) == 59)
frame$lbase1 <- log1p(frame$base)

# -- hash the normalized integer core (must equal the Python value) --------

core_csv <- tempfile(fileext = ".csv")
write.table(
  frame[, c("subject", "period", "y", "treat", "base", "age")],
  core_csv, sep = ",", quote = FALSE, row.names = FALSE, eol = "\n"
)
hash <- strsplit(system(paste("shasum -a 256", shQuote(core_csv)), intern = TRUE), " ")[[1]][1]
cat("data:", nrow(frame), "rows, sha256", hash, "\n")

provenance <- file.path(here, "epil_provenance.json")
if (file.exists(provenance)) {
  text <- paste(readLines(provenance), collapse = "")
  expected <- sub('.*"sha256_normalized_core": "([0-9a-f]+)".*', "\\1", text)
  if (!identical(hash, expected)) {
    stop("normalized-data hash mismatch: R ", hash, " vs Python ", expected)
  }
  cat("hash matches epil_provenance.json\n")
}

# -- shared MAR mask -------------------------------------------------------

mask <- read.csv(file.path(repo, "validation", "masks", "epil_mar_seed_20260723.csv"))
merged <- merge(frame, mask, by = c("subject", "period"), sort = FALSE)
merged <- merged[order(merged$subject, merged$period), ]
available <- merged[merged$observed == 1, ]
cat(
  "MAR mask:", sum(mask$observed), "observed cells,",
  nrow(available), "available-case rows\n"
)

# -- Poisson GEE, exchangeable, robust (sandwich) SEs ----------------------

terms_map <- c(
  "(Intercept)" = "intercept",
  "treat" = "treat",
  "period" = "period",
  "lbase1" = "lbase1",
  "age" = "age",
  "treat:period" = "treat_period"
)

fit_gee <- function(data, analysis) {
  fit <- geeglm(
    y ~ treat * period + lbase1 + age,
    id = subject, data = data,
    family = poisson, corstr = "exchangeable", std.err = "san.se"
  )
  coefs <- summary(fit)$coefficients
  stopifnot(all(rownames(coefs) %in% names(terms_map)))
  data.frame(
    analysis = analysis,
    term = unname(terms_map[rownames(coefs)]),
    estimate = coefs$Estimate,
    robust_se = coefs$Std.err,
    n_rows = nrow(data),
    n_subjects = length(unique(data$subject))
  )
}

results <- rbind(
  fit_gee(frame, "complete"),
  fit_gee(available, "available_case")
)

out <- file.path(here, "results_r.csv")
write.csv(format(results, digits = 17, trim = TRUE), out,
          row.names = FALSE, quote = FALSE)
cat("wrote", out, "\n")
print(results, digits = 6)
