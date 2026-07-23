# Reference values for Rubin pooling from mice::pool.scalar (rule "rubin1987").
#
# Regenerate with:
#     Rscript validation/r/rubin_reference.R
# which rewrites validation/r/rubin_reference.csv. The Python side of the
# parity test (tests/cross_language/test_rubin_r_parity.py) hardcodes the
# same fixtures and must agree at relative tolerance 1e-12.
#
# Fixtures are deterministic constants, not random draws: the pooling rules
# are deterministic, so exact cross-language agreement is the requirement.

suppressPackageStartupMessages(library(mice))

pool_case <- function(case, param, Q, U, n, k) {
  p <- pool.scalar(Q, U, n = n, k = k, rule = "rubin1987")
  m <- length(Q)
  lambda <- (1 + 1 / m) * p$b / p$t
  data.frame(
    case = case,
    param = param,
    m = m,
    dfcom = if (is.infinite(n)) NA else n - k,
    qbar = p$qbar,
    ubar = p$ubar,
    b = p$b,
    t = p$t,
    riv = p$r,
    lambda = lambda,
    fmi = p$fmi,
    df = p$df
  )
}

rows <- list()

# Case A: scalar, M = 5, large-sample reference (dfcom infinite)
QA <- c(1.5, 2.0, 2.5, 1.8, 2.2)
UA <- c(0.36, 0.44, 0.40, 0.38, 0.42)
rows[[length(rows) + 1]] <- pool_case("A", "beta", QA, UA, n = Inf, k = 1)

# Case B: same estimates, Barnard-Rubin with dfcom = 100 - 4 = 96
rows[[length(rows) + 1]] <- pool_case("B", "beta", QA, UA, n = 100, k = 4)

# Case C: scalar, M = 3, small sample, dfcom = 50 - 2 = 48
rows[[length(rows) + 1]] <- pool_case(
  "C", "beta", c(0.10, 0.30, 0.20), c(0.09, 0.11, 0.10), n = 50, k = 2
)

# Case D: three parameters, M = 5, dfcom = 200 - 6 = 194.
# mice pools per parameter from (Q_j, U_jj); longmi additionally carries the
# full covariance matrices, whose diagonals must reproduce these numbers.
QD <- matrix(
  c(
    0.90, 0.42, -0.15,
    1.10, 0.38, -0.09,
    0.95, 0.45, -0.20,
    1.05, 0.40, -0.12,
    1.00, 0.35, -0.14
  ),
  nrow = 5, byrow = TRUE
)
UD <- matrix(
  c(
    0.040, 0.0110, 0.0050,
    0.044, 0.0100, 0.0055,
    0.042, 0.0115, 0.0048,
    0.041, 0.0105, 0.0052,
    0.043, 0.0095, 0.0051
  ),
  nrow = 5, byrow = TRUE
)
params <- c("beta0", "beta1", "beta2")
for (j in seq_along(params)) {
  rows[[length(rows) + 1]] <- pool_case(
    "D", params[j], QD[, j], UD[, j], n = 200, k = 6
  )
}

reference <- do.call(rbind, rows)

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
out <- file.path(dirname(script), "rubin_reference.csv")
write.csv(format(reference, digits = 17, trim = TRUE), out,
          row.names = FALSE, quote = FALSE)
cat("wrote", out, "with", nrow(reference), "rows\n")
sessionInfo()$otherPkgs$mice$Version
