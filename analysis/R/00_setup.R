## 00_setup.R
## Nigeria overweight/obesity incidence back-calculation
## Loads packages and defines shared constants used across the pipeline.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(stringr)
  library(ggplot2)
  library(rstanarm)
})

options(mc.cores = parallel::detectCores())
try(rstan::rstan_options(auto_write = TRUE), silent = TRUE)

set.seed(20260920)

## ---- Compartmental constants (mirrors the hypertension back-calculation) ----
## dP/dt = lambda * (1 - P) - delta * P * (1 - P) - r * P
## lambda = incidence rate (per person-year among the not-yet-affected)
## delta  = excess mortality rate differential associated with the state (obesity/overweight)
## r      = population turnover (net rate at which prevalent cases leave the adult
##          population through death/ageing-out, independent of excess mortality)
##
## These are fixed external constants, exactly as in the hypertension model, because
## the repeated cross-sectional prevalence data alone cannot identify them separately
## from incidence. Values are approximate and documented in the README; a sensitivity
## analysis varies them.
DELTA_OBESITY    <- 0.010   # excess mortality rate differential, obesity vs normal (per year)
DELTA_OVERWEIGHT <- 0.004   # excess mortality rate differential, overweight vs normal (per year)
R_TURNOVER       <- 0.020   # adult population turnover rate (per year)
