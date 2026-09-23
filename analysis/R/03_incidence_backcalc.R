## 03_incidence_backcalc.R
## Stage 2: algebraic incidence back-calculation from the fitted prevalence
## trajectory, using the same compartmental identity as the hypertension
## back-calculation project:
##
##   dP/dt = lambda * (1 - P) - delta * P * (1 - P) - r * P
##
## Solving for lambda (the quantity of interest):
##
##   lambda(t) = [ dP/dt + delta * P(t) * (1 - P(t)) + r * P(t) ] / (1 - P(t))
##
## delta = excess-mortality rate differential for the state (obesity/overweight)
## r     = adult population turnover rate
## Both are treated as fixed external constants (00_setup.R), because repeated
## cross-sectional prevalence alone cannot separately identify incidence,
## remission, and mortality. The full posterior of P(t) from stage 1 is
## propagated through this identity draw-by-draw, so the reported incidence
## intervals reflect prevalence-model uncertainty (not the fixed delta/r,
## which are examined separately in a sensitivity analysis).

source(file.path("R", "00_setup.R"))

post_pred <- readRDS(file.path("output", "prevalence_posterior_draws.rds"))
grid_index <- readRDS(file.path("output", "prevalence_grid_index.rds"))

grid_index$col <- seq_len(nrow(grid_index))

compute_incidence_draws <- function(outcome_name, delta) {
  idx <- grid_index %>% filter(outcome == outcome_name) %>% arrange(study_year)
  years <- idx$study_year
  P <- post_pred[, idx$col, drop = FALSE]   # draws x years

  ## central finite-difference derivative of P wrt calendar year, per draw
  n_t <- length(years)
  dPdt <- matrix(NA_real_, nrow = nrow(P), ncol = n_t)
  dPdt[, 2:(n_t - 1)] <- (P[, 3:n_t] - P[, 1:(n_t - 2)]) /
    matrix(rep(years[3:n_t] - years[1:(n_t - 2)], each = nrow(P)), nrow = nrow(P))
  dPdt[, 1] <- (P[, 2] - P[, 1]) / (years[2] - years[1])
  dPdt[, n_t] <- (P[, n_t] - P[, n_t - 1]) / (years[n_t] - years[n_t - 1])

  lambda <- (dPdt + delta * P * (1 - P) + R_TURNOVER * P) / (1 - P)

  tibble(
    outcome = outcome_name,
    study_year = years,
    incidence_mean = apply(lambda, 2, mean),
    incidence_lo = apply(lambda, 2, quantile, probs = 0.025),
    incidence_hi = apply(lambda, 2, quantile, probs = 0.975),
    prevalence_mean = apply(P, 2, mean)
  )
}

incidence_overweight <- compute_incidence_draws("overweight", DELTA_OVERWEIGHT)
incidence_obesity    <- compute_incidence_draws("obesity", DELTA_OBESITY)

incidence_all <- bind_rows(incidence_overweight, incidence_obesity) %>%
  mutate(
    incidence_per_1000py_mean = incidence_mean * 1000,
    incidence_per_1000py_lo = incidence_lo * 1000,
    incidence_per_1000py_hi = incidence_hi * 1000
  )

write_csv(incidence_all, file.path("output", "incidence_backcalculated.csv"))

cat("\n--- Back-calculated incidence, selected years ---\n")
incidence_all %>%
  filter(study_year %in% range(study_year) | study_year == round(median(study_year))) %>%
  mutate(across(where(is.numeric), ~round(., 2))) %>%
  print(n = Inf)

## ---- Figure: back-calculated incidence trend ----
p <- ggplot(incidence_all, aes(x = study_year, y = incidence_per_1000py_mean, colour = outcome, fill = outcome)) +
  geom_ribbon(aes(ymin = incidence_per_1000py_lo, ymax = incidence_per_1000py_hi), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  scale_colour_manual(values = c(overweight = "#2b6cb0", obesity = "#c53030")) +
  scale_fill_manual(values = c(overweight = "#2b6cb0", obesity = "#c53030")) +
  labs(
    title = "Back-calculated incidence of overweight and obesity, Nigerian adults",
    subtitle = sprintf("Compartmental identity; delta_obesity=%.3f, delta_overweight=%.3f, r=%.3f (per year)",
                        DELTA_OBESITY, DELTA_OVERWEIGHT, R_TURNOVER),
    x = "Study (fieldwork) year", y = "Incidence (per 1,000 person-years)",
    colour = "Outcome", fill = "Outcome"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path("output", "figures", "incidence_trend.png"), p, width = 8, height = 5.5, dpi = 200)

## ---- Sensitivity analysis: vary delta and r ----
sensitivity_grid <- expand_grid(
  delta_mult = c(0.5, 1, 1.5),
  r_mult = c(0.5, 1, 1.5)
)

sens_results <- pmap_dfr(sensitivity_grid, function(delta_mult, r_mult) {
  old_r <- R_TURNOVER
  R_TURNOVER <<- old_r * r_mult
  res <- bind_rows(
    compute_incidence_draws("overweight", DELTA_OVERWEIGHT * delta_mult),
    compute_incidence_draws("obesity", DELTA_OBESITY * delta_mult)
  ) %>% mutate(delta_mult = delta_mult, r_mult = r_mult)
  R_TURNOVER <<- old_r
  res
})

write_csv(sens_results, file.path("output", "incidence_sensitivity.csv"))

cat("\nStage 2 (incidence back-calculation) complete. See output/incidence_backcalculated.csv\n")
cat("and output/incidence_sensitivity.csv for the delta/r sensitivity analysis.\n")
