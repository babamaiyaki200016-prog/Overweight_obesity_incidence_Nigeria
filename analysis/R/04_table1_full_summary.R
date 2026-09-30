## 04_table1_full_summary.R
## Re-fits Stage 1 (same spec/seed as 02_prevalence_model.R) purely to extract
## a complete reporting summary for Table 1: posterior mean, 95% credible
## interval, Rhat, and effective sample size for every fixed effect, plus the
## study-level random-intercept SD. Requested in peer review (Dr Faruk,
## endocrinology) because the original Table 1 rounded the NumPyro column to
## 2 decimals, which is too coarse to judge R/Python concordance, and omitted
## uncertainty/ESS entirely.

source(file.path("R", "00_setup.R"))
source(file.path("R", "01_load_data.R"))

fit <- stan_glmer(
  cbind(cases, sample_size - cases) ~ outcome + outcome:year_c + outcome:year_c2 + outcome:age_c +
    (1 | study_id),
  data = studies_long,
  family = binomial(link = "logit"),
  prior = normal(0, 2.5, autoscale = TRUE),
  prior_intercept = normal(0, 5, autoscale = TRUE),
  prior_covariance = decov(regularization = 2, concentration = 1, shape = 1, scale = 1),
  chains = 4, iter = 4000, warmup = 1500, seed = 20260920,
  adapt_delta = 0.995, control = list(max_treedepth = 15)
)

ci95 <- posterior_interval(fit, prob = 0.95)
s <- summary(fit)

fixed_names <- c("(Intercept)", "outcomeobesity", "outcomeoverweight:year_c",
                  "outcomeobesity:year_c", "outcomeoverweight:year_c2",
                  "outcomeobesity:year_c2", "outcomeoverweight:age_c",
                  "outcomeobesity:age_c")

out <- data.frame(
  parameter = fixed_names,
  mean = round(s[fixed_names, "mean"], 4),
  ci_lo = round(ci95[fixed_names, 1], 4),
  ci_hi = round(ci95[fixed_names, 2], 4),
  rhat = round(s[fixed_names, "Rhat"], 3),
  n_eff = round(s[fixed_names, "n_eff"], 0)
)
print(out, row.names = FALSE)

sigma_name <- "Sigma[study_id:(Intercept),(Intercept)]"
sigma_sd <- sqrt(s[sigma_name, "mean"])
sigma_ci <- sqrt(ci95[sigma_name, ])
cat(sprintf("\nStudy-level random-intercept SD (sqrt of variance component): %.3f (95%% CrI %.3f-%.3f)\n",
            sigma_sd, sigma_ci[1], sigma_ci[2]))
cat(sprintf("Rhat: %.3f, n_eff: %.0f\n", s[sigma_name, "Rhat"], s[sigma_name, "n_eff"]))

write_csv(out, file.path("output", "table1_full_summary.csv"))
