## 02_prevalence_model.R
## Stage 1: Bayesian hierarchical prevalence model.
##
## Mirrors the hypertension back-calculation project's stage 1: logistic
## regression of prevalence on (quadratic) calendar year and study mean age,
## with a per-study random intercept to absorb unmeasured study-level
## heterogeneity (setting, measurement protocol, population subgroup, etc.).
##
## Extension for this project: overweight and obesity are modelled jointly
## from the same set of studies, with outcome-specific year and age slopes but a
## SHARED per-study random intercept, since a study that measures high
## obesity tends to also measure high overweight in the same population.

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

saveRDS(fit, file.path("output", "prevalence_model_fit.rds"))

print(summary(fit, digits = 3))

## Posterior predictive check: does the model reproduce observed prevalence?
pp_check_df <- studies_long %>%
  mutate(
    fitted_p = fitted(fit),
    resid = prevalence_pct / 100 - fitted_p
  )
write_csv(pp_check_df, file.path("output", "prevalence_model_ppcheck.csv"))

cat(sprintf("Posterior predictive residual RMSE: %.4f\n", sqrt(mean(pp_check_df$resid^2))))

## ---- Predicted prevalence surface, by outcome, over calendar years ----
## The time axis is `study_year` (fieldwork year where confirmed, publication
## year fallback otherwise -- see 01_load_data.R), not raw publication year.
year_grid <- seq(min(studies_long$study_year), max(studies_long$study_year), by = 1)
mean_year <- mean(studies$study_year)
mean_age_val <- mean(studies$mean_age)

pred_grid <- expand_grid(
  outcome = factor(c("overweight", "obesity"), levels = levels(studies_long$outcome)),
  study_year = year_grid
) %>%
  mutate(
    year_c = study_year - mean_year,
    year_c2 = year_c^2,
    age_c = 0,     # predictions at the sample's mean age
    sample_size = 1,
    cases = 0,
    study_id = studies$study_id[1]   # placeholder level; ignored since re.form = NA below
  )

## Population-level (marginal, re.form = NA) posterior draws of prevalence
post_pred <- posterior_epred(fit, newdata = pred_grid, re.form = NA)
pred_grid$prevalence_mean <- apply(post_pred, 2, mean)
pred_grid$prevalence_lo   <- apply(post_pred, 2, quantile, probs = 0.025)
pred_grid$prevalence_hi   <- apply(post_pred, 2, quantile, probs = 0.975)

write_csv(pred_grid, file.path("output", "prevalence_predicted.csv"))

## Save raw posterior draws of predicted prevalence for stage 2 (incidence
## back-calculation) to propagate full uncertainty rather than plugging in
## posterior means.
saveRDS(post_pred, file.path("output", "prevalence_posterior_draws.rds"))
saveRDS(pred_grid %>% select(outcome, study_year), file.path("output", "prevalence_grid_index.rds"))

## ---- Figure: prevalence trend with 95% credible band ----
p <- ggplot(pred_grid, aes(x = study_year, y = prevalence_mean * 100, colour = outcome, fill = outcome)) +
  geom_ribbon(aes(ymin = prevalence_lo * 100, ymax = prevalence_hi * 100), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_point(
    data = studies_long,
    mapping = aes(x = study_year, y = prevalence_pct, colour = outcome, size = sample_size),
    inherit.aes = FALSE, alpha = 0.4
  ) +
  scale_colour_manual(values = c(overweight = "#2b6cb0", obesity = "#c53030")) +
  scale_fill_manual(values = c(overweight = "#2b6cb0", obesity = "#c53030")) +
  labs(
    title = "Modelled prevalence of overweight and obesity among Nigerian adults",
    subtitle = sprintf("Bayesian hierarchical logistic model, %d population-based studies (fieldwork year where known, %d/%d)",
                        nrow(studies), sum(!studies$year_imputed), nrow(studies)),
    x = "Study (fieldwork) year", y = "Prevalence (%)", colour = "Outcome", fill = "Outcome", size = "Study N"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path("output", "figures", "prevalence_trend.png"), p, width = 8, height = 5.5, dpi = 200)

cat("Stage 1 (prevalence model) complete.\n")
