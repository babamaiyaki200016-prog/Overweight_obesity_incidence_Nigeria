## 01_load_data.R
## Reads the study-level dataset and reshapes it into a long "outcome-per-row"
## format for the Bayesian prevalence model (one row per study x outcome).

source(file.path("R", "00_setup.R"))

raw <- read_csv(file.path("data", "nigeria_overweight_obesity_studies.csv"),
                 show_col_types = FALSE)

## Basic sanity checks -------------------------------------------------------
stopifnot(nrow(raw) == 44)
stopifnot(all(raw$obesity_pct >= 0 & raw$obesity_pct <= 100, na.rm = TRUE))
stopifnot(all(raw$overweight_pct >= 0 & raw$overweight_pct <= 100, na.rm = TRUE))

## Reconstruct integer case counts from reported prevalence % and sample size.
## Where sex-specific prevalence is available we do NOT attempt to reconstruct
## sex-specific sample sizes (not reported), so the sex-stratified figures are
## used only for descriptive comparison, not in the fitted model.
##
## Time axis: use actual fieldwork/survey year where it could be confirmed
## from the source paper (`fieldwork_year`, `fieldwork_precision` in
## {"exact","approx"}), falling back to publication year where fieldwork year
## could not be recovered (`fieldwork_precision == "unknown"`). This mirrors
## the age-imputation pattern below and is flagged via `year_imputed` so the
## fallback is auditable rather than silently blended in. See README for the
## per-study sourcing and coverage (10 of 43 studies have a confirmed
## fieldwork year as of this pass; the rest use publication year).
studies <- raw %>%
  mutate(
    study_label = paste0(author, " (", pub_year, ")"),
    obesity_cases    = round(obesity_pct    / 100 * sample_size),
    overweight_cases = round(overweight_pct / 100 * sample_size),
    ## studies with missing mean age get the sample-size-weighted mean age of
    ## the rest of the data imputed, flagged via age_imputed
    age_imputed = is.na(mean_age),
    year_imputed = is.na(fieldwork_year),
    study_year = ifelse(is.na(fieldwork_year), pub_year, fieldwork_year)
  )

mean_age_overall <- weighted.mean(studies$mean_age, studies$sample_size, na.rm = TRUE)
studies <- studies %>%
  mutate(mean_age = ifelse(is.na(mean_age), mean_age_overall, mean_age))

## Long format: one row per study x outcome (overweight / obesity), dropping
## rows where that particular outcome was not reported by the study.
long_overweight <- studies %>%
  filter(!is.na(overweight_pct)) %>%
  transmute(study_id, study_label, zone, pub_year, study_year, year_imputed,
            sample_size, mean_age, age_imputed,
            outcome = "overweight",
            cases = overweight_cases,
            prevalence_pct = overweight_pct)

long_obesity <- studies %>%
  filter(!is.na(obesity_pct)) %>%
  transmute(study_id, study_label, zone, pub_year, study_year, year_imputed,
            sample_size, mean_age, age_imputed,
            outcome = "obesity",
            cases = obesity_cases,
            prevalence_pct = obesity_pct)

studies_long <- bind_rows(long_overweight, long_obesity) %>%
  mutate(
    year_c = study_year - mean(study_year),    # centred calendar year (fieldwork year where known)
    year_c2 = year_c^2,                        # quadratic term (as in the hypertension model)
    age_c = mean_age - mean(mean_age),         # centred mean age
    outcome = factor(outcome, levels = c("overweight", "obesity"))
  )

cat(sprintf(
  "Loaded %d studies -> %d overweight rows, %d obesity rows (%d total outcome-rows)\n",
  nrow(studies), nrow(long_overweight), nrow(long_obesity), nrow(studies_long)
))

dir.create("output", showWarnings = FALSE)
write_csv(studies, file.path("output", "studies_clean.csv"))
write_csv(studies_long, file.path("output", "studies_long.csv"))
