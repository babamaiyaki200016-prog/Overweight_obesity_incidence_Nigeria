## run_all.R
## Runs the full R pipeline end to end. Run from the project root:
##   Rscript R/run_all.R

source(file.path("R", "02_prevalence_model.R"))
source(file.path("R", "03_incidence_backcalc.R"))

cat("\nAll done. Outputs are in output/ and output/figures/.\n")
