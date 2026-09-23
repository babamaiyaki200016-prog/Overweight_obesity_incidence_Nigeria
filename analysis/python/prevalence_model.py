"""
prevalence_model.py

Python (NumPyro) cross-check of the R (rstanarm/Stan) Bayesian prevalence
model for Nigerian adult overweight/obesity, and the same compartmental
incidence back-calculation.

This mirrors the structure of the hypertension incidence project: the two
implementations (NumPyro here, rstanarm in R/02_prevalence_model.R) are
fit independently on the same data as a cross-validation of the modelling
approach, not as two different analyses.

Run from the project root:
    python3 python/prevalence_model.py
"""
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_median

numpyro.set_host_device_count(2)

DATA_PATH = "data/nigeria_overweight_obesity_studies.csv"
OUT_DIR = "output"

DELTA_OBESITY = 0.010
DELTA_OVERWEIGHT = 0.004
R_TURNOVER = 0.020


def load_long_data():
    raw = pd.read_csv(DATA_PATH)
    raw["mean_age"] = raw["mean_age"].fillna(
        np.average(raw["mean_age"].dropna(),
                    weights=raw.loc[raw["mean_age"].notna(), "sample_size"])
    )
    # Time axis: fieldwork year where confirmed from the source paper,
    # publication year fallback otherwise (mirrors R/01_load_data.R).
    raw["study_year"] = raw["fieldwork_year"].fillna(raw["pub_year"])

    rows = []
    for _, r in raw.iterrows():
        if pd.notna(r["overweight_pct"]):
            rows.append(dict(
                study_id=r["study_id"], study_year=r["study_year"],
                sample_size=r["sample_size"], mean_age=r["mean_age"],
                outcome="overweight",
                cases=round(r["overweight_pct"] / 100 * r["sample_size"]),
            ))
        if pd.notna(r["obesity_pct"]):
            rows.append(dict(
                study_id=r["study_id"], study_year=r["study_year"],
                sample_size=r["sample_size"], mean_age=r["mean_age"],
                outcome="obesity",
                cases=round(r["obesity_pct"] / 100 * r["sample_size"]),
            ))
    long_df = pd.DataFrame(rows)
    long_df["year_c"] = long_df["study_year"] - long_df["study_year"].mean()
    long_df["year_c2"] = long_df["year_c"] ** 2
    long_df["age_c"] = long_df["mean_age"] - long_df["mean_age"].mean()
    long_df["outcome_idx"] = (long_df["outcome"] == "obesity").astype(int)  # 0=overweight,1=obesity
    long_df["study_idx"], study_levels = pd.factorize(long_df["study_id"])
    return long_df, study_levels


def model(outcome_idx, year_c, year_c2, age_c, study_idx, n_study, n, cases=None):
    # Fixed effects: intercept + outcome, outcome-specific year/year^2/age slopes
    a0 = numpyro.sample("a0", dist.Normal(0, 5))
    a_obesity = numpyro.sample("a_obesity", dist.Normal(0, 5))

    b_year = numpyro.sample("b_year", dist.Normal(0, 2.5), sample_shape=(2,))
    b_year2 = numpyro.sample("b_year2", dist.Normal(0, 2.5), sample_shape=(2,))
    b_age = numpyro.sample("b_age", dist.Normal(0, 2.5), sample_shape=(2,))

    sigma_study = numpyro.sample("sigma_study", dist.HalfNormal(1.0))
    with numpyro.plate("studies", n_study):
        u = numpyro.sample("u", dist.Normal(0, sigma_study))

    logit_p = (
        a0 + a_obesity * outcome_idx
        + b_year[outcome_idx] * year_c
        + b_year2[outcome_idx] * year_c2
        + b_age[outcome_idx] * age_c
        + u[study_idx]
    )
    numpyro.sample("cases", dist.Binomial(total_count=n, logits=logit_p), obs=cases)


def main():
    long_df, study_levels = load_long_data()
    n_study = len(study_levels)

    rng_key = random.PRNGKey(20260921)
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy=init_to_median(num_samples=50))
    mcmc = MCMC(kernel, num_warmup=3000, num_samples=2500, num_chains=4, progress_bar=True)
    mcmc.run(
        rng_key,
        outcome_idx=jnp.array(long_df["outcome_idx"].values),
        year_c=jnp.array(long_df["year_c"].values),
        year_c2=jnp.array(long_df["year_c2"].values),
        age_c=jnp.array(long_df["age_c"].values),
        study_idx=jnp.array(long_df["study_idx"].values),
        n_study=n_study,
        n=jnp.array(long_df["sample_size"].values),
        cases=jnp.array(long_df["cases"].values),
    )
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # ---- Predicted prevalence surface over calendar years (population-level,
    #      i.e. marginalising out the study random effect) ----
    year_grid = np.arange(long_df["study_year"].min(), long_df["study_year"].max() + 1)
    _raw_for_mean = pd.read_csv(DATA_PATH)
    mean_year = _raw_for_mean["fieldwork_year"].fillna(_raw_for_mean["pub_year"]).mean()
    year_c_grid = year_grid - mean_year

    results = []
    draws_by_outcome = {}
    for outcome_name, oidx in [("overweight", 0), ("obesity", 1)]:
        logit_p = (
            samples["a0"][:, None]
            + samples["a_obesity"][:, None] * oidx
            + samples["b_year"][:, oidx][:, None] * year_c_grid[None, :]
            + samples["b_year2"][:, oidx][:, None] * (year_c_grid ** 2)[None, :]
            + samples["b_age"][:, oidx][:, None] * 0.0
        )
        p = jax.nn.sigmoid(logit_p)
        p = np.array(p)
        results.append(pd.DataFrame({
            "outcome": outcome_name,
            "study_year": year_grid,
            "prevalence_mean": p.mean(axis=0),
            "prevalence_lo": np.quantile(p, 0.025, axis=0),
            "prevalence_hi": np.quantile(p, 0.975, axis=0),
        }))
        draws_by_outcome[outcome_name] = p

    pred_df = pd.concat(results, ignore_index=True)
    pred_df.to_csv(f"{OUT_DIR}/prevalence_predicted_python.csv", index=False)

    # ---- Stage 2: incidence back-calculation (same identity as R) ----
    inc_rows = []
    for df_block, delta in zip(results, [DELTA_OVERWEIGHT, DELTA_OBESITY]):
        outcome_name = df_block["outcome"].iloc[0]
        P = draws_by_outcome[outcome_name]  # draws x years
        years = df_block["study_year"].values
        dPdt = np.gradient(P, years, axis=1)
        lam = (dPdt + delta * P * (1 - P) + R_TURNOVER * P) / (1 - P)
        inc_rows.append(pd.DataFrame({
            "outcome": df_block["outcome"].iloc[0],
            "study_year": years,
            "incidence_mean": lam.mean(axis=0),
            "incidence_lo": np.quantile(lam, 0.025, axis=0),
            "incidence_hi": np.quantile(lam, 0.975, axis=0),
        }))
    inc_df = pd.concat(inc_rows, ignore_index=True)
    inc_df["incidence_per_1000py_mean"] = inc_df["incidence_mean"] * 1000
    inc_df["incidence_per_1000py_lo"] = inc_df["incidence_lo"] * 1000
    inc_df["incidence_per_1000py_hi"] = inc_df["incidence_hi"] * 1000
    inc_df.to_csv(f"{OUT_DIR}/incidence_backcalculated_python.csv", index=False)

    print("\nPython (NumPyro) cross-check complete.")
    print(inc_df.round(3))


if __name__ == "__main__":
    main()
