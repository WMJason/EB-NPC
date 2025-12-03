import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist
import os
import shutil
from shutil import copyfile
import gzip
import json
from tqdm import tqdm
import pickle


h=5

output_folder = '7_EB-NPC-step2-1'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    try:
        for ea in os.listdir(output_folder):
            os.remove(output_folder + '/' + ea)
    except:
        for ea in os.listdir(output_folder):
            shutil.rmtree(output_folder + '/' + ea)


df_nodes = pd.read_csv('/content/4_simulated_node_panel_wide.csv')
####FOR treated sites
df_treated = df_nodes[df_nodes['is_treated_flag']==1].copy()
df = df_treated.copy()

data_treated = pd.DataFrame({
  'Index': df["node_id"].values.tolist(),
  'years': 5,
  'majorAADT': df[[f'majorAADT_t_Y{year}' for year in range(5+1,10+1)]].mean(axis=1),
  'minorAADT': df[[f'minorAADT_t_Y{year}' for year in range(5+1,10+1)]].mean(axis=1),
  'observed_crashes': df[[f'count_Y{year}' for year in range(5+1,10+1)]].sum(axis=1),
  'if_treated': 0,
})



data = data_treated.copy()
print(f'There are {len(data)} data.')


with open('/content/2_sp_info_nodes.json', 'r') as f:
    sp_infos = json.load(f)


df_knots = pd.read_csv('/content/1_knots.csv')
print(f"There are {len(df_knots)} knots.")
knots_ids = df_knots["node_id"].values
num_knots = len(df_knots) # Get number of knots

K_weighted = []
for da_id in data["Index"].values:
    row = []

    knot_dict = sp_infos[da_id]
    for knot_id in [str(k) for k in knots_ids]:
        K_star = knot_dict[knot_id]["reweighted_kernel_value"]
        row.append(K_star)
    K_weighted.append(row)
K_weighted = np.array(K_weighted)

# --------------------------------------------------------------------------
## MODIFICATION: LOG-TRANSFORM AND STANDARDIZE AADT
# --------------------------------------------------------------------------

# 1. Log-transform AADT data (using log(AADT + epsilon) to handle potential zeros)
epsilon = 1e-8
data['majorAADT_log'] = data['majorAADT'].apply(lambda x: math.log(x + epsilon))
data['minorAADT_log'] = data['minorAADT'].apply(lambda x: math.log(x + epsilon))

# 2. Extract and Standardize the log-transformed AADT covariates
# Standardize AADT for improved sampling efficiency
scaler = StandardScaler()
X_aadt = data[['majorAADT_log', 'minorAADT_log']].values
X_aadt_scaled = scaler.fit_transform(X_aadt)
# X_aadt_scaled shape is (N_sites, 2)
X = X_aadt_scaled.copy()
# --------------------------------------------------------------------------

df_model = pd.read_csv(f'/content/7_EB-NPC_step1.csv')
with pm.Model() as model:

    # Priors
    beta0 = pm.Normal("beta0", mu=0, sigma=1)  # Tightened priors
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
    var_psi_val = df_model['mean'].values.tolist()[1]
    var_psi_val = pm.Data('var_psi_val', var_psi_val)
    var_psi = pm.Deterministic("var_psi", var_psi_val)
    sigma_psi = pm.Deterministic("sigma_psi", pm.math.sqrt(var_psi))
    psi = pm.Normal("psi", mu=0, sigma=sigma_psi, shape=len(knots_ids))

    # Spatial random effects with reweighted kernel
    Z = pm.math.dot(K_weighted, psi)

    # Expected crashes (log-link with offset)
    log_lambda = pm.math.log(data['years'].values) + beta0 + pm.math.dot(X, beta) + Z
    lambda_ = pm.Deterministic("lambda_", pm.math.exp(log_lambda))

    # Likelihood
    y_obs = pm.Poisson("y_obs", mu=lambda_, observed=data['observed_crashes'].values)

    # Sample
    trace = pm.sample(draws=3000,
                    tune=1000,
                    chains=2,
                    target_accept=0.95,         # More cautious steps
                    max_treedepth=15,
                    random_state=42,
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True})

# Summarize the posterior
# Posterior Summary
summary = az.summary(trace, var_names=["beta0", "beta", "var_psi"], round_to=4, hdi_prob=0.95)

beta_labels = ["majorAADT_log", "minorAADT_log"]#
summary.index = ["Intercept (beta0)"] + [f"Beta[{i+1}] - {label}" for i, label in enumerate(beta_labels)] + ["Var_psi"]#

# Output stats
summary_stats = summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]].copy()
summary_stats.to_csv(f'7_EB-NPC_step2-1.csv')
print(summary_stats)


# --------------------------------------------------------------------------
## MODIFICATION: POSTERIOR PREDICTIVE SUMMARY
# --------------------------------------------------------------------------

# 1. Extract the posterior samples of lambda_ (expected crash rate)
lambda_samples = trace.posterior['lambda_'].stack(sample=('chain', 'draw')).values.T

# 2. Calculate summary statistics (mean, 2.5% HDI, 97.5% HDI) for each site
lambda_summary = az.hdi(lambda_samples, hdi_prob=0.95)

# The mean is the 50th percentile of the samples
median_lambda = np.median(lambda_samples, axis=0)

# 3. Construct the output DataFrame
results_df = pd.DataFrame({
    'node_id': data['Index'].values,
    'observed_crashes': data['observed_crashes'].values,
    'mean_lambda_post': np.mean(lambda_samples, axis=0),
    'median_lambda_post': median_lambda,
    'hdi_2.5%_lambda_post': lambda_summary[:, 0],
    'hdi_97.5%_lambda_post': lambda_summary[:, 1]
})

# 4. Save the results
results_filename = f'7_EB-NPC_step2-1_predictions.csv'
results_df.to_csv(results_filename, index=False)
print(f"\n✅ Site-specific posterior predictions (lambda) exported to {results_filename}")










