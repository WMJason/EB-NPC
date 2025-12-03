import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from sklearn.preprocessing import StandardScaler
import math # Import math for log-transform
import os
import shutil
import json


h=5

output_folder = '/content/7_EB-NPC-step1'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    # Clear output folder contents
    try:
        for ea in os.listdir(output_folder):
            file_path = os.path.join(output_folder, ea)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        print(f"Error clearing output directory: {e}")


df_nodes = pd.read_csv('/content/4_simulated_node_panel_wide.csv')
####FOR treated sites
df_treated = df_nodes[df_nodes['is_treated_flag']==1].copy()
df = df_treated.copy()

data_treated = pd.DataFrame({
  'Index': df["node_id"].values.tolist(),
  'years': 5,
  'majorAADT': df[[f'majorAADT_t_Y{year}' for year in range(1,5+1)]].mean(axis=1),
  'minorAADT': df[[f'minorAADT_t_Y{year}' for year in range(1,5+1)]].mean(axis=1),
  'observed_crashes': df[[f'count_Y{year}' for year in range(1,5+1)]].sum(axis=1),
  'if_treated': 0,
})


#####FOR untreated sites
df_untreated = df_nodes[df_nodes['is_treated_flag']==0].copy()
df = df_untreated.copy()

data_untreated = pd.DataFrame({
  'Index': df["node_id"].values.tolist(),
  'years': 10,
  'majorAADT': df[[f'majorAADT_t_Y{year}' for year in range(1,10+1)]].mean(axis=1),
  'minorAADT': df[[f'minorAADT_t_Y{year}' for year in range(1,10+1)]].mean(axis=1),
  'observed_crashes': df[[f'count_Y{year}' for year in range(1,10+1)]].sum(axis=1),
  'if_treated': 0,
})

data = pd.concat([data_treated, data_untreated])
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

# --------------------------------------------------------------------------


with pm.Model() as model:

    # Priors
    beta0 = pm.Normal("beta0", mu=0, sigma=1)  # Intercept
    
    # --- ADDED: Priors for the AADT covariates ---
    #beta_aadt = pm.Normal("beta_aadt", mu=0, sigma=1, shape=X_aadt_scaled.shape[1])
    
    # Linear predictor for AADT effects
    #mu_aadt = pm.math.dot(X_aadt_scaled, beta_aadt)
    
    var_psi = pm.Gamma("var_psi", alpha=2, beta=2)  # variance
    sigma_psi = pm.Deterministic("sigma_psi", pm.math.sqrt(var_psi))
    
    # psi random effect shape is correct (num_knots)
    psi = pm.Normal("psi", mu=0, sigma=sigma_psi, shape=num_knots) 

    # Spatial random effects with reweighted kernel
    # Z has shape (len(data),)
    Z = pm.math.dot(K_weighted, psi)

    # Expected crashes (log-link with offset)
    # log_lambda = log(years) + beta0 + AADT effects (mu_aadt) + Spatial effects (Z)
    #log_lambda = pm.math.log(data['years'].values) + beta0 + mu_aadt + Z
    log_lambda = pm.math.log(data['years'].values) + beta0 + Z
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
# Posterior Summary (include beta_aadt for the AADT coefficients)
summary = az.summary(trace, var_names=["beta0", "var_psi"], round_to=4, hdi_prob=0.95)

# Output stats
summary_stats = summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]].copy()
summary_stats.to_csv(f'7_EB-NPC_step1.csv')
print(summary_stats)
