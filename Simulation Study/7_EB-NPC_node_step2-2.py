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

output_folder = '/content/7_EB-NPC-step2-2'
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
X = X_aadt_scaled.copy()

# --- SAVE THE SCALER ---
with open('/content/7_aadt_scaler_step2-2.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# --------------------------------------------------------------------------

df_model = pd.read_csv(f'/content/7_EB-NPC_step1.csv')
with pm.Model() as model:

    # Priors
    beta0 = pm.Normal("beta0", mu=0, sigma=1)  # Intercept
    
    # --- ADDED: Priors for the AADT covariates ---
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
    
    var_psi_val = df_model['mean'].values.tolist()[1]
    var_psi_val = pm.Data('var_psi_val', var_psi_val)
    var_psi = pm.Deterministic("var_psi", var_psi_val)
    sigma_psi = pm.Deterministic("sigma_psi", pm.math.sqrt(var_psi))
    psi = pm.Normal("psi", mu=0, sigma=sigma_psi, shape=len(knots_ids))

    # Spatial random effects with reweighted kernel
    # Z has shape (len(data),)
    Z = pm.math.dot(K_weighted, psi)

    # Expected crashes (log-link with offset)
    # log_lambda = log(years) + beta0 + AADT effects (mu_aadt) + Spatial effects (Z)
    log_lambda = pm.math.log(data['years'].values) + beta0 + pm.math.dot(X, beta) + Z
    #log_lambda = pm.math.log(data['years'].values) + beta0 + Z
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
summary = az.summary(trace, var_names=["beta0", "beta", "var_psi"], round_to=4, hdi_prob=0.95)

beta_labels = ["majorAADT_log", "minorAADT_log"]#
summary.index = ["Intercept (beta0)"] + [f"Beta[{i+1}] - {label}" for i, label in enumerate(beta_labels)] + ["Var_psi"]#

# Output stats
summary_stats = summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]].copy()
summary_stats.to_csv(f'7_EB-NPC_step2-2.csv')
print(summary_stats)

####predict counterfactual

#### Predict Counterfactual (EB-NPC)

import os, json, math, pickle
import numpy as np
import pandas as pd
import arviz as az
from tqdm import tqdm

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
OUTPUT_FOLDER = '7_EB-NPC-counterfactual-predict'
CALIBRATION_TRACE_FILE = ''  # If using trace variable directly, leave blank
SCALER_FILE = '/content/7_aadt_scaler_step2-2.pkl'
NODES_FILE = '/content/4_simulated_node_panel_wide.csv'
SP_INFO_FILE = '/content/2_sp_info_nodes.json'
KNOTS_FILE = '/content/1_knots.csv'

TREATMENT_START_YEAR = 6
AFTER_PERIOD_YEARS = 5
END_YEAR = TREATMENT_START_YEAR + AFTER_PERIOD_YEARS

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("--- Starting Counterfactual Prediction (EB-NPC) ---")

# ----------------------------------------------------
# 1. PREPARE DATA FOR TREATED SITES (AFTER PERIOD ONLY)
# ----------------------------------------------------
df_nodes = pd.read_csv(NODES_FILE)
df_treated = df_nodes[df_nodes['is_treated_flag'] == 1].copy()

data_treated_after = pd.DataFrame({
    'Index': df_treated["node_id"].values,
    'years': AFTER_PERIOD_YEARS,
    'majorAADT': df_treated[[f'majorAADT_t_Y{y}' for y in range(TREATMENT_START_YEAR, END_YEAR)]].mean(axis=1),
    'minorAADT': df_treated[[f'minorAADT_t_Y{y}' for y in range(TREATMENT_START_YEAR, END_YEAR)]].mean(axis=1),
    'observed_crashes': df_treated[[f'count_Y{y}' for y in range(TREATMENT_START_YEAR, END_YEAR)]].sum(axis=1),
    'if_treated': 1
})

data_predict = data_treated_after.copy()
print(f'Using {len(data_predict)} treated sites for counterfactual prediction.')

# ----------------------------------------------------
# 2. LOAD SCALER + TRANSFORM AADT
# ----------------------------------------------------
epsilon = 1e-8
data_predict['majorAADT_log'] = data_predict['majorAADT'].apply(lambda x: math.log(x + epsilon))
data_predict['minorAADT_log'] = data_predict['minorAADT'].apply(lambda x: math.log(x + epsilon))

try:
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"ERROR: Scaler file not found at {SCALER_FILE}.")

X_aadt = data_predict[['majorAADT_log', 'minorAADT_log']].values
X_aadt_scaled = scaler.transform(X_aadt)
print("AADT covariates transformed and scaled.")

# ----------------------------------------------------
# 3. PREPARE SPATIAL DATA (K_weighted)
# ----------------------------------------------------
with open(SP_INFO_FILE, 'r') as f:
    sp_infos = json.load(f)

df_knots = pd.read_csv(KNOTS_FILE)
knots_ids = df_knots["node_id"].values

K_weighted = []
for node_id in data_predict["Index"].values:
    knot_dict = sp_infos[str(node_id)]
    row = [knot_dict[str(k)]["reweighted_kernel_value"] for k in knots_ids]
    K_weighted.append(row)

K_weighted = np.array(K_weighted)
print(f"K_weighted matrix shape: {K_weighted.shape}")

# ----------------------------------------------------
# 4. LOAD POSTERIORS FROM STEP 2.2
# ----------------------------------------------------
try:
    # Uncomment if using netCDF:
    # trace_step2_2 = az.from_netcdf(CALIBRATION_TRACE_FILE)
    trace_step2_2 = trace
except FileNotFoundError:
    raise FileNotFoundError(f"ERROR: Trace file not found at {CALIBRATION_TRACE_FILE}.")

stacked = trace_step2_2.posterior.stack(sample=("chain", "draw"))
N_samples = stacked.sample.size
print(f"Loaded {N_samples} posterior samples.")

# β0 samples: (N_samples,)
beta0_samples = stacked["beta0"].values

# β samples BEFORE FIX: shape (2, N_samples)
beta_samples_raw = stacked["beta"].values   # Shape: (covariate_dim, N_samples?)

# FIX β shape → (N_samples, 2)
beta_samples = beta_samples_raw.T
if beta_samples.shape[1] != 2:
    raise ValueError(f"ERROR: beta_samples shape {beta_samples.shape} but expected (N_samples, 2)")

# ψ samples BEFORE FIX: shape (N_knots, N_samples)
psi_samples_raw = stacked["psi"].values

# FIX ψ shape → (N_samples, N_knots)
psi_samples = psi_samples_raw.T
if psi_samples.shape[0] != N_samples:
    raise ValueError(f"ERROR: psi_samples shape {psi_samples.shape} but expected (N_samples, N_knots)")

print(f"Beta samples: {beta_samples.shape}, Psi samples: {psi_samples.shape}")

# ----------------------------------------------------
# 5. PREDICT COUNTERFACTUAL λ_cf
# ----------------------------------------------------
log_lambda_cf_samples = []
log_years = np.log(data_predict['years'].values)

for i in tqdm(range(N_samples), desc="Predicting λ_cf"):

    # AADT effect: (N_sites, 2) @ (2,) → (N_sites,)
    beta_i = beta_samples[i]
    mu_aadt_i = X_aadt_scaled @ beta_i

    # Spatial effect: (N_sites, N_knots) @ (N_knots,) → (N_sites,)
    psi_i = psi_samples[i]
    Z_i = K_weighted @ psi_i

    # Total
    log_lambda_cf = log_years + beta0_samples[i] + mu_aadt_i + Z_i
    log_lambda_cf_samples.append(log_lambda_cf)

log_lambda_cf_samples = np.array(log_lambda_cf_samples)
lambda_cf_samples = np.exp(log_lambda_cf_samples)

# ----------------------------------------------------
# 6. SUMMARIZE + EXPORT RESULTS
# ----------------------------------------------------
lambda_cf_hdi = az.hdi(lambda_cf_samples, hdi_prob=0.95)

results_df = pd.DataFrame({
    'node_id': data_predict['Index'].values,
    'observed_crashes': data_predict['observed_crashes'].values,
    'mean_lambda_cf': lambda_cf_samples.mean(axis=0),
    'median_lambda_cf': np.median(lambda_cf_samples, axis=0),
    'var_lambda_cf':np.var(lambda_cf_samples, axis=0),
    'std_lambda_cf':np.std(lambda_cf_samples, axis=0),
    'hdi_2.5%_lambda_cf': lambda_cf_hdi[:, 0],
    'hdi_97.5%_lambda_cf': lambda_cf_hdi[:, 1]
})

output_file = os.path.join(OUTPUT_FOLDER, '7_EB-NPC_step2-2_predictions.csv')
results_df.to_csv(output_file, index=False)

print(f"\n✅ Counterfactual predictions saved to: {output_file}\n")








