import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from sklearn.preprocessing import StandardScaler
import math # Import math for log-transform
import os
import shutil
import json
import pickle
from scipy.stats import gamma

##################
def compute_metrics(true_cf, lb, ub, pred_mean, alpha=0.05):
    """
    true_cf: array-like true counterfactual mean (or true observed CF if you simulate counts)
    lb, ub, pred_mean: arrays from method
    Returns dict of metrics
    """
    true_cf = np.asarray(true_cf)
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    pred_mean = np.asarray(pred_mean)
    n = len(true_cf)

    # coverage
    covered = (lb <= true_cf) & (true_cf <= ub)
    coverage = covered.mean()

    # mean interval width
    widths = ub - lb
    miw = widths.mean()

    # bias and RMSE
    bias = np.mean(pred_mean - true_cf)
    mae = np.mean(np.abs(pred_mean - true_cf))
    rmse = np.sqrt(np.mean((pred_mean - true_cf)**2))

    # interval score components (alpha=0.05)
    alpha = alpha
    width_sum = widths.mean()
    penalty = np.zeros(n)
    below = true_cf < lb
    above = true_cf > ub
    penalty[below] = (2/alpha)*(lb[below] - true_cf[below])
    penalty[above] = (2/alpha)*(true_cf[above] - ub[above])
    iscore = widths + penalty
    mean_iscore = iscore.mean()

    # fraction of IS due to penalty (misses)
    mean_penalty = penalty.mean()
    frac_penalty = mean_penalty / mean_iscore if mean_iscore>0 else np.nan

    return {
        "n": n,
        "coverage": coverage,
        "mean_interval_width": miw,
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "mean_is": mean_iscore,
        "mean_penalty": mean_penalty,
        "frac_penalty": frac_penalty
    }



LAMBDA_RANGES = [1, 5, 12]
post_years = list(range(6,10+1))

for LAMBDA_RANGE in LAMBDA_RANGES:

    EB_PG_folder = f'5_EB-PG_forloop_R{LAMBDA_RANGE}'
    EB_NPC_folder = f'11_reduction_percentages-no_step2-1_R{LAMBDA_RANGE}'
    cmf = 0.745

    df_to_exports = pd.DataFrame()
    
    ebpg_names = []
    ebpg_sites = []
    ebpg_mus = []
    ebpg_cf_mus = []
    ebpg_obss = []
    ebpg_lbs = []
    ebpg_ubs = []
    ebpg_means = []
    ebpg_vars = []
    ebpg_iss = [] #interval score

    ebnpc_names = []
    ebnpc_sites = []
    ebnpc_mus = []
    ebnpc_cf_mus = []
    ebnpc_obss = []
    ebnpc_lbs = []
    ebnpc_ubs = []
    ebnpc_means = []
    ebnpc_vars = []
    ebnpc_iss = [] #interval score


    for file in os.listdir(EB_PG_folder):
        if '.csv' in file:

            df_ebpg = pd.read_csv(os.path.join(EB_PG_folder, file))
            
            ea_ebpg_sites = df_ebpg["node_id"].values.tolist()

            ea_ebpg_mus = df_ebpg[["mu_simulated_Y"+str(year) for year in post_years]].sum(axis=1).values.tolist()
                
            ea_ebpg_cf_mus = df_ebpg[["mu_simulated_cf_true_Y"+str(year) for year in post_years]].sum(axis=1).values.tolist()

            ea_ebpg_obss = df_ebpg[["count_Y"+str(year) for year in post_years]].sum(axis=1).values.tolist()

            df_ebpg['gamma_alpha'] = (df_ebpg['(27)Expected average crash frequency in after period without treatment']**2)/df_ebpg['(30)Variance Term']
            df_ebpg['gamma_scale'] = df_ebpg['(30)Variance Term'] / df_ebpg['(27)Expected average crash frequency in after period without treatment']
            
            lbs = []
            ubs = []
            for i in range(len(df_ebpg)):
                # Define the parameters
                a_param = df_ebpg['gamma_alpha'].iloc[i]  # Example shape parameter 'a'
                scale_param = df_ebpg['gamma_scale'].iloc[i] # Scale (theta)
                
                # Calculate the quantiles
                lower_quantile = gamma.ppf(0.025, a=a_param, scale=scale_param)
                upper_quantile = gamma.ppf(0.975, a=a_param, scale=scale_param)
                
                lbs.append(lower_quantile)
                ubs.append(upper_quantile)
            df_ebpg['LB'] = lbs
            df_ebpg['UB'] = ubs
            ea_ebpg_lbs = df_ebpg['LB'].values.tolist()
            ea_ebpg_ubs = df_ebpg['UB'].values.tolist()

            ea_ebpg_means = df_ebpg['(27)Expected average crash frequency in after period without treatment'].values.tolist()

            ea_ebpg_vars = df_ebpg['(30)Variance Term'].values.tolist()

            ea_ebpg_iss = []
            for i in range(len(df_ebpg)):
                cf_lb = lbs[i]
                cf_ub = ubs[i]
                cf_mu = ea_ebpg_cf_mus[i]

                if cf_lb <= cf_mu <= cf_ub:
                    interval_score = cf_ub - cf_lb
                elif cf_mu < cf_lb:
                    interval_score = (cf_ub - cf_lb) + 40 * (cf_lb - cf_mu)
                elif cf_mu > cf_ub:
                    interval_score = (cf_ub - cf_lb) + 40 * (cf_mu - cf_ub)

                ea_ebpg_iss.append(interval_score)
        
            ebpg_names += ([file]*len(df_ebpg))
            ebpg_sites += ea_ebpg_sites
            ebpg_mus += ea_ebpg_mus
            ebpg_cf_mus += ea_ebpg_cf_mus
            ebpg_obss += ea_ebpg_obss
            ebpg_lbs += ea_ebpg_lbs
            ebpg_ubs += ea_ebpg_ubs
            ebpg_means += ea_ebpg_means
            ebpg_vars += ea_ebpg_vars
            ebpg_iss += ea_ebpg_iss
            

            ebnpc_file = file.replace('5_','8_',1)
            df_ebnpc = pd.read_csv(os.path.join(EB_NPC_folder, ebnpc_file))

            ea_ebnpc_sites = df_ebnpc["node_id"].values.tolist()

            ea_ebnpc_mus = ea_ebpg_mus
                
            ea_ebnpc_cf_mus = ea_ebpg_cf_mus

            ea_ebnpc_obss = df_ebnpc["observed_crashes"].values.tolist()

            ea_ebnpc_lbs = df_ebnpc['hdi_2.5%_lambda_cf'].values.tolist()
            ea_ebnpc_ubs = df_ebnpc['hdi_97.5%_lambda_cf'].values.tolist()

            ea_ebnpc_means = df_ebnpc['mean_lambda_cf'].values.tolist()

            ea_ebnpc_vars = df_ebnpc['var_lambda_cf'].values.tolist()

            ea_ebnpc_iss = []
            for i in range(len(df_ebnpc)):
                cf_lb = ea_ebnpc_lbs[i]
                cf_ub = ea_ebnpc_ubs[i]
                cf_mu = ea_ebnpc_cf_mus[i]

                if cf_lb <= cf_mu <= cf_ub:
                    interval_score = cf_ub - cf_lb
                elif cf_mu < cf_lb:
                    interval_score = (cf_ub - cf_lb) + 40 * (cf_lb - cf_mu)
                elif cf_mu > cf_ub:
                    interval_score = (cf_ub - cf_lb) + 40 * (cf_mu - cf_ub)

                ea_ebnpc_iss.append(interval_score)
        
            ebnpc_names += ([file]*len(df_ebnpc))
            ebnpc_sites += ea_ebnpc_sites
            ebnpc_mus += ea_ebnpc_mus
            ebnpc_cf_mus += ea_ebnpc_cf_mus
            ebnpc_obss += ea_ebnpc_obss
            ebnpc_lbs += ea_ebnpc_lbs
            ebnpc_ubs += ea_ebnpc_ubs
            ebnpc_means += ea_ebnpc_means
            ebnpc_vars += ea_ebnpc_vars
            ebnpc_iss += ea_ebnpc_iss


    results_ebpg = compute_metrics(ebpg_cf_mus, ebpg_lbs, ebpg_ubs, ebpg_means)
    print('')
    print(f'results of EB-PG (R={LAMBDA_RANGE}):')
    print(results_ebpg)

    results_ebnpc = compute_metrics(ebnpc_cf_mus, ebnpc_lbs, ebnpc_ubs, ebnpc_means)
    print('')
    print(f'results of EB-NPC (R={LAMBDA_RANGE}):')
    print(results_ebnpc)


    ebpg_names.append('overall_EB-PG')
    ebpg_sites.append('overall')
    ebpg_mus.append('NA')
    ebpg_cf_mus.append('NA')
    ebpg_obss.append('NA')
    ebpg_lbs.append('NA')
    ebpg_ubs.append('NA')
    ebpg_means.append('NA')
    ebpg_vars.append('NA')
    ebpg_iss.append(sum(ebpg_iss)/len(ebpg_iss))
    
    ebnpc_names.append('overall_EB-NPC')
    ebnpc_sites.append('overall')
    ebnpc_mus.append('NA')
    ebnpc_cf_mus.append('NA')
    ebnpc_obss.append('NA')
    ebnpc_lbs.append('NA')
    ebnpc_ubs.append('NA')
    ebnpc_means.append('NA')
    ebnpc_vars.append('NA')
    ebnpc_iss.append(sum(ebnpc_iss)/len(ebnpc_iss))

    df_to_exports['filename_EB-PG'] = ebpg_names
    df_to_exports['site_id_EB-PG'] = ebpg_sites
    df_to_exports['EB-PG_after_mu'] = ebpg_mus
    df_to_exports['EB-PG_cf_mu'] = ebpg_cf_mus
    df_to_exports['EB-PG_after_obss'] = ebpg_obss
    df_to_exports['EB-PG_cf_lb'] = ebpg_lbs
    df_to_exports['EB-PG_cf_ub'] = ebpg_ubs
    df_to_exports['EB-PG_cf_mean'] = ebpg_means
    df_to_exports['EB-PG_cf_var'] = ebpg_vars
    df_to_exports['EB-PG_interval_score'] = ebpg_iss


    df_to_exports['filename_EB-NPC'] = ebnpc_names
    df_to_exports['site_id_EB-NPC'] = ebnpc_sites
    df_to_exports['EB-NPC_after_mu'] = ebnpc_mus
    df_to_exports['EB-NPC_cf_mu'] = ebnpc_cf_mus
    df_to_exports['EB-NPC_after_obss'] = ebnpc_obss
    df_to_exports['EB-NPC_cf_lb'] = ebnpc_lbs
    df_to_exports['EB-NPC_cf_ub'] = ebnpc_ubs
    df_to_exports['EB-NPC_cf_mean'] = ebnpc_means
    df_to_exports['EB-NPC_cf_var'] = ebnpc_vars
    df_to_exports['EB-NPC_interval_score'] = ebnpc_iss

    df_to_exports.to_csv(f'13_interval_scores_EB-PG_EB-NPC_R{LAMBDA_RANGE}.csv', index=False)



#####filter those NB-failed cases
print('')
print('Filtered those NB-failed cases:')
for LAMBDA_RANGE in LAMBDA_RANGES:

    EB_PG_folder = f'5_EB-PG_forloop_R{LAMBDA_RANGE}'
    EB_NPC_folder = f'11_reduction_percentages-no_step2-1_R{LAMBDA_RANGE}'
    cmf = 0.745

    df_to_exports = pd.DataFrame()
    
    ebpg_names = []
    ebpg_sites = []
    ebpg_mus = []
    ebpg_cf_mus = []
    ebpg_obss = []
    ebpg_lbs = []
    ebpg_ubs = []
    ebpg_means = []
    ebpg_vars = []
    ebpg_iss = [] #interval score

    ebnpc_names = []
    ebnpc_sites = []
    ebnpc_mus = []
    ebnpc_cf_mus = []
    ebnpc_obss = []
    ebnpc_lbs = []
    ebnpc_ubs = []
    ebnpc_means = []
    ebnpc_vars = []
    ebnpc_iss = [] #interval score


    for file in os.listdir(EB_PG_folder):
        if '.csv' in file:

            df_ebpg = pd.read_csv(os.path.join(EB_PG_folder, file))

            alpha_p = df_ebpg['(20)Overdispersion parameter k_pvalue'].values[0]
            if alpha_p <= 0.05:
            
                ea_ebpg_sites = df_ebpg["node_id"].values.tolist()

                ea_ebpg_mus = df_ebpg[["mu_simulated_Y"+str(year) for year in post_years]].sum(axis=1).values.tolist()
                    
                ea_ebpg_cf_mus = df_ebpg[["mu_simulated_cf_true_Y"+str(year) for year in post_years]].sum(axis=1).values.tolist()

                ea_ebpg_obss = df_ebpg[["count_Y"+str(year) for year in post_years]].sum(axis=1).values.tolist()

                df_ebpg['gamma_alpha'] = (df_ebpg['(27)Expected average crash frequency in after period without treatment']**2)/df_ebpg['(30)Variance Term']
                df_ebpg['gamma_scale'] = df_ebpg['(30)Variance Term'] / df_ebpg['(27)Expected average crash frequency in after period without treatment']

                lbs = []
                ubs = []
                for i in range(len(df_ebpg)):
                    # Define the parameters
                    a_param = df_ebpg['gamma_alpha'].iloc[i]  # Example shape parameter 'a'
                    scale_param = df_ebpg['gamma_scale'].iloc[i] # Scale (theta)
                    
                    # Calculate the quantiles
                    lower_quantile = gamma.ppf(0.025, a=a_param, scale=scale_param)
                    upper_quantile = gamma.ppf(0.975, a=a_param, scale=scale_param)
                    
                    lbs.append(lower_quantile)
                    ubs.append(upper_quantile)
                df_ebpg['LB'] = lbs
                df_ebpg['UB'] = ubs
                ea_ebpg_lbs = df_ebpg['LB'].values.tolist()
                ea_ebpg_ubs = df_ebpg['UB'].values.tolist()

                ea_ebpg_means = df_ebpg['(27)Expected average crash frequency in after period without treatment'].values.tolist()

                ea_ebpg_vars = df_ebpg['(30)Variance Term'].values.tolist()

                ea_ebpg_iss = []
                for i in range(len(df_ebpg)):
                    cf_lb = lbs[i]
                    cf_ub = ubs[i]
                    cf_mu = ea_ebpg_cf_mus[i]

                    if cf_lb <= cf_mu <= cf_ub:
                        interval_score = cf_ub - cf_lb
                    elif cf_mu < cf_lb:
                        interval_score = (cf_ub - cf_lb) + 40 * (cf_lb - cf_mu)
                    elif cf_mu > cf_ub:
                        interval_score = (cf_ub - cf_lb) + 40 * (cf_mu - cf_ub)

                    ea_ebpg_iss.append(interval_score)
            
                ebpg_names += ([file]*len(df_ebpg))
                ebpg_sites += ea_ebpg_sites
                ebpg_mus += ea_ebpg_mus
                ebpg_cf_mus += ea_ebpg_cf_mus
                ebpg_obss += ea_ebpg_obss
                ebpg_lbs += ea_ebpg_lbs
                ebpg_ubs += ea_ebpg_ubs
                ebpg_means += ea_ebpg_means
                ebpg_vars += ea_ebpg_vars
                ebpg_iss += ea_ebpg_iss
                

                ebnpc_file = file.replace('5_','8_',1)
                df_ebnpc = pd.read_csv(os.path.join(EB_NPC_folder, ebnpc_file))

                ea_ebnpc_sites = df_ebnpc["node_id"].values.tolist()

                ea_ebnpc_mus = ea_ebpg_mus
                    
                ea_ebnpc_cf_mus = ea_ebpg_cf_mus

                ea_ebnpc_obss = df_ebnpc["observed_crashes"].values.tolist()

                ea_ebnpc_lbs = df_ebnpc['hdi_2.5%_lambda_cf'].values.tolist()
                ea_ebnpc_ubs = df_ebnpc['hdi_97.5%_lambda_cf'].values.tolist()

                ea_ebnpc_means = df_ebnpc['mean_lambda_cf'].values.tolist()

                ea_ebnpc_vars = df_ebnpc['var_lambda_cf'].values.tolist()

                ea_ebnpc_iss = []
                for i in range(len(df_ebnpc)):
                    cf_lb = ea_ebnpc_lbs[i]
                    cf_ub = ea_ebnpc_ubs[i]
                    cf_mu = ea_ebnpc_cf_mus[i]

                    if cf_lb <= cf_mu <= cf_ub:
                        interval_score = cf_ub - cf_lb
                    elif cf_mu < cf_lb:
                        interval_score = (cf_ub - cf_lb) + 40 * (cf_lb - cf_mu)
                    elif cf_mu > cf_ub:
                        interval_score = (cf_ub - cf_lb) + 40 * (cf_mu - cf_ub)

                    ea_ebnpc_iss.append(interval_score)
            
                ebnpc_names += ([file]*len(df_ebnpc))
                ebnpc_sites += ea_ebnpc_sites
                ebnpc_mus += ea_ebnpc_mus
                ebnpc_cf_mus += ea_ebnpc_cf_mus
                ebnpc_obss += ea_ebnpc_obss
                ebnpc_lbs += ea_ebnpc_lbs
                ebnpc_ubs += ea_ebnpc_ubs
                ebnpc_means += ea_ebnpc_means
                ebnpc_vars += ea_ebnpc_vars
                ebnpc_iss += ea_ebnpc_iss


    results_ebpg = compute_metrics(ebpg_cf_mus, ebpg_lbs, ebpg_ubs, ebpg_means)
    print('')
    print(f'results of EB-PG (R={LAMBDA_RANGE}):')
    print(results_ebpg)

    results_ebnpc = compute_metrics(ebnpc_cf_mus, ebnpc_lbs, ebnpc_ubs, ebnpc_means)
    print('')
    print(f'results of EB-NPC (R={LAMBDA_RANGE}):')
    print(results_ebnpc)


    ebpg_names.append('overall_EB-PG')
    ebpg_sites.append('overall')
    ebpg_mus.append('NA')
    ebpg_cf_mus.append('NA')
    ebpg_obss.append('NA')
    ebpg_lbs.append('NA')
    ebpg_ubs.append('NA')
    ebpg_means.append('NA')
    ebpg_vars.append('NA')
    ebpg_iss.append(sum(ebpg_iss)/len(ebpg_iss))
    
    ebnpc_names.append('overall_EB-NPC')
    ebnpc_sites.append('overall')
    ebnpc_mus.append('NA')
    ebnpc_cf_mus.append('NA')
    ebnpc_obss.append('NA')
    ebnpc_lbs.append('NA')
    ebnpc_ubs.append('NA')
    ebnpc_means.append('NA')
    ebnpc_vars.append('NA')
    ebnpc_iss.append(sum(ebnpc_iss)/len(ebnpc_iss))

    df_to_exports['filename_EB-PG'] = ebpg_names
    df_to_exports['site_id_EB-PG'] = ebpg_sites
    df_to_exports['EB-PG_after_mu'] = ebpg_mus
    df_to_exports['EB-PG_cf_mu'] = ebpg_cf_mus
    df_to_exports['EB-PG_after_obss'] = ebpg_obss
    df_to_exports['EB-PG_cf_lb'] = ebpg_lbs
    df_to_exports['EB-PG_cf_ub'] = ebpg_ubs
    df_to_exports['EB-PG_cf_mean'] = ebpg_means
    df_to_exports['EB-PG_cf_var'] = ebpg_vars
    df_to_exports['EB-PG_interval_score'] = ebpg_iss


    df_to_exports['filename_EB-NPC'] = ebnpc_names
    df_to_exports['site_id_EB-NPC'] = ebnpc_sites
    df_to_exports['EB-NPC_after_mu'] = ebnpc_mus
    df_to_exports['EB-NPC_cf_mu'] = ebnpc_cf_mus
    df_to_exports['EB-NPC_after_obss'] = ebnpc_obss
    df_to_exports['EB-NPC_cf_lb'] = ebnpc_lbs
    df_to_exports['EB-NPC_cf_ub'] = ebnpc_ubs
    df_to_exports['EB-NPC_cf_mean'] = ebnpc_means
    df_to_exports['EB-NPC_cf_var'] = ebnpc_vars
    df_to_exports['EB-NPC_interval_score'] = ebnpc_iss

    df_to_exports.to_csv(f'13_interval_scores_EB-PG_EB-NPC_R{LAMBDA_RANGE}_NB_success.csv', index=False)


 

