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

##################

LAMBDA_RANGES = [1, 5, 12]

for LAMBDA_RANGE in LAMBDA_RANGES:
    
    EB_PG_folder = f'5_EB-PG_forloop_R{LAMBDA_RANGE}'
    EB_NPC_folder = f'11_reduction_percentages-no_step2-1_R{LAMBDA_RANGE}'
    cmf = 0.745

    df_to_exports = pd.DataFrame()
    ebpg_names = []
    ebpg_reductions = []
    ebpg_vars = []
    ebpg_reductions_maes = []

    ebnpc_names = []
    ebnpc_reductions = []
    ebnpc_vars = []
    ebnpc_reductions_maes = []

    ebnpc_reductions_obss = []
    ebnpc_reductions_maes_obss = []

    for file in os.listdir(EB_PG_folder):
        if '.csv' in file:

            df_ebpg = pd.read_csv(os.path.join(EB_PG_folder, file))
            ebpg_reduction = df_ebpg['Step 10: Safety Effectiveness'].mean()
            ebpg_var = df_ebpg['(30)Variance Term'].mean()
            ebpg_reduction_mae = abs(ebpg_reduction - (1-cmf))

            ebnpc_file = file.replace('5_','8_',1)
            df_ebnpc = pd.read_csv(os.path.join(EB_NPC_folder, ebnpc_file))
            ebnpc_reduction = df_ebnpc['overall_reduction_mean'].mean()
            ebnpc_var = df_ebnpc['var_lambda_cf'].mean()
            ebnpc_reduction_mae = abs(ebnpc_reduction - (1-cmf))

            ebnpc_reduction_obs = df_ebnpc['overall_reduction_mean_no_step2-1'].mean()
            ebnpc_reduction_mae_obs = abs(ebnpc_reduction_obs - (1-cmf))

            ebpg_names.append(file)
            ebpg_reductions.append(ebpg_reduction)
            ebpg_vars.append(ebpg_var)
            ebpg_reductions_maes.append(ebpg_reduction_mae)

            ebnpc_names.append(ebnpc_file)
            ebnpc_reductions.append(ebnpc_reduction)
            ebnpc_vars.append(ebnpc_var)
            ebnpc_reductions_maes.append(ebnpc_reduction_mae)

            ebnpc_reductions_obss.append(ebnpc_reduction_obs)
            ebnpc_reductions_maes_obss.append(ebnpc_reduction_mae_obs)


    ebpg_names.append('overall_EB-PG')
    ebpg_reductions.append(sum(ebpg_reductions)/len(ebpg_reductions))
    ebpg_vars.append(sum(ebpg_vars)/len(ebpg_vars))
    ebpg_reductions_maes.append(sum(ebpg_reductions_maes)/len(ebpg_reductions_maes))

    ebnpc_names.append('overall_EB-NPC')
    ebnpc_reductions.append(sum(ebnpc_reductions)/len(ebnpc_reductions))
    ebnpc_vars.append(sum(ebnpc_vars)/len(ebnpc_vars))
    ebnpc_reductions_maes.append(sum(ebnpc_reductions_maes)/len(ebnpc_reductions_maes))

    ebnpc_reductions_obss.append(sum(ebnpc_reductions_obss)/len(ebnpc_reductions_obss))
    ebnpc_reductions_maes_obss.append(sum(ebnpc_reductions_maes_obss)/len(ebnpc_reductions_maes_obss))

    df_to_exports['filename_EB-PG'] = ebpg_names
    df_to_exports['filename_EB-NPC'] = ebnpc_names

    df_to_exports['ebpg_reduction'] = ebpg_reductions
    df_to_exports['ebpg_cf_var'] = ebpg_vars
    df_to_exports['ebpg_reductions_mae'] = ebpg_reductions_maes

    df_to_exports['ebnpc_reduction'] = ebnpc_reductions
    df_to_exports['ebnpc_cf_var'] = ebnpc_vars
    df_to_exports['ebnpc_reductions_mae'] = ebnpc_reductions_maes

    df_to_exports['ebnpc_reduction_no_step2-1'] = ebnpc_reductions_obss
    df_to_exports['ebnpc_reductions_mae_no_step2-1'] = ebnpc_reductions_maes_obss

    df_to_exports.to_csv(f'12_summary_EB-PG_EB-NPC_R{LAMBDA_RANGE}.csv', index=False)


####filter out the NB-failed ones
for LAMBDA_RANGE in LAMBDA_RANGES:
    
    EB_PG_folder = f'5_EB-PG_forloop_R{LAMBDA_RANGE}'
    EB_NPC_folder = f'11_reduction_percentages-no_step2-1_R{LAMBDA_RANGE}'
    cmf = 0.745

    df_to_exports = pd.DataFrame()
    ebpg_names = []
    ebpg_reductions = []
    ebpg_vars = []
    ebpg_reductions_maes = []

    ebnpc_names = []
    ebnpc_reductions = []
    ebnpc_vars = []
    ebnpc_reductions_maes = []

    ebnpc_reductions_obss = []
    ebnpc_reductions_maes_obss = []

    for file in os.listdir(EB_PG_folder):
        if '.csv' in file:

            df_ebpg = pd.read_csv(os.path.join(EB_PG_folder, file))
            ebpg_reduction = df_ebpg['Step 10: Safety Effectiveness'].mean()
            ebpg_alpha_p = df_ebpg['(20)Overdispersion parameter k_pvalue'].values[0]
            if ebpg_alpha_p <= 0.05:
                ebpg_var = df_ebpg['(30)Variance Term'].mean()
                ebpg_reduction_mae = abs(ebpg_reduction - (1-cmf))

                ebnpc_file = file.replace('5_','8_',1)
                df_ebnpc = pd.read_csv(os.path.join(EB_NPC_folder, ebnpc_file))
                ebnpc_reduction = df_ebnpc['overall_reduction_mean'].mean()
                ebnpc_var = df_ebnpc['var_lambda_cf'].mean()
                ebnpc_reduction_mae = abs(ebnpc_reduction - (1-cmf))

                ebnpc_reduction_obs = df_ebnpc['overall_reduction_mean_no_step2-1'].mean()
                ebnpc_reduction_mae_obs = abs(ebnpc_reduction_obs - (1-cmf))

                ebpg_names.append(file)
                ebpg_reductions.append(ebpg_reduction)
                ebpg_vars.append(ebpg_var)
                ebpg_reductions_maes.append(ebpg_reduction_mae)

                ebnpc_names.append(ebnpc_file)
                ebnpc_reductions.append(ebnpc_reduction)
                ebnpc_vars.append(ebnpc_var)
                ebnpc_reductions_maes.append(ebnpc_reduction_mae)

                ebnpc_reductions_obss.append(ebnpc_reduction_obs)
                ebnpc_reductions_maes_obss.append(ebnpc_reduction_mae_obs)


    ebpg_names.append('overall_EB-PG')
    ebpg_reductions.append(sum(ebpg_reductions)/len(ebpg_reductions))
    ebpg_vars.append(sum(ebpg_vars)/len(ebpg_vars))
    ebpg_reductions_maes.append(sum(ebpg_reductions_maes)/len(ebpg_reductions_maes))

    ebnpc_names.append('overall_EB-NPC')
    ebnpc_reductions.append(sum(ebnpc_reductions)/len(ebnpc_reductions))
    ebnpc_vars.append(sum(ebnpc_vars)/len(ebnpc_vars))
    ebnpc_reductions_maes.append(sum(ebnpc_reductions_maes)/len(ebnpc_reductions_maes))

    ebnpc_reductions_obss.append(sum(ebnpc_reductions_obss)/len(ebnpc_reductions_obss))
    ebnpc_reductions_maes_obss.append(sum(ebnpc_reductions_maes_obss)/len(ebnpc_reductions_maes_obss))

    df_to_exports['filename_EB-PG'] = ebpg_names
    df_to_exports['filename_EB-NPC'] = ebnpc_names

    df_to_exports['ebpg_reduction'] = ebpg_reductions
    df_to_exports['ebpg_cf_var'] = ebpg_vars
    df_to_exports['ebpg_reductions_mae'] = ebpg_reductions_maes

    df_to_exports['ebnpc_reduction'] = ebnpc_reductions
    df_to_exports['ebnpc_cf_var'] = ebnpc_vars
    df_to_exports['ebnpc_reductions_mae'] = ebnpc_reductions_maes

    df_to_exports['ebnpc_reduction_no_step2-1'] = ebnpc_reductions_obss
    df_to_exports['ebnpc_reductions_mae_no_step2-1'] = ebnpc_reductions_maes_obss

    df_to_exports.to_csv(f'12_summary_EB-PG_EB-NPC_R{LAMBDA_RANGE}_NB-success.csv', index=False)

 

