import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import shutil
from shutil import copyfile

import re
import gzip
import json
import arviz as az
from tqdm import tqdm
import pickle

##################
output_folder = '8_reduction_percentages'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    try:
        for ea in os.listdir(output_folder):
            os.remove(output_folder + '/' + ea)
    except:
        for ea in os.listdir(output_folder):
            shutil.rmtree(output_folder + '/' + ea)


before_folder = '7_EB-NPC_step2-2'
after_folder = '7_EB-NPC_step2-1'

for file in os.listdir(before_folder):
    if 'predictions.csv' in file:
        df_before = pd.read_csv(before_folder+'/'+file)
        df_after = pd.read_csv(after_folder+'/'+file.replace('step2-2','step2-1'))

        to_add_cols = ['mean_lambda_post',
                       'median_lambda_post',
                       'hdi_2.5%_lambda_post',
                       'hdi_97.5%_lambda_post']
        df_before[to_add_cols] = df_after[to_add_cols].values

        df_before['reduction_mean'] = (df_before['mean_lambda_cf'].values - df_after['mean_lambda_post'].values)/df_before['mean_lambda_cf'].values
        df_before['reduction_median'] = (df_before['median_lambda_cf'].values - df_after['median_lambda_post'].values)/df_before['median_lambda_cf'].values
        
        df_before['overall_reduction_mean'] = (df_before['mean_lambda_cf'].sum() - df_after['mean_lambda_post'].sum())/df_before['mean_lambda_cf'].sum()
        df_before['overall_reduction_median'] = (df_before['median_lambda_cf'].sum() - df_after['median_lambda_post'].sum())/df_before['median_lambda_cf'].sum()
        
        df_before.to_csv(output_folder+'/'+file.replace('7_EB-NPC_step2-2_predictions','8_reduction_perc'))
































