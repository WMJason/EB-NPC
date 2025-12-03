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

LAMBDA_RANGES = [1, 5, 12]

for LAMBDA_RANGE in LAMBDA_RANGES:

    output_folder = f'11_reduction_percentages-no_step2-1_R{LAMBDA_RANGE}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        try:
            for ea in os.listdir(output_folder):
                os.remove(output_folder + '/' + ea)
        except:
            for ea in os.listdir(output_folder):
                shutil.rmtree(output_folder + '/' + ea)


    input_folder = f'8_reduction_percentages-R{LAMBDA_RANGE}'
    for file in os.listdir(input_folder):
        if '.csv' in file:
            df = pd.read_csv(os.path.join(input_folder,file))
            df['overall_reduction_mean_no_step2-1'] = (df['mean_lambda_cf'].sum() - df['observed_crashes'].sum())/df['mean_lambda_cf'].sum()
            df['overall_reduction_median_no_step2-1'] = (df['median_lambda_cf'].sum() - df['observed_crashes'].sum())/df['median_lambda_cf'].sum()

            df.to_csv(os.path.join(output_folder,file), index=False)






























