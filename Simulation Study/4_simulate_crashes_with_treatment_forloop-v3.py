import math
import numpy as np
import pandas as pd
import ast
from typing import Dict, Tuple, Any, List
import os
import shutil # Import shutil for folder cleanup

# --- Helper Functions ---

def load_node_data(csv_filename: str) -> pd.DataFrame:
    """Loads node information (including AADT and X_speed)."""
    try:
        return pd.read_csv(csv_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Node data file not found: {csv_filename}")

def standardize_column(series: pd.Series) -> Tuple[pd.Series, float, float]:
    """Standardizes a Series (Z-score) and returns the mean and std dev."""
    mean = series.mean()
    # Handle near-zero standard deviation to prevent division by zero
    std = series.std() if series.std() > 1e-8 else 1.0 
    standardized_series = (series - mean) / std
    return standardized_series, mean, std

# --- REVISED: Wide Panel Reshaping Function (Includes Dual Mu) ---

def reshape_to_wide_panel(
    crashes_df: pd.DataFrame, 
    output_filename: str,
    base_year: int
) -> pd.DataFrame:
    """
    Reshapes the long-format simulated crash data into a wide-format panel, 
    including the counterfactual true mu (mu_simulated_cf_true).
    """
    
    # 1. Determine Time-Invariant (Node-Constant) Treatment Attributes
    
    agg_df = crashes_df.groupby('node_id').agg(
        # X_speed_original is constant across all years for a node
        X_speed_original=('X_speed_original', 'first'), 
        # is_treated_flag: 1 if 'treated' is ever 1 (max of the series)
        is_treated_flag=('treated', 'max'),
        # CMF_applied: 0.745 if treated (min of the series), 1.0 if control
        CMF_applied=('CMF_applied', 'min'),
        # Since speed is no longer manipulated, X_speed_simulated == X_speed_original
        X_speed_simulated=('X_speed_original', 'first')
    ).reset_index()
    
    # We do not need to merge/fill X_speed_simulated anymore as it is constant.
    wide_base_df = agg_df
    
    # 2. Pivot Time-Varying Columns
    time_varying_cols = [
        'majorAADT_t', 
        'minorAADT_t', 
        'count',
        'mu_simulated',           # Final mu (with CMF applied if treated)
        'mu_simulated_cf_true'    # Baseline mu (True Counterfactual)
    ]
    
    wide_df_parts = []
    for col in time_varying_cols:
        temp_pivot = crashes_df.pivot(
            index='node_id', 
            columns='year', 
            values=col
        )
        temp_pivot.columns = [f'{col}_Y{y}' for y in temp_pivot.columns]
        wide_df_parts.append(temp_pivot)
        
    # 3. Concatenate and Export
    wide_base_df = wide_base_df.set_index('node_id')
    
    final_wide_df = wide_base_df
    for part in wide_df_parts:
        final_wide_df = final_wide_df.join(part, how='left')

    final_wide_df = final_wide_df.reset_index()

    final_wide_df.to_csv(output_filename, index=False)
    print(f"\n✅ Exported WIDE panel data to {output_filename}")
    
    return final_wide_df


# --- REVISED: Core Simulation Function (Direct CMF & Dual Mu Storage) ---

def generate_node_crashes_direct_cmf(
    nodes_df: pd.DataFrame,
    beta_node: Dict[str, float],
    sigma_heterogeneity: float,
    years: List[int],
    treated_nodes_set: set,
    treat_start_year: int,
    cmf_factor: float, # The CMF value (e.g., 0.745)
    aadt_growth_major_rate: float,
    aadt_growth_minor_rate: float,
    seed: int = None,
    output_filename: str = "simulated_node_crashes_direct_cmf_long.csv"
) -> pd.DataFrame:
    """
    Simulates node-level yearly crash counts, applying the CMF directly to mu_t.
    Stores both the baseline mu (mu_t_baseline, the CF truth) and the final
    simulated mu (mu_t, the reduced value).
    """
    if seed is not None:
        np.random.seed(seed)
    
    nodes_rows = []
    
    # 1. Calculate and Standardize Covariates (Base Year Values)
    
    # Get Standardization parameters for X_speed
    nodes_df['X_speed_std'], x_mean, x_std = standardize_column(nodes_df['X_speed'])
    
    # Log Transform then Standardize AADT
    nodes_df['majorAADT_log_base'] = nodes_df['majorAADT'].apply(lambda x: math.log(x + 1e-8))
    nodes_df['minorAADT_log_base'] = nodes_df['minorAADT'].apply(lambda x: math.log(x + 1e-8))
    
    nodes_df['majorAADT_log_std_base'], major_log_mean, major_log_std = standardize_column(nodes_df['majorAADT_log_base'])
    nodes_df['minorAADT_log_std_base'], minor_log_mean, minor_log_std = standardize_column(nodes_df['minorAADT_log_base'])
    
    print(f"Standardization done: X_speed (mean={x_mean:.2f}, std={x_std:.2f})")
    print(f"Log(AADT) standardization done: Major (mean={major_log_mean:.2f}), Minor (mean={minor_log_mean:.2f})")
    
    # 2. Simulate Crashes
    for index, row in nodes_df.iterrows():
        node_id = row['node_id']
        major_base = row['majorAADT']
        minor_base = row['minorAADT']
        
        X_speed_original = row['X_speed']
        X_speed_std_original = row['X_speed_std']
        
        # Recalculate standardization factors (robustness check)
        major_log_mean = nodes_df['majorAADT_log_base'].mean()
        major_log_std = nodes_df['majorAADT_log_base'].std() if nodes_df['majorAADT_log_base'].std() > 1e-8 else 1.0
        minor_log_mean = nodes_df['minorAADT_log_base'].mean()
        minor_log_std = nodes_df['minorAADT_log_base'].std() if nodes_df['minorAADT_log_base'].std() > 1e-8 else 1.0
        
        epsilon = np.random.normal(loc=0.0, scale=sigma_heterogeneity)
        
        for year in years:
            
            # 3a. Calculate Time-Varying AADT and standardise
            year_index = year - 1 
            major_aadt_t = major_base * ((1 + aadt_growth_major_rate) ** year_index)
            minor_aadt_t = minor_base * ((1 + aadt_growth_minor_rate) ** year_index)
            
            major_aadt_t_log = math.log(major_aadt_t + 1e-8)
            minor_aadt_t_log = math.log(minor_aadt_t + 1e-8)
            
            major_aadt_t_log_std = (major_aadt_t_log - major_log_mean) / major_log_std
            minor_aadt_t_log_std = (minor_aadt_t_log - minor_log_mean) / minor_log_std
            
            
            # 3b. Determine X_speed (Always original speed)
            treated_flag = (node_id in treated_nodes_set) and (year >= treat_start_year)
            
            X_speed_std_t = X_speed_std_original
            X_speed_simulated = X_speed_original
            
            
            # 3c. Calculate baseline expected crash count (mu_t_baseline) - THE CF TRUTH
            log_mu_base = (
                beta_node['beta0'] 
                + beta_node['beta1'] * major_aadt_t_log_std
                + beta_node['beta2'] * minor_aadt_t_log_std
                + beta_node['beta3'] * X_speed_std_t
                #+ epsilon 
            )
            # Store the expected value WITHOUT CMF, this is the true counterfactual lambda (lambda_CF_true)
            mu_t_baseline = math.exp(log_mu_base)
            
            # 3d. Apply CMF factor directly to get the final simulated mean (mu_t)
            mu_t = mu_t_baseline * (cmf_factor if treated_flag else 1.0)
            
            # 3e. Simulate crash count
            # Using Poisson distribution for simplicity in the simulation
            count = int(np.random.poisson(mu_t)) 
            
            nodes_rows.append({
                'node_id': node_id, 
                'year': year, 
                'majorAADT_t': major_aadt_t,
                'minorAADT_t': minor_aadt_t,
                'X_speed_original': X_speed_original,    
                'X_speed_simulated': X_speed_simulated,
                'CMF_applied': cmf_factor if treated_flag else 1.0, 
                'mu_simulated_cf_true': mu_t_baseline, # True Counterfactual Mean
                'mu_simulated': mu_t,                 # Final Simulated Mean (Reduced)
                'treated': int(treated_flag),
                'count': count
            })

    # 4. Create and Export DataFrame (Long Format)
    nodes_crashes_df_long = pd.DataFrame(nodes_rows)
    nodes_crashes_df_long['node_id'] = nodes_crashes_df_long['node_id'].apply(lambda x: str(x))
    nodes_crashes_df_long.to_csv(output_filename, index=False)
    print(f"✅ Exported simulated node crashes (Direct CMF application) in LONG format to {output_filename}")
    return nodes_crashes_df_long

if __name__ == "__main__":

    LAMBDA_RANGES = [1, 5, 12]
    # Define beta3s based on LAMBDA_RANGES, as implied by your original code
    beta3s = {1:0.5, 5:0.5, 12:0.5} 

    for LAMBDA_RANGE in LAMBDA_RANGES:
        ##################
        output_folder = f'4_simulated_crashes_R{LAMBDA_RANGE}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            # Clean up output folder before running
            for item in os.listdir(output_folder):
                item_path = os.path.join(output_folder, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        input_folder = f'3_adding_speeding_info_forloop_R{LAMBDA_RANGE}'
        for file in os.listdir(input_folder):
            if '.csv' in file:
                # --- Configuration ---
                NODES_XSPEED_CSV_FILE = os.path.join(input_folder, file) 
                
                # Model parameters
                BETA_NODE_COEFFICIENTS = {
                    'beta0': -1.5,
                    'beta1': 0.6,  # Major AADT
                    'beta2': 0.3,  # Minor AADT
                    'beta3': beta3s[LAMBDA_RANGE] # X_speed (Unchanged by CMF in this version)
                }
                SIGMA_HETERO = 0.0 
                SIMULATION_YEARS = list(range(1, 11))
                SIMULATION_SEED = 42
                
                # OUTPUT FILES
                OUTPUT_LONG_CSV_FILE = '4_long_temp.csv'
                OUTPUT_WIDE_CSV_FILE = os.path.join(output_folder,file.replace('3_','4_wide',1))

                # --- AADT and Treatment Configuration ---
                AADT_GROWTH_MAJOR_RATE = 0.02
                AADT_GROWTH_MINOR_RATE = 0.015
                INTERSECTION_DEVICE_CMF = 0.745 
                TREATMENT_START_YEAR = 6
                
                # Determine treated nodes (top 10 by major AADT)
                try:
                    df_nodes_temp = load_node_data(NODES_XSPEED_CSV_FILE)
                    # Robust node_id parsing
                    df_nodes_temp['node_id'] = df_nodes_temp['node_id'].apply(lambda x: str(ast.literal_eval(x)) if isinstance(x, str) and x.startswith('(') else str(x))
                    treated_nodes_set = set(
                        df_nodes_temp
                        .sort_values(by='majorAADT', ascending=False)
                        .head(30)['node_id']
                    )

                    '''treated_nodes_set = set(
                        df_nodes_temp['node_id']
                        .sample(n=30, random_state=42) # Random sampling
                        .tolist()
                    )'''
                    print('treated sites ID:')
                    print(treated_nodes_set)
                except Exception as e:
                    print(f"Warning: Could not determine treated nodes. Using empty set. Error: {e}")
                    treated_nodes_set = set()

                print("--- Simulating Node Crashes: DIRECT CMF APPLICATION ---")
                
                try:
                    # 1. Load Data
                    df_nodes = load_node_data(NODES_XSPEED_CSV_FILE)
                    df_nodes['node_id'] = df_nodes['node_id'].apply(lambda x: str(ast.literal_eval(x)) if isinstance(x, str) and x.startswith('(') else str(x))

                    # 2. Run Simulation (Outputs LONG format)
                    crashes_df_long = generate_node_crashes_direct_cmf(
                        nodes_df=df_nodes,
                        beta_node=BETA_NODE_COEFFICIENTS,
                        sigma_heterogeneity=SIGMA_HETERO,
                        years=SIMULATION_YEARS,
                        treated_nodes_set=treated_nodes_set,
                        treat_start_year=TREATMENT_START_YEAR,
                        cmf_factor=INTERSECTION_DEVICE_CMF,
                        aadt_growth_major_rate=AADT_GROWTH_MAJOR_RATE,
                        aadt_growth_minor_rate=AADT_GROWTH_MINOR_RATE,
                        seed=SIMULATION_SEED,
                        output_filename=OUTPUT_LONG_CSV_FILE
                    )
                    
                    # 3. Reshape and Export WIDE Panel Data
                    reshape_to_wide_panel(
                        crashes_df_long, 
                        OUTPUT_WIDE_CSV_FILE,
                        base_year=SIMULATION_YEARS[0] 
                    )
                    
                    print(f"\nSimulation Summary: Total simulated crashes over {len(SIMULATION_YEARS)} years: {crashes_df_long['count'].sum()}")
                    
                except FileNotFoundError as e:
                    print(f"\n❌ ERROR: {e}")
                    print("Ensure all prerequisite files are correctly generated.")
                except Exception as e:
                    print(f"\n❌ An unexpected error occurred: {e}")
