import pandas as pd
import numpy as np
import statsmodels.api as sm
import os, shutil

# =====================================================
# CONTROL PARAMETERS
# =====================================================
LAMBDA_RANGES = [1, 5, 12]

# Significance threshold for NB alpha
ALPHA_P_THRESHOLD = 0.05


# =====================================================
# LOOP OVER SIMULATION RANGES
# =====================================================
for LAMBDA_RANGE in LAMBDA_RANGES:

    output_folder = f'5_EB-PG_forloop_R{LAMBDA_RANGE}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        try:
            for ea in os.listdir(output_folder):
                os.remove(os.path.join(output_folder, ea))
        except:
            for ea in os.listdir(output_folder):
                shutil.rmtree(os.path.join(output_folder, ea))

    filenames = []
    ranges = []
    overdispersion_alphas = []
    overdispersion_alphas_ps = []
    reduction_ratios = []
    sigs = []

    input_folder = f'4_simulated_crashes_R{LAMBDA_RANGE}'

    # =====================================================
    # FILE LOOP
    # =====================================================
    for file in os.listdir(input_folder):
        if ".csv" not in file:
            continue

        df = pd.read_csv(os.path.join(input_folder, file))
        rnge = float(file.split('_')[-2].replace('R',''))

        df_untreated = df[df["is_treated_flag"]==0].copy()
        df_treated = df[df["is_treated_flag"]==1].copy()

        # =====================================================
        # PREPARE DATA FOR SPF (10 years)
        # =====================================================
        YEARS_SPAN = 10
        COLLISION_COLS = [f"count_Y{y}" for y in range(1, 11)]

        df_untreated["total_crashes"] = df_untreated[COLLISION_COLS].sum(axis=1)
        df_untreated["maj_vol_avg"]   = df_untreated[[f"majorAADT_t_Y{y}" for y in range(1, 11)]].mean(axis=1)
        df_untreated["min_vol_avg"]   = df_untreated[[f"minorAADT_t_Y{y}" for y in range(1, 11)]].mean(axis=1)

        eps = 1e-6
        df_untreated["maj_vol_avg_log"] = np.log(df_untreated["maj_vol_avg"] + eps)
        df_untreated["min_vol_avg_log"] = np.log(df_untreated["min_vol_avg"] + eps)

        # SPF variables
        Y = df_untreated["total_crashes"]
        X = sm.add_constant(df_untreated[["maj_vol_avg_log", "min_vol_avg_log"]], prepend=False)
        log_exposure = np.log(YEARS_SPAN) * np.ones(len(df_untreated))

        # =====================================================
        # TRY FITTING NB FIRST
        # =====================================================
        try:
            nb_model = sm.NegativeBinomial(Y, X, offset=log_exposure)
            nb_results = nb_model.fit(maxiter=200, disp=False)
        except:
            nb_results = None

        use_nb = False
        alpha_hat = 0
        alpha_p = 1

        if nb_results is not None and "alpha" in nb_results.params.index:
            alpha_hat = nb_results.params["alpha"]
            alpha_p   = nb_results.pvalues["alpha"]

            # **Use NB only when alpha > 0 and statistically significant**
            if alpha_hat > 0 and alpha_p < ALPHA_P_THRESHOLD:
                use_nb = True

        # =====================================================
        # FIT POISSON IF NB IS NOT SIGNIFICANT
        # =====================================================
        if not use_nb:
            print(f"⚠ Using Poisson SPF for {file} (alpha not significant).")
            pois_model = sm.GLM(Y, X, offset=log_exposure, family=sm.families.Poisson())
            pois_results = pois_model.fit()
            model = pois_results
        else:
            print(f"✔ Using NB SPF for {file}. alpha={alpha_hat:.4f}, p={alpha_p:.4f}")
            model = nb_results

        # =====================================================
        # EB-PG: SPF PREDICTION FOR BEFORE/AFTER
        # =====================================================
        YEARS_BEFORE = 5
        YEARS_AFTER = 5

        BEFORE_COLS = [f"count_Y{y}" for y in range(1, YEARS_BEFORE + 1)]
        AFTER_COLS = [f"count_Y{y}" for y in range(YEARS_BEFORE + 1, 11)]

        df_treated["obs_before"] = df_treated[BEFORE_COLS].sum(axis=1)
        df_treated["obs_after"]  = df_treated[AFTER_COLS].sum(axis=1)

        # Predict year by year
        before_pred = []
        after_pred = []

        for y in range(1, YEARS_BEFORE+1):
            Xtmp = sm.add_constant(
                pd.DataFrame({
                    "maj_vol_avg_log": np.log(df_treated[f"majorAADT_t_Y{y}"] + eps),
                    "min_vol_avg_log": np.log(df_treated[f"minorAADT_t_Y{y}"] + eps)
                }),
                prepend=False
            )
            before_pred.append(model.predict(Xtmp, offset=np.zeros(len(df_treated))))

        for y in range(YEARS_BEFORE+1, 11):
            Xtmp = sm.add_constant(
                pd.DataFrame({
                    "maj_vol_avg_log": np.log(df_treated[f"majorAADT_t_Y{y}"] + eps),
                    "min_vol_avg_log": np.log(df_treated[f"minorAADT_t_Y{y}"] + eps)
                }),
                prepend=False
            )
            after_pred.append(model.predict(Xtmp, offset=np.zeros(len(df_treated))))

        df_treated["sp_before"] = np.vstack(before_pred).sum(axis=0)
        df_treated["sp_after"]  = np.vstack(after_pred).sum(axis=0)

        # =====================================================
        # EB STEP: WEIGHTING
        # =====================================================
        if use_nb:
            k = alpha_hat
            df_treated["w"] = 1 / (1 + k * df_treated["sp_before"])
        else:
            df_treated["w"] = 0  # Poisson has no overdispersion

        df_treated["EB_before"] = df_treated["w"] * df_treated["sp_before"] + \
                                  (1 - df_treated["w"]) * df_treated["obs_before"]

        # Adjustment factor
        df_treated["r"] = df_treated["sp_after"] / df_treated["sp_before"]
        df_treated["expected_after_no_trt"] = df_treated["EB_before"] * df_treated["r"]

        # =====================================================
        # FINAL EFFECT ESTIMATES
        # =====================================================
        total_obs_after = df_treated["obs_after"].sum()
        total_exp_after = df_treated["expected_after_no_trt"].sum()
        OR = total_obs_after / total_exp_after
        Safety = 1 - OR

        df_treated.to_csv(os.path.join(output_folder, file.replace('4_','5_',1)), index=False)

        filenames.append(file)
        ranges.append(rnge)
        overdispersion_alphas.append(alpha_hat)
        overdispersion_alphas_ps.append(alpha_p)
        reduction_ratios.append(Safety)
        sigs.append(None if np.isnan(Safety) else Safety)

    # End file loop

    df_out = pd.DataFrame({
        "filename": filenames,
        "range": ranges,
        "alpha": overdispersion_alphas,
        "alpha_pvalue": overdispersion_alphas_ps,
        "reduction_ratio": reduction_ratios,
        "significance": sigs
    })

    df_out.to_csv(f'5_EB-PG_summary_results_R{LAMBDA_RANGE}.csv', index=False)

