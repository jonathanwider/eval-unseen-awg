import os


configfile: "configs/paths.yaml"
configfile: "configs/evaluation.yaml"


dir_code_core = config["paths"]["dir_code_core"]


# use one of the weather generator configs defined in the main unseen-awg repo.
configfile: os.path.join(dir_code_core, "configs/weather_generators/reforecasts.yaml")  # main options: reforecasts, reforecasts_no_bias_correction, era5


include: os.path.join(dir_code_core, "workflow/Snakefile")  # Inlcude snakefile from main repository.

dir_eval = config["paths"]["dir_code_eval"]

CLIM_IMPACT_VARIABLES_GT = expand(
    os.path.join(
        config["paths"]["dir_preprocessed_datasets"],
        "climatology_preprocessed_impact_variables_era5/combined_{hash_preprocess_impact_variables_era5}_"
        + "{hash_compute_climatology}.nc",
    ),
    hash_compute_climatology=get_params_hash("compute_climatology"),
    hash_preprocess_impact_variables_era5=get_params_hash(
        "preprocess_impact_variables_era5"
    ),
)[0]
IMPACT_VARIABLES_GT = expand(
    os.path.join(
        get_preprocessed_dir("impact_variables", "era5"),
        "combined_{hash_preprocess_impact_variables_era5}.zarr",
    ),
    hash_preprocess_impact_variables_era5=get_params_hash(
        "preprocess_impact_variables_era5"
    ),
)[0]
IMPACT_VARIABLES_GT_RECHUNK = expand(
    os.path.join(
        get_preprocessed_dir("impact_variables", "era5"),
        "rechunk_combined_{hash_preprocess_impact_variables_era5}.zarr",
    ),
    hash_preprocess_impact_variables_era5=get_params_hash(
        "preprocess_impact_variables_era5"
    ),
)[0]

CLIM_IMPACT_VARIABLES_WG = (
    IMPACT_VARIABLES_WG.replace("rechunk_", "")
    .replace("preprocessed_", "climatology_preprocessed_")
    .replace(".zarr", f"_{get_params_hash("compute_climatology")}.nc")
)

TRAJECTORIES_CLIMATOLOGY = []
BIAS_FOR_CONSISTENCY_TEST = []
if DS_TYPE == "reforecasts":
    TRAJECTORIES_CLIMATOLOGY = (
        (
            expand(
                os.path.join(
                    config["paths"]["dir_simulations"],
                    "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
                    "{seed}_{sigma}_{blocksize}/trajectory.nc",
                ),
                merge_ds_type=config["dataset_type"],
                hash_combine_circulation=get_params_hash(
                    f"preprocess_circulation_{DS_TYPE}"
                ),
                hash_merge_restructure_ds=get_params_hash("merge_restructure_ds"),
                hash_wg=get_params_hash("weather_generator"),
                hash_traj=get_params_hash("simulate_trajectory"),
                seed=np.arange(config["trajectories_climatology"]["n_seeds"]),
                sigma=config["trajectories_climatology"]["sigmas"],
                blocksize=config["trajectories_climatology"]["blocksizes"],
            ),
        ),
    )
    if config["use_bias_corrected"]:
        BIAS_FOR_CONSISTENCY_TEST = expand(
            os.path.join(
                os.path.join(
                    get_preprocessed_dir("impact_variables", "reforecasts"),
                    "bias_{hash_era5}_{hash_re}_{hash_bias}",
                ),
                "era5_mode_{split_mode}_n_{n_partitions}.zarr",
            ),
            hash_era5=get_params_hash("preprocess_impact_variables_era5"),
            hash_re=get_params_hash("preprocess_impact_variables_reforecasts"),
            hash_bias=get_params_hash("bias_robustness"),
            split_mode=["chronological", "random-years", "random"],
            n_partitions=2,
        )


rule all_evaluations:
    input:
        bias=BIAS_FOR_CONSISTENCY_TEST,
        trajectories_climatology=TRAJECTORIES_CLIMATOLOGY,
        tuned_wg=expand(
            os.path.join(
                config["paths"]["dir_wgs"],
                "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_tune",
                    "{seed}_{n_analogs}_{forecast_lead_time_days}_optuna_study.pkl",
                ),
                merge_ds_type=DS_TYPE,
                hash_combine_circulation=get_params_hash(
                f"preprocess_circulation_{DS_TYPE}"
            ),
            hash_merge_restructure_ds=get_params_hash("merge_restructure_ds"),
            hash_wg=get_params_hash("weather_generator"),
            seed=0,
            n_analogs=80,
            forecast_lead_time_days=[3, 4],
        ),
        autocorrelation=expand(
            os.path.join(
                config["paths"]["dir_results"],
                "autocorrelation",
                "params_impact_variables_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}_"
                + hashlib.md5(
                    (
                        f"{CLIM_IMPACT_VARIABLES_GT}_{CLIM_IMPACT_VARIABLES_WG}_{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}"
                    ).encode()
                ).hexdigest()[:8]
                + "_{hash_autocorrelation}.yaml",
            ),
            hash_combine_circulation=get_params_hash(
                f"preprocess_circulation_{DS_TYPE}"
            ),
            hash_merge_restructure_ds=get_params_hash("merge_restructure_ds"),
            hash_wg=get_params_hash("weather_generator"),
            hash_traj=get_params_hash("simulate_trajectory"),
            hash_autocorrelation=get_params_hash("eval_autocorrelation"),
        ),
        eval_hot_day_connected_components=expand(
            os.path.join(
                config["paths"]["dir_results"],
                "hot_day_connected_components",
                "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
                "{use_lsm}_"
                + hashlib.md5(
                    (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
                ).hexdigest()[:8]
                + "_{hash_hot_connected}",
                "params.yaml",
            ),
            hash_combine_circulation=get_params_hash(
                f"preprocess_circulation_{DS_TYPE}"
            ),
            hash_merge_restructure_ds=get_params_hash("merge_restructure_ds"),
            hash_wg=get_params_hash("weather_generator"),
            hash_traj=get_params_hash("simulate_trajectory"),
            use_lsm=[True, False],
            hash_hot_connected=get_params_hash("eval_hot_day_connected_components"),
        ),
        eval_quantile_maps=expand(
            os.path.join(
                config["paths"]["dir_results"],
                "quantile_maps",
                "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
                "{months}_"
                + hashlib.md5(
                    ("{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
                ).hexdigest()[:8]
                + "_{hash_quantile_maps}",
                "params.yaml",
            ),
            hash_combine_circulation=get_params_hash(
                f"preprocess_circulation_{DS_TYPE}"
            ),
            hash_merge_restructure_ds=get_params_hash("merge_restructure_ds"),
            hash_wg=get_params_hash("weather_generator"),
            hash_traj=get_params_hash("simulate_trajectory"),
            hash_quantile_maps=get_params_hash("eval_quantile_maps"),
            months=["DJF", "MAM", "JJA", "SON"],
        ),
        eval_seasonal_correlation=expand(
            os.path.join(
                config["paths"]["dir_results"],
                "seasonal_correlation",
                    "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
                    hashlib.md5(
                        (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
                    ).hexdigest()[:8]
                    + "_{hash_season_corr}",
                    "params.yaml",
                ),
                hash_combine_circulation=get_params_hash(
                f"preprocess_circulation_{DS_TYPE}"
            ),
            hash_merge_restructure_ds=get_params_hash("merge_restructure_ds"),
            hash_wg=get_params_hash("weather_generator"),
            hash_traj=get_params_hash("simulate_trajectory"),
            hash_season_corr=get_params_hash("eval_seasonal_correlation"),
        ),
        eval_rolling_droughts=expand(
            os.path.join(
                config["paths"]["dir_results"],
                "rolling_droughts",
                "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
                "{use_anomalies}_{n_months_rolling}_"
                + hashlib.md5(
                    (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
                ).hexdigest()[:8]
                + "_{hash_rolling}",
                "params.yaml",
            ),
            hash_combine_circulation=get_params_hash(
                f"preprocess_circulation_{DS_TYPE}"
            ),
            hash_merge_restructure_ds=get_params_hash("merge_restructure_ds"),
            hash_wg=get_params_hash("weather_generator"),
            hash_traj=get_params_hash("simulate_trajectory"),
            use_anomalies=[True, False],
            n_months_rolling=[6, 12],
            hash_rolling=get_params_hash("eval_rolling_droughts"),
        ),


rule eval_autocorrelation:
    input:
        nc_era5_climatology=CLIM_IMPACT_VARIABLES_GT,
        nc_wg_climatology=CLIM_IMPACT_VARIABLES_WG,
        zarr_rechunk_era5=IMPACT_VARIABLES_GT_RECHUNK,
        zarr_rechunk_wg=IMPACT_VARIABLES_WG_RECHUNK,
        trajectories=expand(
            os.path.join(
                config["paths"]["dir_simulations"],
                "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
                "{seed}_{sigma}_{blocksize}/trajectory.nc",
            ),
            merge_ds_type=config["dataset_type"],
            seed=np.arange(config["eval_autocorrelation"]["n_seeds"]),
            blocksize=config["eval_autocorrelation"]["subset_tau"],
            sigma=config["sigma"],
            allow_missing=True,
        ),
    output:
        params=os.path.join(
            config["paths"]["dir_results"],
            "autocorrelation",
            "params_impact_variables_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}_"
            + hashlib.md5(
                (
                    f"{CLIM_IMPACT_VARIABLES_GT}_{CLIM_IMPACT_VARIABLES_WG}_{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}"
                ).encode()
            ).hexdigest()[:8]
            + "_{hash_autocorrelation}.yaml",
        ),
        images=expand(
            os.path.join(
                config["paths"]["dir_images"],
                "autocorrelation",
                "impact_variables_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}_"
                + hashlib.md5(
                    (
                        f"{CLIM_IMPACT_VARIABLES_GT}_{CLIM_IMPACT_VARIABLES_WG}_{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}"
                    ).encode()
                ).hexdigest()[:8]
                + "_{hash_autocorrelation}_{ss_tau}.png",
            ),
            ss_tau=[
                "_".join(
                    map(str, config["eval_autocorrelation"]["subset_tau"][: i + 1])
                )
                for i in range(len(config["eval_autocorrelation"]["subset_tau"]))
            ],
            allow_missing=True,
        ),
        autocorrelation_wg=os.path.join(
            config["paths"]["dir_results"],
            "autocorrelation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            hashlib.md5(
                    (
                        f"{CLIM_IMPACT_VARIABLES_GT}_{CLIM_IMPACT_VARIABLES_WG}_{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}"
                    ).encode()
                ).hexdigest()[:8]
                + "_{hash_autocorrelation}_wg.nc",
        ),
        autocorrelation_gt=os.path.join(
            config["paths"]["dir_results"],
            "autocorrelation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            hashlib.md5(
                    (
                        f"{CLIM_IMPACT_VARIABLES_GT}_{CLIM_IMPACT_VARIABLES_WG}_{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}"
                    ).encode()
                ).hexdigest()[:8]
                + "_{hash_autocorrelation}_gt.nc",
        ),
    params:
        all_params=get_all_params_dict("eval_autocorrelation"),
        tracked_params=get_params_dict_for_saving("eval_autocorrelation"),
    log:
        stdout=os.path.join(
            config["paths"]["dir_results"],
            "autocorrelation",
            "logs/log_impact_variables_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}_",
            hashlib.md5(
                (
                    f"{CLIM_IMPACT_VARIABLES_GT}_{CLIM_IMPACT_VARIABLES_WG}_{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}"
                ).encode()
            ).hexdigest()[:8]
            + "_{hash_autocorrelation}.out",
        ),
        stderr=os.path.join(
            config["paths"]["dir_results"],
            "autocorrelation",
            "logs/log_impact_variables_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}_",
            hashlib.md5(
                (
                    f"{CLIM_IMPACT_VARIABLES_GT}_{CLIM_IMPACT_VARIABLES_WG}_{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}"
                ).encode()
            ).hexdigest()[:8]
            + "_{hash_autocorrelation}.err",
        ),
    resources:
        runtime=60,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=1,
    script:
        f"{dir_eval}/analyses/scripts/autocorrelation.py"

rule eval_hot_day_connected_components:
    input:
        zarr_rechunk_era5=IMPACT_VARIABLES_GT_RECHUNK,
        zarr_wg=IMPACT_VARIABLES_WG,
        zarr_rechunk_wg=IMPACT_VARIABLES_WG_RECHUNK,
        trajectories=expand(
            os.path.join(
                config["paths"]["dir_simulations"],
                "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
                "{seed}_{sigma}_{blocksize}/trajectory.nc",
            ),
            merge_ds_type=config["dataset_type"],
            seed=np.arange(config["eval_hot_day_connected_components"]["n_seeds"]),
            blocksize=config["eval_hot_day_connected_components"]["blocksize"],
            sigma=config["sigma"],
            allow_missing=True,
        ),
    output:
        nc_max_fraction_area_gt=os.path.join(
            config["paths"]["dir_results"],
            "hot_day_connected_components",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{use_lsm}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_hot_connected}",
            "max_fraction_area_gt.nc",
        ),
        nc_max_fraction_area_wg=os.path.join(
            config["paths"]["dir_results"],
            "hot_day_connected_components",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{use_lsm}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_hot_connected}",
            "max_fraction_area_wg.nc",
        ),
        nc_quantiles_gt=os.path.join(
            config["paths"]["dir_results"],
            "hot_day_connected_components",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{use_lsm}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_hot_connected}",
            "quantiles_gt.nc",
        ),
        nc_quantiles_wg=os.path.join(
            config["paths"]["dir_results"],
            "hot_day_connected_components",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{use_lsm}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_hot_connected}",
            "quantiles_wg.nc",
        ),
        params=os.path.join(
            config["paths"]["dir_results"],
            "hot_day_connected_components",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{use_lsm}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_hot_connected}",
            "params.yaml",
        ),
    params:
        all_params=get_all_params_dict("eval_hot_day_connected_components"),
        tracked_params=get_params_dict_for_saving("eval_hot_day_connected_components"),
    resources:
        runtime=2400,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=1,
    log:
        stdout=os.path.join(
            config["paths"]["dir_results"],
            "hot_day_connected_components",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "logs/log_{use_lsm}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_hot_connected}.out",
        ),
        stderr=os.path.join(
            config["paths"]["dir_results"],
            "hot_day_connected_components",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "logs/log_{use_lsm}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_hot_connected}.err",
        ),
    script:
        f"{dir_eval}/analyses/scripts/hot_day_connected_components.py"

rule eval_quantile_maps:
    input:
        zarr_rechunk_era5=IMPACT_VARIABLES_GT_RECHUNK,
        zarr_rechunk_wg=IMPACT_VARIABLES_WG_RECHUNK,
        trajectories=expand(
            os.path.join(
                config["paths"]["dir_simulations"],
                "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
                "{seed}_{sigma}_{blocksize}/trajectory.nc",
            ),
            merge_ds_type=config["dataset_type"],
            seed=np.arange(config["eval_quantile_maps"]["n_seeds"]),
            blocksize=config["eval_quantile_maps"]["blocksize"],
            sigma=config["sigma"],
            allow_missing=True,
        ),
    output:
        nc_quantiles_gt=os.path.join(
            config["paths"]["dir_results"],
            "quantile_maps",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{months}_"
            + hashlib.md5(
                ("{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_quantile_maps}",
            "quantiles_gt.nc",
        ),
        nc_quantiles_wg=os.path.join(
            config["paths"]["dir_results"],
            "quantile_maps",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{months}_"
            + hashlib.md5(
                ("{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_quantile_maps}",
            "quantiles_wg.nc",
        ),
        nc_quantiles_yearly_gt=os.path.join(
            config["paths"]["dir_results"],
            "quantile_maps",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{months}_"
            + hashlib.md5(
                ("{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_quantile_maps}",
            "quantiles_yearly_gt.nc",
        ),
        nc_quantiles_yearly_wg=os.path.join(
            config["paths"]["dir_results"],
            "quantile_maps",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{months}_"
            + hashlib.md5(
                ("{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_quantile_maps}",
            "quantiles_yearly_wg.nc",
        ),
        params=os.path.join(
            config["paths"]["dir_results"],
            "quantile_maps",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{months}_"
            + hashlib.md5(
                ("{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_quantile_maps}",
            "params.yaml",
        ),
    log:
        stdout=os.path.join(
            config["paths"]["dir_results"],
            "quantile_maps",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "logs/log_{months}_"
            + hashlib.md5(
                ("{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_quantile_maps}.out",
        ),
        stderr=os.path.join(
            config["paths"]["dir_results"],
            "quantile_maps",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "logs/log_{months}_"
            + hashlib.md5(
                ("{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_quantile_maps}.err",
        ),
    params:
        all_params=get_all_params_dict("eval_quantile_maps"),
        tracked_params=get_params_dict_for_saving("eval_quantile_maps"),
    resources:
        runtime=600,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=8,
    script:
        f"{dir_eval}/analyses/scripts/quantile_maps.py"


rule eval_rolling_droughts:
    input:
        nc_era5_climatology=CLIM_IMPACT_VARIABLES_GT,
        zarr_era5=IMPACT_VARIABLES_GT,
        zarr_wg=IMPACT_VARIABLES_WG,
        trajectories=expand(
            os.path.join(
                config["paths"]["dir_simulations"],
                "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
                "{seed}_{sigma}_{blocksize}/trajectory.nc",
            ),
            merge_ds_type=config["dataset_type"],
            seed=np.arange(config["eval_rolling_droughts"]["n_seeds"]),
            blocksize=config["eval_rolling_droughts"]["blocksize"],
            sigma=config["sigma"],
            allow_missing=True,
        ),
    output:
        nc_rolling_sum_wg=os.path.join(
            config["paths"]["dir_results"],
            "rolling_droughts",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{use_anomalies}_{n_months_rolling}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_rolling}",
            "quantiles_wg.nc",
        ),
        nc_rolling_sum_gt=os.path.join(
            config["paths"]["dir_results"],
            "rolling_droughts",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{use_anomalies}_{n_months_rolling}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_rolling}",
            "quantiles_gt.nc",
        ),
        params=os.path.join(
            config["paths"]["dir_results"],
            "rolling_droughts",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "{use_anomalies}_{n_months_rolling}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_rolling}",
            "params.yaml",
        ),
    params:
        all_params=get_all_params_dict("eval_rolling_droughts"),
        tracked_params=get_params_dict_for_saving("eval_rolling_droughts"),
    log:
        stdout=os.path.join(
            config["paths"]["dir_results"],
            "rolling_droughts",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "logs/log_{use_anomalies}_{n_months_rolling}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_rolling}.out",
        ),
        stderr=os.path.join(
            config["paths"]["dir_results"],
            "rolling_droughts",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "logs/log_{use_anomalies}_{n_months_rolling}_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_rolling}.err",
        ),
    resources:
        runtime=2500,
        mem_mb_per_cpu=GB_TO_MB * 256,
        cpus_per_task=1,
    script:
        f"{dir_eval}/analyses/scripts/rolling_droughts.py"


rule eval_seasonal_correlation:
    input:
        zarr_rechunk_era5=IMPACT_VARIABLES_GT_RECHUNK,
        zarr_rechunk_wg=IMPACT_VARIABLES_WG_RECHUNK,
        trajectories=expand(
            os.path.join(
                config["paths"]["dir_simulations"],
                "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
                "{seed}_{sigma}_{blocksize}/trajectory.nc",
            ),
            merge_ds_type=config["dataset_type"],
            seed=np.arange(config["eval_seasonal_correlation"]["n_seeds"]),
            blocksize=config["eval_seasonal_correlation"]["blocksize"],
            sigma=config["sigma"],
            allow_missing=True,
        ),
    output:
        nc_tas_pr_pearsonr=os.path.join(
            config["paths"]["dir_results"],
            "seasonal_correlation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_season_corr}",
            "tas_pr_pearsonr.nc",
        ),
        nc_tas_pr_pearsonp=os.path.join(
            config["paths"]["dir_results"],
            "seasonal_correlation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_season_corr}",
            "tas_pr_pearsonp.nc",
        ),
        nc_tas_pr_spearmanr=os.path.join(
            config["paths"]["dir_results"],
            "seasonal_correlation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_season_corr}",
            "tas_pr_spearmanr.nc",
        ),
        nc_tas_pr_spearmanp=os.path.join(
            config["paths"]["dir_results"],
            "seasonal_correlation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_season_corr}",
            "tas_pr_spearmanp.nc",
        ),
        # nc_e_dist=os.path.join(
        #     config["paths"]["dir_results"],
        #     "seasonal_correlation",
        #     "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
        #     hashlib.md5(
        #         (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
        #     ).hexdigest()[:8]
        #     + "_{hash_season_corr}",
        #     "e_dist.nc",
        # ),
        # nc_p_value_e_dist=os.path.join(
        #     config["paths"]["dir_results"],
        #     "seasonal_correlation",
        #     "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
        #     hashlib.md5(
        #         (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
        #     ).hexdigest()[:8]
        #     + "_{hash_season_corr}",
        #     "p_value_e_dist.nc",
        # ),
        params=os.path.join(
            config["paths"]["dir_results"],
            "seasonal_correlation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_season_corr}",
            "params.yaml",
        ),
    params:
        all_params=get_all_params_dict("eval_seasonal_correlation"),
        tracked_params=get_params_dict_for_saving("eval_seasonal_correlation"),
    log:
        stdout=os.path.join(
            config["paths"]["dir_results"],
            "seasonal_correlation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "logs/log_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_season_corr}.out",
        ),
        stderr=os.path.join(
            config["paths"]["dir_results"],
            "seasonal_correlation",
            "{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}",
            "logs/log_"
            + hashlib.md5(
                (f"{IMPACT_VARIABLES_GT}_{IMPACT_VARIABLES_WG}").encode()
            ).hexdigest()[:8]
            + "_{hash_season_corr}.err",
        ),
    resources:
        runtime=2000,
        mem_mb_per_cpu=GB_TO_MB * 256,
        cpus_per_task=1,
    script:
        f"{dir_eval}/analyses/scripts/seasonal_correlation.py"

