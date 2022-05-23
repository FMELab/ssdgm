from os import renames
import optuna
from optuna.samplers import TPESampler

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score, mean_absolute_percentage_error

from src.datamodules.datamodules import *

import time
import argparse
import re
import warnings

from pathlib import Path



SEED = 42


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor_name")
    parser.add_argument("--mode")

    args = parser.parse_args()

    mode = args.mode
    regressor_name = args.regressor_name

    

    base_directory = '/home/flo/ssdgm'
    log_directory = 'logs/experiments/baselines'
    

    warnings.filterwarnings("ignore")

    
    results_directory_stub = os.path.join(base_directory, log_directory)

    if mode == "tuning":

        tuning_directory_stub = os.path.join(results_directory_stub, mode, regressor_name)
        if not Path(tuning_directory_stub).is_dir():
            os.makedirs(tuning_directory_stub)

        n_trials = 1000
        sampler = TPESampler(
            seed=42,
            consider_prior=True,
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=False,
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=False,
            warn_independent_sampling=True,
        )

        # Setup the datamodules
        use_unlabeled_dataloader = False
        batch_size=500

        datamodules = [
            SkillcraftDataModule(batch_size=batch_size, use_unlabeled_dataloader=False),
            ProteinDataModule(batch_size=batch_size, use_unlabeled_dataloader=False),
            ParkinsonDataModule(batch_size=batch_size, use_unlabeled_dataloader=False),
            ElevatorsDataModule(batch_size=batch_size, use_unlabeled_dataloader=False),
            CTSliceDataModule(batch_size=batch_size, use_unlabeled_dataloader=False),
            BlogDataModule(batch_size=batch_size, use_unlabeled_dataloader=False),
        ]

        results_list = []
        for dm in datamodules:
            dm.prepare_data()
            dm.setup()

            dataset_name = dm.__class__.__name__.lower().split("datamodule")[0]


            train_indices = dm.data_train_labeled.indices
            val_indices = dm.data_val.indices

            X_train, y_train = dm.dataset[train_indices]
            X_train, y_train = X_train.numpy(), y_train.numpy().ravel()

            X_val, y_val = dm.dataset[val_indices]
            X_val, y_val = X_val.numpy(), y_val.numpy().ravel()

            
            optuna.logging.set_verbosity(optuna.logging.DEBUG)
            study = optuna.create_study(
                study_name=regressor_name + '_' + dataset_name,
                direction="minimize",
                sampler=sampler
            )

            for i in range(n_trials):
                
                
                print(f"\rTrial {i + 1}/{n_trials}", sep='', end='', flush=True)
                trial = study.ask()

                if regressor_name == "RFR":
                    rf_n_estimators = trial.suggest_int('rf_n_estimators', low=50, high=1000)
                    rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 20)
                    rf_min_samples_leaf =trial.suggest_int('rf_min_samples_leaf', 1, 20)
                    rf_max_features = trial.suggest_uniform('rf_max_features', 0.5, 1.0)
                    rf_max_samples = trial.suggest_uniform('rf_max_samples', 0.6, 1.0)
                    regressor = RandomForestRegressor(
                        n_estimators=rf_n_estimators,
                        min_samples_split=rf_min_samples_split,
                        min_samples_leaf=rf_min_samples_leaf,
                        max_features=rf_max_features,
                        max_samples=rf_max_samples,
                    )
                elif regressor_name == "SVR":
                    svr_kernel = trial.suggest_categorical("svr_kernel", ["rbf", "sigmoid"])
                    svr_degree = trial.suggest_int("svr_degree", 3, 15)
                    svr_C = trial.suggest_loguniform("svr_C", 1e-5, 1e2)
                    svr_epsilon = trial.suggest_loguniform("svr_epsilon", 1e-5, 1e2)
                    regressor = SVR(
                        kernel=svr_kernel,
                        degree=svr_degree,
                        C=svr_C,
                        epsilon=svr_epsilon,
                    )
                elif regressor_name == "Ridge":
                    ridge_alpha = trial.suggest_loguniform("ridge_alpha", 1e-5, 1e2)
                    regressor = Ridge(
                        alpha=ridge_alpha,
                        fit_intercept=False,
                        random_state=42,
                    )
                else:
                    raise RuntimeError("Invalid regressor! Only RandomForestRegressor (RFR) and SupportVectorRegressor (SVR) are valid regressors.")


                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_val)

                mse = mean_squared_error(y_val, y_pred)

                study.tell(trial, mse)
                

            df_study_results = pd.DataFrame.from_dict([study.best_params])
            df_study_results["dataset"] = dataset_name
            df_study_results["best_value"] = study.best_value
            results_list.append(df_study_results)
            print(f"\nFinished study for dataset: {dataset_name}")

        df_results = pd.concat(results_list, axis=0, ignore_index=True)
        df_results["regressor"] = regressor_name
        df_results.to_csv(os.path.join(tuning_directory_stub, f'df_optuna_results_{n_trials}.csv'), index=False)
        
        print("Tuning finished.")

    elif mode == "production":
        
        n_samples = [100, 200, 300, 400, 500]

        production_directory_stub = os.path.join(results_directory_stub, "production", regressor_name)        
        if not Path(production_directory_stub).is_dir():
            os.makedirs(production_directory_stub)

        tuning_directory_stub = os.path.join(results_directory_stub, "tuning", regressor_name)

        if regressor_name == "RFR" or regressor_name == "SVR" or regressor_name == "Ridge":
            df_params = pd.read_csv(os.path.join(tuning_directory_stub, 'df_optuna_results_1000.csv'))

        datasets = ["skillcraft", "protein", "parkinson", "elevators", "ctslice", "blog"]

        results_dict = {
            "regressor_name": [],
            "dataset": [],
            "n_train_samples": [],
            "run": [],
            "mse": [],
            "rmse": [],
            "explained_variance": [],
            "mae": [],
            "mape": [],
            "r2": [],
        }
        for ds in datasets:
            if regressor_name == "RFR" or regressor_name == "SVR" or regressor_name == "Ridge":
                df_model_params = df_params[df_params["dataset"] == ds]

            for n in n_samples:

                for i in range(30):
                    print(f"[INFO]: dataset={ds}    n_samples={n}    run={i}", flush=True)
                    print('\033[1A', end='\x1b[2K')

                    seed = SEED + i
                    if ds == "skillcraft":
                        dm = SkillcraftDataModule(n_samples_train_labeled=n, split_seed=seed)
                    elif ds == "protein":
                        dm = ProteinDataModule(n_samples_train_labeled=n, split_seed=seed)
                    elif ds == "parkinson":
                        dm = ParkinsonDataModule(n_samples_train_labeled=n, split_seed=seed)
                    elif ds == "elevators":
                        dm = ElevatorsDataModule(n_samples_train_labeled=n, split_seed=seed)
                    elif ds == "ctslice":
                        dm = CTSliceDataModule(n_samples_train_labeled=n, split_seed=seed)
                    elif ds == "blog":
                        dm = BlogDataModule(n_samples_train_labeled=n, split_seed=seed)

                    dm.prepare_data()
                    dm.setup()

                    train_indices = dm.data_train_labeled.indices
                    test_indices = dm.data_test.indices

                    X_train, y_train = dm.dataset[train_indices]
                    X_train, y_train = X_train.numpy(), y_train.numpy().ravel()

                    X_test, y_test = dm.dataset[test_indices]
                    X_test, y_test = X_test.numpy(), y_test.numpy().ravel()
                    
                    # Initialize regressor
                    if regressor_name == "Dummy":
                        regressor = DummyRegressor()
                    elif regressor_name == "LR":
                        regressor = LinearRegression(fit_intercept=False)
                    elif regressor_name == "Ridge":
                        regressor = Ridge(
                            alpha=float(df_model_params["ridge_alpha"]),
                            fit_intercept=False,
                            random_state=42,
                            )
                    elif regressor_name == "RFR":
                        regressor = RandomForestRegressor(
                            n_estimators=int(df_model_params["rf_n_estimators"]),
                            min_samples_split=int(df_model_params["rf_min_samples_split"]),
                            min_samples_leaf=int(df_model_params["rf_min_samples_leaf"]),
                            max_features=float(df_model_params["rf_max_features"]),
                            max_samples=float(df_model_params["rf_max_samples"]),
                        )
                    elif regressor_name == "SVR":
                        regressor = SVR(
                            kernel=df_model_params["svr_kernel"].item(),
                            degree=int(df_model_params["svr_degree"]),
                            C=float(df_model_params["svr_C"]),
                            epsilon=float(df_model_params["svr_epsilon"]),
                        )
                    elif regressor_name == "MLP":
                        regressor = MLPRegressor(
                            hidden_layer_sizes=[1024, 1024, 1024],
                            learning_rate="adaptive",
                        )
                    else:
                        raise NameError("Invalid regressor name! Only RandomForestRegressor (RFR) and SupportVectorRegressor (SVR) are valid regressors.")

                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    
                    #print("pred:", y_pred)
                    #print("true:", y_test)
                    #print(regressor.n_features_in_)

                    mse = mean_squared_error(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    explained_variance = explained_variance_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    if regressor_name == "SVR":
                        results_dict["regressor_name"].append(regressor_name + '+' + df_model_params["svr_kernel"].item())
                    else:
                        results_dict["regressor_name"].append(regressor_name)
                    
                    results_dict["dataset"].append(ds)
                    results_dict["n_train_samples"].append(n)
                    results_dict["run"].append(i)
                    results_dict["mse"].append(mse)
                    results_dict["rmse"].append(rmse)
                    results_dict["explained_variance"].append(explained_variance)
                    results_dict["mae"].append(mae)
                    results_dict["mape"].append(mape)
                    results_dict["r2"].append(r2)

                    del regressor

        df_results = pd.DataFrame.from_dict(results_dict)
        df_results.to_csv(os.path.join(production_directory_stub, 'df_optuna_results_1000.csv'), index=False)
        print(f"Finished varying labeled data points for regressor: {regressor_name}")

            
    else:
        raise NameError("Invalid mode: ", mode)

    # sk_dm = SkillcraftDataModule(batch_size=500, use_unlabeled_dataloader=False)
    # sk_dm.prepare_data()
    # sk_dm.setup()
    # print(sk_dm.data_train_labeled)

    # park_dm = ParkinsonDataModule(batch_size=500, use_unlabeled_dataloader=False)
    # park_dm.prepare_data()
    # park_dm.setup()

    # X_train, y_train = sk_dm.dataset[sk_dm.data_train_labeled.indices]

    # X_train, y_train = X_train.numpy(), y_train.numpy().ravel()

    # start = time.time()
    
    # rf = RandomForestRegressor()
    # for i in range(600):
    #     rf.fit(X_train, y_train)
    #     print(i)
    
    # print(f"Execution time: {time.time() - start} secs")
    # print("FINISHED.")

