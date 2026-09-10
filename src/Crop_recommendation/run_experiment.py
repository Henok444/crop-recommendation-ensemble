
from src.Crop_recommendation.experiment import run_random_forest_experiment , run_xgboost_experiment , run_random_forest_tuning_experiment
from src.Crop_recommendation.experiment_io import save_experiment , load_experiment , create_experiment_path
from src.Crop_recommendation.comparision import compare_experiments
from src.Crop_recommendation.experiment_history import (
    load_all_experiments, 
)
from src.Crop_recommendation.reporting import (
    create_experiment_report,save_report
)
from pathlib import Path 


def run_expriment():
    random_forest_result = run_random_forest_tuning_experiment()
    rf_path = create_experiment_path("random_forest_experiment")
    save_experiment(random_forest_result , rf_path )


    xgboost_result = run_xgboost_experiment()
    xgb_path = create_experiment_path("xgboost_experiment")
    save_experiment(xgboost_result , xgb_path)

    print(f"Random Forest experiment saved to: {rf_path}")
    print(f"XGBoost experiment saved to: {xgb_path}")



    experiments = [random_forest_result , xgboost_result]

    best_model = compare_experiments(experiments)

    

    return experiments , best_model

if __name__ == "__main__":
    experiments , best_model = run_expriment()
    print("Experiment complete and saved .")
    print("\nExperiments:-\n" , experiments )
    print("\nBest_model:-\n" , best_model)


    all_saved_experiments = load_all_experiments(Path("results"))
    report = create_experiment_report(all_saved_experiments)
    save_report(report , Path("results/reports/model_comparision.csv") )
    print(f"Report_created to path : results/reports/model_comparision.csv")





    