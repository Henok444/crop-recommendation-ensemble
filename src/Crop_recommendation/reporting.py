import pandas as pd 
from pathlib import Path

def create_experiment_report(experiments):
    if not experiments :
        raise ValueError("No experiments provided")


    rows = []

    for experiment in experiments:
        rows.append({
            "model": experiment["model"],
            "accuracy": experiment["accuracy"],
            "weighted_f1": experiment["weighted_f1"],
            "experiment_file": experiment["experiment_file"],
        })
    report = pd.DataFrame(rows)

    return report.sort_values(by = "weighted_f1" , ascending= False ).reset_index(drop= True)
def save_report(report , path :Path ):
    path.parent.mkdir(exist_ok=True , parents= True)
    report.to_csv(path , index = False)


        