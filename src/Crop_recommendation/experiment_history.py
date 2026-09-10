from pathlib import Path
import json



def load_all_experiments(result_path: Path):

    

    experiments = []

    for path in result_path.glob("*.json"):

        with open(path , "r") as file:
            results = json.load(file)
            results["experiment_file"] = path.name
            experiments.append(results)
    return experiments

