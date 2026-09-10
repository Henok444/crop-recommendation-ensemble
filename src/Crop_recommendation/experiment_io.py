import json 
from pathlib import Path 
from datetime import datetime

def save_experiment(result , path: Path ):
    path.parent.mkdir(parents=True , exist_ok= True )

    with open(path , "w") as file: 
        json.dump(result , file , indent= 4 )
def load_experiment(path : Path ):

    with open(path , "r") as file:
        return json.load(file)
def create_experiment_path(model_name: str )->Path:
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return Path(f"results/{time_stamp}_{model_name}.json")



                