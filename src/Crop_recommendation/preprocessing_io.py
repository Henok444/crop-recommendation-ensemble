from pathlib import Path 
import joblib


# for the encoder part 

def save_encoded_label(lr , path: Path):

    path.parent.mkdir(parents = True , exist_ok = True )
    joblib.dump(lr , path)
def load_encoder(path: Path):
    return joblib.load(path)