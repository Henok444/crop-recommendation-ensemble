from pathlib import Path 
import joblib 

def save_model(model , path: Path ):
    path.parent.mkdir(parents = True , exist_ok = True )
    joblib.dump(model , path)

def load_model(path : Path ):
    return joblib.load(path)

# for the encoder part 

def save_encoded_label(lr , path: Path):

    path.parent.mkdir(parents = True , exist_ok = True )
    joblib.dump(lr , path)
def load_encoder(path: Path):
    return joblib.load(path)