from src.Crop_recommendation.data import load_data 

def test_load_data():
    df = load_data()

    assert df.shape == (2200,8)
    assert len(df.columns) == 8