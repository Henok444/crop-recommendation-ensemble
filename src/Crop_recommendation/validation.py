feature_columns   = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

def validate_features(features: dict ) -> None : 

    missing = [feature  
               for feature in feature_columns 
               if feature not in features ]

    if missing:
        raise ValueError(f"Repuired feature {missing} is missing.  ")
    for feature in feature_columns:
        if not isinstance(features[feature] , (int, float)):
            raise ValueError(f"{feature} must be numeric.")