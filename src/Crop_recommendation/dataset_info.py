def get_dataset_info(df):

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "features": [
            column
            for column in df.columns
            if column != "label"
        ],
        "target": "label",
    }