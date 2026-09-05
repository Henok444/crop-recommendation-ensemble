

def compare_experiments(experiments):
    best_model = max(
        experiments , key= lambda x : x["weighted_f1"]
    )
    return best_model 

