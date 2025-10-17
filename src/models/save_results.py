import json
from datetime import datetime
import os

def save_model_results(model_name, r2_score, cv_r2_mean=None, cv_r2_std=None):
    """
    Save model results to a JSON file
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results dictionary
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            model_name: {
                "r2_score": r2_score,
                "cv_r2_mean": cv_r2_mean,
                "cv_r2_std": cv_r2_std
            }
        }
    }
    
    # Save to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f'model_results_{timestamp}.json')
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    return filename

