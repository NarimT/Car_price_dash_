import pandas as pd
import numpy as np
import joblib
from src import *

# Load model once at module level
MODEL_URI = "models/logistic_model.joblib"
loaded_model = joblib.load(MODEL_URI)
sample_data = [[2014, 103.52, 0 ,27, 21.14,]]
column_names = ['year', 'max_power', 'fuel','brand', 'mileage']
sample_df = pd.DataFrame(sample_data, columns=column_names)
sample_array = sample_df.to_numpy()
sample_with_bias = np.insert(sample_array, 0, 1, axis=1)



def test_model_input():
    """Test that the model accepts correct input format"""
    
    # Test that model accepts the input without errors
    try:
        prediction = loaded_model.predict(sample_with_bias)
        print("✓ Model accepts correct input format")
        return True
    except Exception as e:
        print(f"✗ Model failed to accept input: {e}")
        return False


def test_model_output_shape():
    """Test that the model output has the correct shape"""
    
    # Get prediction
    prediction = loaded_model.predict(sample_with_bias)
    
    # Check output shape
    expected_length = len(sample_with_bias)
    actual_length = len(prediction)
    
    if actual_length == expected_length:
        print(f"✓ Output shape is correct: {actual_length} prediction(s) for {expected_length} input(s)")
        print(f"  Predicted value: {prediction[0]}")
        return True
    else:
        print(f"✗ Output shape is incorrect: expected {expected_length}, got {actual_length}")
        return False

if __name__ == "__main__":
    import sys
    
    print("Running Model Tests...\n")
    
    test1_passed = test_model_input()
    test2_passed = test_model_output_shape()
    
    print("\n" + "="*50)
    if test1_passed and test2_passed:
        print("All tests passed! ✓")
        sys.exit(0)  # Exit with success code
    else:
        print("Some tests failed! ✗")
        sys.exit(1)  # Exit with failure code