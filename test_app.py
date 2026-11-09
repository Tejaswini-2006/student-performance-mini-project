"""
Automated Test Suite for Student Performance Prediction System
==============================================================
Validates machine learning pipeline, dataset integrity, input shapes,
prediction logic, and edge-case handling.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EXPECTED_CATEGORICAL = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]

EXPECTED_NUMERICAL = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
]

TOTAL_FEATURES = len(EXPECTED_CATEGORICAL) + len(EXPECTED_NUMERICAL)

def test_model_file_exists():
    print("[TEST 1] Checking model serialization file...")
    assert os.path.exists("student_performance_random_forest.pkl"), "Random forest model file missing!"
    print("  ✓ Model file exists.")

def test_model_pipeline_structure():
    print("[TEST 2] Validating Scikit-learn Pipeline architecture...")
    model = joblib.load("student_performance_random_forest.pkl")
    assert isinstance(model, Pipeline), "Loaded object is not an sklearn Pipeline!"
    assert "preprocessor" in model.named_steps, "Pipeline missing 'preprocessor' step!"
    assert "model" in model.named_steps, "Pipeline missing 'model' step!"
    print("  ✓ Pipeline structure verified.")

def test_dataset_integrity():
    print("[TEST 3] Checking dataset dimensions and columns...")
    assert os.path.exists("student-mat.csv"), "student-mat.csv dataset missing!"
    df = pd.read_csv("student-mat.csv", sep=";")
    assert df.shape == (395, 33), f"Unexpected dataset shape: {df.shape}"
    assert "G3" in df.columns, "Target column G3 missing from dataset!"
    assert "G1" in df.columns and "G2" in df.columns, "G1 or G2 columns missing from raw dataset!"
    print(f"  ✓ Dataset verified: {df.shape[0]} rows, {df.shape[1]} columns.")

def test_feature_definitions():
    print("[TEST 4] Validating 30 input features (excluding G1, G2, G3)...")
    df = pd.read_csv("student-mat.csv", sep=";")
    features = [c for c in df.columns if c not in ["G1", "G2", "G3"]]
    assert len(features) == TOTAL_FEATURES == 30, f"Expected 30 features, got {len(features)}"
    for cat in EXPECTED_CATEGORICAL:
        assert cat in features, f"Missing categorical feature: {cat}"
    for num in EXPECTED_NUMERICAL:
        assert num in features, f"Missing numerical feature: {num}"
    print(f"  ✓ All 30 input features properly accounted for.")

def test_model_evaluation_metrics():
    print("[TEST 5] Validating historical model evaluation reproduction...")
    model = joblib.load("student_performance_random_forest.pkl")
    df = pd.read_csv("student-mat.csv", sep=";")
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["G1", "G2", "G3"])
    y = df["G3"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"  Observed: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    assert round(mae, 3) == 2.998, f"MAE mismatch! Expected 2.998, got {round(mae, 3)}"
    assert round(rmse, 3) == 3.795, f"RMSE mismatch! Expected 3.795, got {round(rmse, 3)}"
    assert round(r2, 3) == 0.298, f"R2 mismatch! Expected 0.298, got {round(r2, 3)}"
    print("  ✓ Exact reproduction of documented experimental metrics verified!")

def test_different_student_profiles():
    print("[TEST 6] Testing prediction on diverse student profiles...")
    model = joblib.load("student_performance_random_forest.pkl")
    
    # 1. High Achiever Profile
    high_student = pd.DataFrame([{
        'school': 'GP', 'sex': 'F', 'age': 16, 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'T', 'Medu': 4, 'Fedu': 4, 'Mjob': 'health', 'Fjob': 'teacher',
        'reason': 'reputation', 'guardian': 'mother', 'traveltime': 1, 'studytime': 4,
        'failures': 0, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no',
        'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
        'romantic': 'no', 'famrel': 5, 'freetime': 2, 'goout': 2, 'Dalc': 1,
        'Walc': 1, 'health': 5, 'absences': 1
    }])
    pred_high = model.predict(high_student)[0]
    print(f"  High Achiever predicted grade: {pred_high:.2f} / 20")
    assert 0 <= pred_high <= 20, "Predicted grade out of valid bounds!"
    
    # 2. Average Student Profile
    avg_student = pd.DataFrame([{
        'school': 'GP', 'sex': 'M', 'age': 17, 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'T', 'Medu': 2, 'Fedu': 2, 'Mjob': 'services', 'Fjob': 'other',
        'reason': 'course', 'guardian': 'mother', 'traveltime': 2, 'studytime': 2,
        'failures': 0, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no',
        'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
        'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 3, 'Dalc': 1,
        'Walc': 2, 'health': 3, 'absences': 6
    }])
    pred_avg = model.predict(avg_student)[0]
    print(f"  Average Student predicted grade: {pred_avg:.2f} / 20")
    assert 0 <= pred_avg <= 20, "Predicted grade out of valid bounds!"
    
    # 3. At-Risk Student Profile
    risk_student = pd.DataFrame([{
        'school': 'MS', 'sex': 'M', 'age': 19, 'address': 'R', 'famsize': 'LE3',
        'Pstatus': 'A', 'Medu': 1, 'Fedu': 1, 'Mjob': 'at_home', 'Fjob': 'other',
        'reason': 'other', 'guardian': 'other', 'traveltime': 3, 'studytime': 1,
        'failures': 3, 'schoolsup': 'yes', 'famsup': 'no', 'paid': 'no',
        'activities': 'no', 'nursery': 'no', 'higher': 'no', 'internet': 'no',
        'romantic': 'yes', 'famrel': 2, 'freetime': 4, 'goout': 5, 'Dalc': 3,
        'Walc': 4, 'health': 2, 'absences': 24
    }])
    pred_risk = model.predict(risk_student)[0]
    print(f"  At-Risk Student predicted grade: {pred_risk:.2f} / 20")
    assert 0 <= pred_risk <= 20, "Predicted grade out of valid bounds!"
    assert pred_high > pred_risk, "High achiever should score higher than at-risk student!"
    print("  ✓ All profile test cases executed with sound relative predictions.")

def test_unseen_categorical_handling():
    print("[TEST 7] Testing resilience against unknown categorical labels...")
    model = joblib.load("student_performance_random_forest.pkl")
    odd_student = pd.DataFrame([{
        'school': 'UNKNOWN_SCHOOL', 'sex': 'OTHER', 'age': 18, 'address': 'SUBURB',
        'famsize': 'UNKNOWN', 'Pstatus': 'DIVORCED', 'Medu': 2, 'Fedu': 2,
        'Mjob': 'astronaut', 'Fjob': 'diver', 'reason': 'scholarship',
        'guardian': 'grandparent', 'traveltime': 2, 'studytime': 2,
        'failures': 0, 'schoolsup': 'no', 'famsup': 'no', 'paid': 'no',
        'activities': 'no', 'nursery': 'no', 'higher': 'yes', 'internet': 'yes',
        'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 3, 'Dalc': 1,
        'Walc': 1, 'health': 4, 'absences': 4
    }])
    # Pipeline should handle gracefully via OneHotEncoder(handle_unknown='ignore')
    pred_odd = model.predict(odd_student)[0]
    print(f"  Unknown category prediction: {pred_odd:.2f} / 20")
    assert not np.isnan(pred_odd), "Prediction returned NaN on unseen category!"
    print("  ✓ OneHotEncoder handle_unknown='ignore' successfully absorbed unseen values.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING STUDENT PERFORMANCE PREDICTION TEST SUITE")
    print("=" * 60)
    test_model_file_exists()
    test_model_pipeline_structure()
    test_dataset_integrity()
    test_feature_definitions()
    test_model_evaluation_metrics()
    test_different_student_profiles()
    test_unseen_categorical_handling()
    print("=" * 60)
    print("ALL 7 SYSTEMATIC TESTS PASSED SUCCESSFULLY! (100% PASS)")
    print("=" * 60)
