import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_preprocessing import preprocess_data
from model_training import train_model
from evaluation import evaluate_model

def test_preprocessing():
    train_data, test_data = preprocess_data()
    assert train_data[0].shape[0] > 0
    assert test_data[0].shape[0] > 0

def test_model_training():
    train_data, _ = preprocess_data()
    model = train_model(train_data)
    assert model is not None
    assert hasattr(model, "predict")

def test_evaluation_creates_report():
    train_data, test_data = preprocess_data()
    model = train_model(train_data)
    evaluate_model(model, test_data)
    assert os.path.exists("reports/metrics.txt")
