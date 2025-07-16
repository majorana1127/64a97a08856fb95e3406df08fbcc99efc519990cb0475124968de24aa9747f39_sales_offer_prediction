from data_preprocessing import preprocess_data
from model_training import train_model
from evaluation import evaluate_model

def main():
    train_data, test_data = preprocess_data()
    model = train_model(train_data)
    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()
