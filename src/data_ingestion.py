import os
import subprocess

def download_bank_data(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "bank.csv")
    if os.path.exists(csv_path):
        print("bank.csv already exists. Skipping download.")
        return csv_path

    print("Downloading bank.csv from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download", 
        "-d", "janiobachmann/bank-marketing-dataset", 
        "-f", "bank.csv",
        "-p", data_dir,
        "--unzip"
    ], check=True)
    print("Download complete.")
    return csv_path

if __name__ == "__main__":
    download_bank_data()