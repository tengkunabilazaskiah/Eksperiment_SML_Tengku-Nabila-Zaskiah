import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_heart_dataset():
    # Load dataset dari folder dataset_raw
    df = pd.read_csv("dataset_raw/heart.csv")

    # 🔍 Debug print untuk memastikan kolom yang terbaca
    print("Kolom dataset:", df.columns.tolist())
    print(df.head())

    # Hapus data duplikat
    df = df.drop_duplicates()

    # Pisahkan fitur dan target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Gabungkan kembali ke DataFrame
    df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    df_preprocessed["target"] = y.values

    # Simpan hasil preprocessing ke folder preprocessing
    df_preprocessed.to_csv("preprocessing/heart_preprocessed.csv", index=False)

    print("✅ Preprocessing selesai. File 'heart_preprocessed.csv' berhasil dibuat.")

if __name__ == "__main__":
    preprocess_heart_dataset()
