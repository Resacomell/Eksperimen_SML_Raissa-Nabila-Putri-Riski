import os  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def automation_preprocessing(input_path, output_train_path, output_test_path):
    
    df = pd.read_csv(input_path)
    print(f"- Data berhasil dimuat. Total baris awal: {len(df)}")
    
    # 1. Menangani Data Kosong (Missing Values)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['blood_glucose_level'] = df['blood_glucose_level'].fillna(df['blood_glucose_level'].median())
    
    # 2. Menghapus Data Duplikatt
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"- Data duplikat dihapus. Total baris sekarang: {len(df)}")
    
    # 3. Deteksi dan Penanganan Outlier (IQR Capping pada BMI)
    Q1 = df['bmi'].quantile(0.25)
    Q3 = df['bmi'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['bmi'] = np.where(df['bmi'] > upper_bound, upper_bound, 
                         np.where(df['bmi'] < lower_bound, lower_bound, df['bmi']))
    
    # 4. Binning (Pengelompokan Data Umur)
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 60, 100], labels=[0, 1, 2], include_lowest=True)
    df['age_group'] = df['age_group'].astype(int)
    
    # 5. Encoding Data Kategorikal
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    le_smoking = LabelEncoder()
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])
    
    # Memisahkan Fitur dan Target sebelum standardisasi
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # 6. Normalisasi atau Standarisasi Fitur
    scaler = StandardScaler()
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Gabungkan kembali fitur dan target untuk disimpan sebagai file siap latih
    train_clean = X_train.copy()
    train_clean['diabetes'] = y_train
    
    test_clean = X_test.copy()
    test_clean['diabetes'] = y_test
    
    # Menyimpan file dataset hasil preprocessing ke folder yang sama
    train_clean.to_csv(output_train_path, index=False)
    test_clean.to_csv(output_test_path, index=False)
    
    print(f"Otomatisasi Berhasil! Data siap dilatih disimpan di: {output_train_path}")
    return train_clean, test_clean

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    INPUT_DATA = os.path.join(current_dir, "..", "diabetes_prediction_dataset.csv")
    OUTPUT_TRAIN = os.path.join(current_dir, "diabetes_prediction_dataset_preprocessing.csv")
    OUTPUT_TEST = os.path.join(current_dir, "diabetes_prediction_dataset_testing.csv")
    
    # Eksekusi fungsi
    automation_preprocessing(INPUT_DATA, OUTPUT_TRAIN, OUTPUT_TEST)
