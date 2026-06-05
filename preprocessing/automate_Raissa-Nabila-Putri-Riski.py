import os  
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def automation_preprocessing(input_path, output_clean_path):
    
    df = pd.read_csv(input_path)
    print(f"- Data berhasil dimuat. Total baris awal: {len(df)}")
    
    # 1. Menangani Data Kosong (Missing Values)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['blood_glucose_level'] = df['blood_glucose_level'].fillna(df['blood_glucose_level'].median())
    
    # 2. Menghapus Data Duplikat
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
    
    scaler = StandardScaler()
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    df.to_csv(output_clean_path, index=False)
    
    print(f"Otomatisasi Berhasil! Data siap dilatih disimpan di: {output_clean_path}")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    INPUT_DATA = os.path.join(current_dir, "..", "diabetes_prediction_dataset.csv")
    
    OUTPUT_CLEAN = os.path.join(current_dir, "diabetes_prediction_dataset_preprocessing.csv")
    
    automation_preprocessing(INPUT_DATA, OUTPUT_CLEAN)