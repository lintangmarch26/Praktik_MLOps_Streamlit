import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np

print("Sedang melatih model AI dengan dataset yang lebih besar...")

# Data Latih (20 Data Mahasiswa: [IPK, Total SKS])
X_train = np.array([
    # [IPK, SKS]
    [3.8, 150], [3.5, 144], [3.2, 140], [3.6, 148], [3.9, 155], # Sangat Layak Lulus
    [3.1, 144], [2.9, 142], [3.0, 140], [2.8, 144], [3.4, 146], # Lulus Syarat Batas
    [2.0, 100], [2.5, 110], [1.8, 90],  [2.2, 105], [2.4, 115], # Jelas Tidak Lulus (SKS & IPK Rendah)
    [3.9, 80],  [3.7, 95],  [3.5, 110], # Kasus Khusus: IPK Tinggi tapi SKS Kurang
    [2.3, 144], [2.1, 140]              # Kasus Khusus: SKS Cukup tapi IPK Kurang
])

# Label: 1 = Lulus Tepat Waktu, 0 = Tidak Lulus
y_train = np.array([
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0,
    0, 0
])

# Inisialisasi & Latih Model Decision Tree
# max_depth dibatasi agar model membuat pola cabang yang rapi
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Simpan model ke file .pkl
joblib.dump(model, 'model_kelulusan.pkl')
print("Sukses! Model baru dengan 20 data telah berhasil disimpan sebagai 'model_kelulusan.pkl'")