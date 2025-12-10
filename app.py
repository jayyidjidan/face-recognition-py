from flask import Flask, render_template, request, jsonify, redirect, url_for
import mysql.connector
import face_recognition
import numpy as np
import base64
import os
from datetime import datetime
import cv2

app = Flask(__name__)

# --- KONFIGURASI DATABASE ---
db_config = {
    'host': 'localhost',
    'user': 'root',      # Default XAMPP user
    'password': '',      # Default XAMPP password (kosong)
    'database': 'db_absensi'
}

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable untuk menyimpan encoding wajah agar cepat (cache)
known_face_encodings = []
known_face_names = []
known_face_ids = []

def get_db_connection():
    return mysql.connector.connect(**db_config)

def load_known_faces():
    """Memuat semua wajah dari database ke memory saat aplikasi start/update"""
    global known_face_encodings, known_face_names, known_face_ids
    
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM pegawai")
    pegawais = cursor.fetchall()
    
    print("Memuat data wajah...")
    for p in pegawais:
        try:
            image_path = p['foto_path']
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            
            if len(encoding) > 0:
                known_face_encodings.append(encoding[0])
                known_face_names.append(p['nama'])
                known_face_ids.append(p['id'])
        except Exception as e:
            print(f"Error loading {p['nama']}: {e}")
            
    cursor.close()
    conn.close()
    print(f"Selesai. {len(known_face_names)} wajah dimuat.")

# Load wajah saat aplikasi pertama kali jalan
load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Ambil data absensi gabung dengan nama pegawai
    query = """
        SELECT a.waktu_masuk, p.nama, p.jabatan 
        FROM absensi a 
        JOIN pegawai p ON a.pegawai_id = p.id 
        ORDER BY a.waktu_masuk DESC
    """
    cursor.execute(query)
    data_absen = cursor.fetchall()
    
    cursor.execute("SELECT * FROM pegawai")
    data_pegawai = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return render_template('dashboard.html', absensi=data_absen, pegawai=data_pegawai)

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    nama = data['nama']
    jabatan = data['jabatan']
    image_data = data['image'] # Base64 string

    # Decode Base64 Image
    header, encoded = image_data.split(",", 1)
    file_bytes = base64.b64decode(encoded)
    
    # Simpan File
    filename = f"{nama}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as f:
        f.write(file_bytes)

    # Simpan ke Database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO pegawai (nama, jabatan, foto_path) VALUES (%s, %s, %s)", 
                   (nama, jabatan, filepath))
    conn.commit()
    cursor.close()
    conn.close()

    # Reload wajah agar user baru langsung dikenali
    load_known_faces()

    return jsonify({"status": "success", "message": "Pendaftaran berhasil!"})

@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.json
    image_data = data['image']

    # Convert Base64 ke format opencv/numpy
    header, encoded = image_data.split(",", 1)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert ke RGB (face_recognition butuh RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Deteksi wajah
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    name = "Unknown"
    
    for face_encoding in face_encodings:
        # Bandingkan dengan database
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            pegawai_id = known_face_ids[best_match_index]
            
            # Catat Absensi ke DB
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Cek agar tidak spam absen dalam 1 menit terakhir (Opsional logic)
            cursor.execute("INSERT INTO absensi (pegawai_id) VALUES (%s)", (pegawai_id,))
            conn.commit()
            cursor.close()
            conn.close()
            
            return jsonify({"status": "success", "match": True, "name": name})

    return jsonify({"status": "success", "match": False})

if __name__ == '__main__':
    app.run(debug=True)