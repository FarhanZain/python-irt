from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
import numpy as np
import pandas as pd
from girth import rasch_mml, twopl_mml, threepl_mml, ability_eap, ability_3pl_eap

app = FastAPI(title="IRT Analysis API Service")

# --- KONFIGURASI CORS ---
# Mengizinkan Next.js (frontend) untuk mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan ["http://localhost:3000"] di produksi demi keamanan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG DATABASE ---
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="irtpl2"
    )

# --- FUNGSI HELPER IRT ---
def hitung_skor_skala(theta, tipe_paket, benar_semua=False):
    is_utbk = "utbk" in tipe_paket.lower()
    if benar_semua:
        return 1000.0 if is_utbk else 100.0
        
    if is_utbk:
        skor = ((theta + 3) / 6) * 1000
        return float(np.clip(skor, 0, 1000))
    else:
        skor = 70 + (12 * theta)
        return float(np.clip(skor, 0, 100))

def hitung_probabilitas_soal(theta, diffs, discs, guesses=None):
    if guesses is None: 
        guesses = np.zeros(len(diffs))
    exponent = -discs * (theta - diffs)
    return guesses + (1 - guesses) / (1 + np.exp(exponent))

def tentukan_kategori_soal(nilai):
    if nilai < -1.0: return "Mudah"
    elif -1.0 <= nilai <= 1.0: return "Sedang"
    else: return "Susah"

def tentukan_kategori_daya_pembeda(nilai):
    if nilai < 0.0: return "Sangat Buruk"
    elif 0.0 <= nilai < 0.35: return "Buruk"
    elif 0.35 <= nilai < 0.65: return "Cukup"
    elif 0.65 <= nilai <= 1.35: return "Baik"
    else: return "Sangat Baik"

# def kategorikan_kemampuan(theta):
#     if theta < -0.5: return "Lemah"
#     elif -0.5 <= theta <= 0.5: return "Normal"
#     else: return "Kuat"

def kategorikan_kemampuan(theta, mean_theta, std_theta):
    # Jika sekelompok peserta memiliki theta yang homogen (std sangat kecil/0), beri batas default
    if std_theta < 1e-4:
        std_theta = 1.0
        
    batas_bawah = mean_theta - (0.5 * std_theta)
    batas_atas = mean_theta + (0.5 * std_theta)
    
    if theta < batas_bawah: 
        return "Lemah"
    elif batas_bawah <= theta <= batas_atas: 
        return "Normal"
    else: 
        return "Kuat"


# --- ENDPOINT UTAMA ANALYSIS IRT ---
@app.post("/api/analisis-irt/{id_paket}")
async def jalankan_analisis_irt(id_paket: int):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    try:
        # 1. AMBIL INFORMASI PAKET & CEK APAKAH ADA
        cursor.execute("SELECT nama_paket, tipe_soal FROM paket_soal WHERE id_paket = %s", (id_paket,))
        paket_info = cursor.fetchone()

        if not paket_info:
            raise HTTPException(status_code=404, detail=f"Paket ID {id_paket} tidak ditemukan.")

        TIPE_PAKET = paket_info['tipe_soal']

        # 2. CEK APAKAH SUDAH PERNAH DIOLAH
        cursor.execute("SELECT id_irt_peserta FROM irt_peserta WHERE paket_id = %s LIMIT 1", (id_paket,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail=f"Paket ID {id_paket} sudah pernah diolah.")

        # 3. AMBIL DATA JAWABAN DARI DATABASE
        query = """
        SELECT 
            u.user_id,
            t.id_topik,
            t.nama_topik,
            jp.soal_id,
            jp.is_benar
        FROM jawaban_peserta jp
        JOIN ujian_peserta u ON jp.ujian_id = u.id_ujian
        JOIN soal s ON jp.soal_id = s.id_soal
        JOIN topik t ON s.topik_id = t.id_topik
        WHERE t.paket_id = %s
        ORDER BY u.user_id ASC, jp.soal_id ASC
        """
        cursor.execute(query, (id_paket,))
        rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=422, detail="Data jawaban tidak ditemukan untuk ID Paket tersebut.")

        df = pd.DataFrame(rows)
        daftar_topik = df['id_topik'].unique()

        # 4. ANALISIS PER TOPIK
        for id_topik in daftar_topik:
            df_t = df[df['id_topik'] == id_topik]
            nama_topik = df_t['nama_topik'].iloc[0]
            
            # Pivot data: baris = user_id, kolom = soal_id, value = is_benar
            pv_t = df_t.pivot(index='user_id', columns='soal_id', values='is_benar').fillna(0)
            ds_t = pv_t.values.T.astype(int)  # Matrix item response
            
            n_peserta_topik, n_soal_topik = pv_t.shape
            if n_soal_topik < 2:
                continue

            # Pemodelan IRT berdasarkan jumlah peserta
            if n_peserta_topik < 100:
                model_t = "1PL"
                res_t = rasch_mml(ds_t)
                diff_t = res_t['Difficulty']
                disc_t = np.ones(len(diff_t))
                guess_t = np.zeros(len(diff_t))
                th_t = ability_eap(ds_t, diff_t, disc_t)
            elif n_peserta_topik < 1000:
                model_t = "2PL"
                res_t = twopl_mml(ds_t)
                diff_t = res_t['Difficulty']
                disc_t = res_t['Discrimination']
                guess_t = np.zeros(len(diff_t))
                th_t = ability_eap(ds_t, diff_t, disc_t)
            else:
                model_t = "3PL"
                res_t = threepl_mml(ds_t)
                diff_t = res_t['Difficulty']
                disc_t = res_t['Discrimination']
                guess_t = res_t['Guessing']
                th_t = ability_3pl_eap(ds_t, diff_t, disc_t, guess_t)

            # --- Simpan Parameter Soal ke irt_soal ---
            sql_soal = """INSERT INTO irt_soal 
                        (paket_id, soal_id, tingkat_kesulitan, kategori_kesulitan, daya_pembeda, kategori_daya_pembeda, tebakan, model_irt) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            
            for idx_s, id_soal in enumerate(pv_t.columns):
                val_soal = (
                    id_paket, int(id_soal), 
                    float(diff_t[idx_s]), tentukan_kategori_soal(diff_t[idx_s]), 
                    float(disc_t[idx_s]), tentukan_kategori_daya_pembeda(disc_t[idx_s]), 
                    float(guess_t[idx_s]), model_t
                )
                cursor.execute(sql_soal, val_soal)

            # HITUNG PARAMETER DISTRIBUSI SECARA DINAMIS PER TOPIK
            mean_theta_topik = float(np.mean(th_t))
            std_theta_topik = float(np.std(th_t))

            # --- Simpan Kemampuan Peserta ke irt_peserta ---
            sql_peserta = """INSERT INTO irt_peserta 
                            (paket_id, user_id, topik_id, theta, skor, akurasi, status_kemampuan) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            
            for i, user_id in enumerate(pv_t.index):
                theta_individu = th_t[i]
                jawaban_user = pv_t.iloc[i].values

                total_benar = int(np.sum(jawaban_user))
                total_soal = len(jawaban_user)
                benar_semua = (total_benar == total_soal)
                
                skor_raw = hitung_skor_skala(theta_individu, TIPE_PAKET, benar_semua=benar_semua)
                skor_final = round(skor_raw, 2)
                
                prob_raw = np.mean(hitung_probabilitas_soal(theta_individu, diff_t, disc_t, guess_t))
                akurasi_persen = round(float(prob_raw * 100), 2)
                
                val_peserta = (
                    id_paket, int(user_id), int(id_topik),
                    float(theta_individu), skor_final, akurasi_persen,
                    kategorikan_kemampuan(theta_individu, mean_theta_topik, std_theta_topik)
                )
                cursor.execute(sql_peserta, val_peserta)

        # ======================================================
        # SIMPAN NOTIFIKASI
        # ======================================================
        sql_notifikasi = """
        INSERT INTO notifikasi
        (paket_id, nama_paket_snapshot, tipe)
        VALUES (%s, %s, %s)
        """

        cursor.execute(
            sql_notifikasi,
            (
                id_paket,
                paket_info["nama_paket"],
                "irt"
            )
        )

        # Commit perubahan data ke database
        db.commit()
        return {"status": "success", "message": f"Analisis IRT berhasil diproses dan disimpan."}

    except mysql.connector.Error as db_err:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(db_err)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Sistem error: {str(e)}")
    finally:
        cursor.close()
        db.close()

# Untuk menjalankan via command line: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)