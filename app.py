import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Config upload folder
UPLOAD_FOLDER = 'static/gambar'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load dataset awal dari Excel
DATA_FILE = 'data-kuliner.xlsx'

def load_data():
    df = pd.read_excel(DATA_FILE)

    # Hanya kolom teks yang diisi NaN-nya dengan ''
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('').astype(str)

    # Buat id unik jika belum ada
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)

    return df


# Inisialisasi stopword remover dan stemmer
factory_stopwords = StopWordRemoverFactory()
stopword_remover = factory_stopwords.create_stop_word_remover()

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

def preprocess(text):
    text = str(text).lower()  # case folding
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = stopword_remover.remove(text)  # stopword removal
    text = stemmer.stem(text)  # stemming
    return text

# Simpan data ke Excel (untuk update admin)
def save_data(df):
    df.to_excel(DATA_FILE, index=False)
    
# Cek ekstensi file gambar
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Fungsi buat dokumen teks untuk TF-IDF dari beberapa kolom yang jadi fitur (jenis tempat, lokasi, menu, layanan, dll)
def create_corpus(df):
    corpus = []
    for _, row in df.iterrows():
        # Gabungkan string fitur untuk tiap tempat
        combined = ' '.join([
            str(row.get('namatempat', '')),
            str(row.get('rating', '')),
            str(row.get('jenis', '')),
            str(row.get('harga', '')),
            str(row.get('lokasi', '')),
            str(row.get('jam_operasi', '')),
            str(row.get('layanan1', '')),
            str(row.get('layanan2', '')),
            str(row.get('menu', ''))
        ])
        # Panggil fungsi preprocess di sini
        corpus.append(preprocess(combined))
    return corpus


df = load_data()
corpus = create_corpus(df)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
feature_names = tfidf.get_feature_names_out()


# ================= ROUTES USER =================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/rekomendasi_nama', methods=['GET', 'POST'])
def rekomendasi_nama():
    if request.method == 'POST':
        nama_tempat = request.form.get('nama', '').strip()

        try:
            jumlah = int(request.form.get('jumlah', 1))
            if jumlah < 1 or jumlah > 10:
                flash('Jumlah rekomendasi harus antara 1 sampai 10', 'error')
                return redirect(url_for('rekomendasi_nama'))
        except:
            flash('Jumlah rekomendasi harus angka', 'error')
            return redirect(url_for('rekomendasi_nama'))

        # Load data
        df = load_data()
        corpus = create_corpus(df)

        # Buat TF-IDF matrix
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(corpus)

        # Cari indeks tempat sesuai input
        matches = df[df['namatempat'].str.lower() == nama_tempat.lower()]
        if matches.empty:
            return render_template('rekomendasi_nama.html', error=True)

        idx_target = matches.index[0]

        # Hitung similarity dengan semua tempat
        cosine_similarities = cosine_similarity(tfidf_matrix[idx_target], tfidf_matrix).flatten()

        # Urutkan dari skor tertinggi selain diri sendiri
        similar_indices = cosine_similarities.argsort()[::-1]
        similar_indices = [i for i in similar_indices if i][:jumlah]
        

        # Ambil data hasil rekomendasi
        hasil_df = df.iloc[similar_indices].copy()
        hasil_rekomendasi = hasil_df.to_dict(orient='records')

        return render_template('hasil_rekomendasi_nama.html',
                               rekomendasi=hasil_rekomendasi,
                               tempat_asli=nama_tempat)

    return render_template('rekomendasi_nama.html', error=False)



@app.route('/rekomendasi_keyword', methods=['GET', 'POST'])
def rekomendasi_keyword():
    if request.method == 'POST':
        keywords = request.form.get('keywords', '').strip().lower()
        try:
            jumlah = int(request.form.get('jumlah', 1))
            if jumlah < 1 or jumlah > 10:
                flash('Jumlah rekomendasi harus antara 1 sampai 10', 'error')
                return redirect(url_for('rekomendasi_keyword'))
        except:
            flash('Jumlah rekomendasi harus angka', 'error')
            return redirect(url_for('rekomendasi_keyword'))

        # Load data dan buat corpus
        df = load_data()
        corpus = create_corpus(df)

        # TF-IDF Vectorization untuk corpus + query keyword
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(corpus + [keywords])  # keyword sebagai dokumen terakhir

        # Hitung similarity antara keyword dan seluruh dokumen corpus
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        # Ambil indeks dokumen dengan skor tertinggi
        top_indices = cosine_similarities.argsort()[::-1][:jumlah]
        scores = cosine_similarities[top_indices]

        # Ambil data hasil rekomendasi
        hasil_df = df.iloc[top_indices].copy()
        hasil_rekomendasi = hasil_df.to_dict(orient='records')

        return render_template('hasil_rekomendasi_keyword.html',
                               keywords=keywords,
                               rekomendasi=hasil_rekomendasi,
                               scores=scores)
    return render_template('rekomendasi_keyword.html')


@app.route('/data_kuliner')
def data_kuliner():
    query = request.args.get('q', '').strip().lower()
    df = load_data()

    if query:
        # Filter berdasarkan kolom 'namatempat' yang mengandung query (case-insensitive)
        df = df[df['namatempat'].str.lower().str.contains(query)]

    return render_template('data_kuliner.html', data=df.to_dict(orient='records'))



@app.route('/detail_kuliner/<int:id>')
def detail_kuliner(id):
    df = load_data()
    tempat = df[df['id'] == id]
    if tempat.empty:
        return "Data tidak ditemukan", 404
    tempat = tempat.iloc[0]
    return render_template('detail_kuliner.html', tempat=tempat)

# ================= ROUTES ADMIN =================

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        # Ganti sesuai username/password admin yang valid
        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Username atau password salah', 'error')
    return render_template('admin_login.html')

def admin_login_required(func):
    from functools import wraps
    @wraps(func)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return func(*args, **kwargs)
    return decorated

@app.route('/admin/logout')
@admin_login_required
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
@admin_login_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin/data')
@admin_login_required
def admin_data():
    df = load_data()
    return render_template('admin_data.html', data=df.to_dict(orient='records'))

@app.route('/admin/tambah', methods=['GET', 'POST'])
@admin_login_required
def admin_tambah():
    df = load_data()
    if request.method == 'POST':
        form = request.form.to_dict()
        new_id = df['id'].max() + 1 if not df.empty else 1

        # Buat dictionary data baru
        data_baru = {
            'id': new_id,
            'namatempat': form.get('namatempat', ''),
            'rating': form.get('rating', ''),
            'jenis': form.get('jenis', ''),
            'harga': form.get('harga', ''),
            'lokasi': form.get('lokasi', ''),
            'jam_operasi': form.get('jam_operasi', ''),
            'layanan1': form.get('layanan1', ''),
            'layanan2': form.get('layanan2', ''),
            'menu': form.get('menu', ''),
            'gambar': form.get('gambar', '')
        }

        # Tambahkan ke DataFrame dan simpan
        df = pd.concat([df, pd.DataFrame([data_baru])], ignore_index=True)
        save_data(df)
        flash('Data berhasil ditambahkan.')
        return redirect(url_for('admin_data'))

    return render_template('admin_tambah.html')

@app.route('/admin/edit/<int:id>', methods=['GET', 'POST'])
@admin_login_required
def admin_edit(id):
    df = load_data()
    tempat = df[df['id'] == id].to_dict(orient='records')[0]

    if request.method == 'POST':
        form = request.form.to_dict()
        idx = df[df['id'] == id].index[0]
        df.loc[idx, 'namatempat'] = form.get('namatempat', '')
        df.loc[idx, 'rating'] = form.get('rating', '')
        df.loc[idx, 'jenis'] = form.get('jenis', '')
        df.loc[idx, 'harga'] = form.get('harga', '')
        df.loc[idx, 'lokasi'] = form.get('lokasi', '')
        df.loc[idx, 'jam_operasi'] = form.get('jam_operasi', '')
        df.loc[idx, 'layanan1'] = form.get('layanan1', '')
        df.loc[idx, 'layanan2'] = form.get('layanan2', '')
        df.loc[idx, 'menu'] = form.get('menu', '')
        df.loc[idx, 'gambar'] = form.get('gambar', '')

        save_data(df)
        flash('Data berhasil diubah.')
        return redirect(url_for('admin_data'))

    return render_template('admin_edit.html', tempat=tempat)

@app.route('/admin/hapus/<int:id>', methods=['POST'])
@admin_login_required
def admin_hapus(id):
    df = load_data()

    # Pastikan data dengan id tersebut ada
    if id in df['id'].values:
        df = df[df['id'] != id]
        save_data(df)
        flash('Data berhasil dihapus.')
    else:
        flash('Data tidak ditemukan.')

    return redirect(url_for('admin_data'))



if __name__ == '__main__':
    app.run(debug=True)
