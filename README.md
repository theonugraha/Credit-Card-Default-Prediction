[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/E-jHmiR4)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11524016&assignment_repo_type=AssignmentRepo)
# Milestones 1

*Milestones ini dibuat guna mengevaluasi pembelajaran pada Hacktiv8 Data Science Fulltime Program khususnya pada Phase 1 dalam konsep Supervised Learning.*

---

## Assignment Objectives

*Milestone 1* ini dibuat guna mengevaluasi konsep Supervised Learning sebagai berikut:

- Mampu memperoleh data menggunakan BigQuery.

- Mampu memahami konsep Supervised Learning.

- Mampu mempersiapkan data untuk digunakan dalam model Supervised Learning.

- Mampu mengimplementasikan Supervised Learning dengan data yang diberikan.

- Mampu melakukan Hyperparameter Tuning dan Model Improvement.

---

## Dataset

```{attention}
Perhatikan petunjuk penggunaan dataset!
```

1. Buka [Google Cloud Platform](https://console.cloud.google.com/), masuk ke BigQuery, lalu buka tab `bigquery-public-data` atau klik link [berikut](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=samples&page=dataset&_ga=2.245085957.1471931019.1642739417-486643658.1638156099) atau link [berikut](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=ml_datasets&t=credit_card_default&page=table) untuk langsung menuju ke dataset.

2. Gunakan dataset `ml_datasets` dari database bernama `credit_card_default`.

3. Buatlah query dengan kriteria sebagai berikut:
   - Pilih **HANYA** kolom `limit_balance, sex, education_level, marital_status, age, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6, bill_amt_1, bill_amt_2, bill_amt_3, bill_amt_4, bill_amt_5, bill_amt_6, pay_amt_1, pay_amt_2, pay_amt_3, pay_amt_4, pay_amt_5, pay_amt_6, default_payment_next_month`.
   
   - Pada kolom yang diambil diatas, terdapat beberapa kolom bertipe `STRING`. Pada saat pengambilan data dengan menggunakan perintah `SELECT`, lakukan konversi tipe data kolom-kolom bertipe `STRING` ke tipe numerik dengan panduan dibawah ini : 
     | Kolom | Tipe Data Awal | Tipe Data Akhir |
     | --- | --- | --- |
     | `sex` | STRING | INT |
     | `education_level` | STRING | INT |
     | `marital_status` | STRING | INT |
     | `pay_5` | STRING | FLOAT |
     | `pay_6` | STRING | FLOAT |
     | `default_payment_next_month` | STRING | INT |
   
   - Konversi tipe data harus dilakukan dalam sintaks yang sama saat melakukan query ke Google BigQuery.
   
   - Kolom diatas hanya digunakan sebagai dataset awal. Silakan lakukan Feature Selection di-notebook setelah dataset dibuat.
   
   - Limit jumlah data menjadi sebanyak `nomor batch dikali dengan tahun lahir kalian`. ex: Batch 10 dan lahir tahun 1995, 10 x 1995 = 19950.

4. Simpan dataset dalam bentuk csv, dengan nama `h8dsft_P1M1_<nama-students>.csv`.

5. Salin query yang telah dibuat di Google Cloud Platform, tulislah pada bagian atas notebook !

6. Tampilkan `10 data pertama` dan `10 data terakhir` dari dataset pada notebook !

---

## Problems

Buatlah model Classification untuk memprediksi `default_payment_next_month` menggunakan dataset yang sudah kalian simpan.

---

## Conceptual Problems

_Jawab pertanyaan berikut:_

1. Apa yang dimaksud dengan `criterion` pada Decision Tree ? Jelaskan criterion yang kalian pakai dalam kasus ini !

2. Jelaskan apa yang dimaksud dengan `pruning` pada Tree-based model (alasan, definisi, jenis, dll) !

3. Bagaimana cara memilih `K` yang optimal pada KNN ?

4. Jelaskan apa yang dimaksud dengan `Cross Validation` !

5. Apa yang dimaksud dengan metrics-metrics berikut : `Accuracy`, `Precision`, `Recall`, `F1 Score`, dan kapan waktu yang tepat untuk menggunakannya ?

---

## Assignment Instructions

_Milestones 1_ dikerjakan dalam format _notebook_ dengen beberapa **kriteria wajib** di bawah ini:

1. Machine learning framework yang digunakan adalah _Scikit-Learn_.

2. Ada penggunaan library visualisasi, seperti _matplotlib_, _seaborn_, atau yang lain.

3. Isi _notebook_ harus mengikuti _outline_ di bawah ini:
   1. Perkenalan
      > Bab pengenalan harus diisi dengan identitas, **query yang telah kalian buat pada Google Cloud Platform!**, dan _objective_ yang ingin dicapai.

   2. Query SQL
      > Tulis query yang telah dibuat untuk mengambil data dari Google Cloud Platform di bagian ini.

   3. Import Libraries
      > _Cell_ pertama pada _notebook_ **harus berisi dan hanya berisi** semua _library_ yang digunakan dalam _project_.

   4. Data Loading
      > Bagian ini berisi proses penyiapan data sebelum dilakukan eksplorasi data lebih lanjut. Proses Data Loading dapat berupa memberi nama baru untuk setiap kolom, mengecek ukuran dataset, dll.

   5. Exploratory Data Analysis (EDA)
      > Bagian ini berisi eksplorasi data pada dataset diatas dengan menggunakan query, grouping, visualisasi sederhana, dan lain sebagainya.

   6. Feature Engineering
      > Bagian ini berisi proses penyiapan data untuk proses pelatihan model, seperti pembagian data menjadi train-test, transformasi data (normalisasi, encoding, dll.), dan proses-proses lain yang dibutuhkan.

   7. Model Definition
      > Bagian ini berisi cell untuk mendefinisikan model. Jelaskan alasan menggunakan suatu algoritma/model, hyperparameter yang dipakai, jenis penggunaan metrics yang dipakai, dan hal lain yang terkait dengan model.

   8. Model Training
      > Cell pada bagian ini hanya berisi code untuk melatih model dan output yang dihasilkan. Lakukan beberapa kali proses training dengan hyperparameter yang berbeda untuk melihat hasil yang didapatkan. Analisis dan narasikan hasil ini pada bagian Model Evaluation.

   9. Model Evaluation
      > Pada bagian ini, dilakukan evaluasi model yang harus menunjukkan bagaimana performa model berdasarkan metrics yang dipilih. Hal ini harus dibuktikan dengan visualisasi tren performa dan/atau tingkat kesalahan model. **Lakukan analisis terkait dengan hasil pada model dan tuliskan hasil analisisnya**.

   10. Model Saving
       > Pada bagian ini, dilakukan proses penyimpanan model dan file-file lain yang terkait dengan hasil proses pembuatan model. **Dengan melihat hasil Model Evaluation, pilihlah model terbaik untuk disimpan. Model terbaik ini akan digunakan kembali dalam melakukan Model Inference dan Model Deployment.**
   
   11. Model Inference
       > Model yang sudah dilatih akan dicoba pada data yang bukan termasuk ke dalam train-set ataupun test-set. Data ini harus dalam format yang asli, bukan data yang sudah di-scaled. Gunakan model terbaik berdasarkan hasil Model Evaluation.

   12. Pengambilan Kesimpulan
       > Pada bagian terakhir ini, **harus berisi** kesimpulan yang mencerminkan hasil yang didapat dengan _objective_ yang sudah ditulis di bagian pengenalan.

4. *Notebook* harus diupload dalam akun GitHub masing-masing student untuk selanjutnya dinilai.

5. Presentasikan model yang telah dibuat pada P1W3D5AM.

---

## Assignment Submission

- Simpan assignment pada sesi ini dengan nama `h8dsft_P1M1_<nama-student>.ipynb`, misal `h8dsft_P1M1_raka_ardhi.ipynb`.

- Push Assigment yang telah Anda buat ke akun Github Classroom Anda masing-masing.

---

## Assignment Rubrics

### Code Review

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| SQL | Mampu melakukan query data dengan kriteria yang telah diberikan | 15 pts |
| Feature Engineering | Mampu melakukan preprocessing dataset sebelum melakukan proses modeling (split data, normalisasi, encoding, dll) | 35 pts |
| Logistic Regression | Mengimplementasikan Logistic Regression dan menentukan hyperparameter yang tepat dengan Scikit-Learn | 5 pts |
| SVM | Mengimplementasikan SVM dan menentukan hyperparameter yang tepat dengan Scikit-Learn | 5 pts |
| Decision Tree | Mengimplementasikan Decision Tree dan menentukan hyperparameter yang tepat dengan Scikit-Learn | 5 pts |
| Random Forest | Mengimplementasikan Random Forest dan menentukan hyperparameter yang tepat dengan Scikit-Learn | 5 pts |
| KNN | Mengimplementasikan KNN dan menentukan hyperparameter yang tepat dengan Scikit-Learn | 5 pts |
| Naive Bayes | Mengimplementasikan Naive Bayes dan menentukan hyperparameter yang tepat dengan Scikit-Learn | 5 pts |
| Other Algorithm | Mengimplementasikan algoritma lain selain yang tersebut diatas dan menentukan hyperparameter yang tepat | 5 pts |
| Cross Validation | Mengimplementasikan Cross Validation dengan Scikit-Learn | 25 pts |
| Hyperparameter Tuning | Mengimplementasikan Hyperparameter Tuning dengan Scikit-Learn | 20 pts |
| Model Inference | Mencoba model yang telah dibuat dengan data baru | 10 pts |
| Runs Perfectly | Kode berjalan tanpa ada error. Seluruh kode berfungsi dan dibuat dengan benar | 10 pts |

```
Pada rubrik Milestone 1 diatas terdapat point Cross Validation dan Hyperparameter Tuning (GridSearchCV, RandomSearchCV, dll). 
Kedua hal yang dimaksud ini adalah dua hal yang berbeda bukan satu kesatuan. Petunjuk : 

1. Lakukan model training dengan menggunakan parameter default (baseline model) dari setiap algoritma yang diminta.
2. Kemudian, gunakan `cross_val_score` atau `cross_validate` untuk mencari nilai performansi `mean` dan `std` dari setiap model. 
3. Pilih agoritma yang terbaik dari hasil poin 2.
4. Lakukan Hyperparameter Tuning pada algoritma terbaik (berdasarkan poin 2) dengan menggunakan GridSearchCV, RandomSearchCV, dll.
5. Bandingkan performansi antara sebelum dan sesudah dilakukan Hyperparameter Tuning.
```

### Concepts

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Classifications | Mampu menjawab pertanyaan dengan singkat, jelas, dan padat serta sesuai dengan konsep dan logika yang ada mengenai Conceptual Problems (10 each number) | 50 pts |

### Readability

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Tertata Dengan Baik | Semua baris kode terdokumentasi dengan baik dengan Markdown untuk penjelasan kode | 15 pts |

```
Kriteria tertata dengan baik diantaranya adalah: 

1. Terdapat section Perkenalan yang jelas dan lengkap terkait masalah dan latar belakang masalah yang akan diselesaikan.
2. Tidak menyalin markdown dari tugas lain.
3. Import library rapih (terdapat dalam 1 cell dan tidak ada unused libs).
4. Pemakaian fungsi markdown yang optimal (Heading, text formating, dll).
5. Terdapat komentar pada setiap baris kode.
6. Adanya pemisah yang jelas antar section, dll.
7. Tidak adanya typo.
```

### Analysis

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Model Analysis | Menganalisa informasi dari model yang telah dibuat | 35 pts |
| Overall Analysis | Menarik informasi/kesimpulan dari keseluruhan kegiatan yang dilakukan | 20 pts |

```
Contoh kriteria analisa yang baik diantaranya adalah: 

1. Terdapat penjelasan macam-macam hasil metric evaluasi dan interpretasinya terhadap kasus yang diselesaikan.
2. Dapat menjelaskan KELEBIHAN dan KELEMAHAN dari model yang dibuat DENGAN KAITANNYA DENGAN DOMAIN BUSINESS YANG DIHADAPI yang dibuktikan dengan eksplorasi sederhana (grafik, plot, teori, dll).
3. Dapat memberikan statement untuk improvement selanjutnya dari model yang dibuat. 
4. Dapat menyebutkan insight yang dapat diambil setelah proses EDA, dll.
```

---

```
Total Points : 270
```

---

## Notes

- **Deadline : P1W3D3 pukul 23:59 WIB.**

- **Keterlambatan pengumpulan tugas mengakibatkan skor Milestone 1 menjadi 0.**
