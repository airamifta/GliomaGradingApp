# 🧠 Glioma Grading App

Aplikasi untuk klasifikasi tingkat keganasan glioma (LGG vs GBM) menggunakan kombinasi machine learning, deep learning, dan multi-strategy feature selection.

🌐 Demo App (Streamlit):  
https://gliomagradingapp-14579.streamlit.app/

## 🚀 Cara Pakai

1. Buka link Streamlit di atas  
2. Masukkan nilai fitur klinis/molekuler pada form input  
3. Klik tombol "Predict"  
4. Hasil klasifikasi (LGG atau GBM) akan ditampilkan  

## 🤖 Model

- Logistic Regression → baseline model untuk klasifikasi linear  
- Random Forest → menangkap hubungan non-linear dan mengurangi overfitting  
- MLP (Multilayer Perceptron) → memodelkan pola kompleks pada data tabular  
- CNN-1D + PSO → model terbaik, memanfaatkan pola lokal pada fitur dan optimasi seleksi fitur menggunakan Particle Swarm Optimization  

## 📊 Hasil

Model terbaik: CNN-1D + PSO  
- Accuracy: 88.1%  
- F1-score: 88.2%  
- AUC: 0.92  

## 🖼️ Tampilan Aplikasi

![App Screenshot](https://raw.githubusercontent.com/airamifta/GliomaGradingApp/main/Screenshot%202026-03-26%20200239.png)
