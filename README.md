📘 RAG Soru-Cevap Sistemi

PDF dokümanları üzerinde yapay zeka destekli soru-cevap sistemi. Groq API ve Sentence Transformers ile semantic search kullanarak doğru cevaplar üretir.

🚀 Özellikler
PDF yükleme ve işleme
Semantic search (Sentence Transformers + FAISS)
Groq API (Llama 3.3 70B) ile cevap üretimi
Modern Streamlit arayüzü
Kaynak gösterimi ve 500MB’a kadar PDF desteği
⚙️ Kurulum
git clone <repo-url>
cd rag-project
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
![Görsel](img width="1533" height="853" alt="image" src="https://github.com/user-attachments/assets/63cae1d5-ef24-44eb-8769-0a7db3503065" )


