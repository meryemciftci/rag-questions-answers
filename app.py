import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RAG Soru-Cevap",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 RAG Soru-Cevap Sistemi")
st.markdown("---")

# Session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_..."
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ API Key OK")
    
    st.markdown("---")
    
    if groq_api_key:
        st.subheader("📄 PDF Yükle")
        
        st.caption("Maksimum: 500MB | Önerilen: 50MB altı")
        
        uploaded_file = st.file_uploader("PDF seçin", type=['pdf'])
        
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📊 Dosya boyutu: {file_size_mb:.2f} MB")
            
            if file_size_mb > 50:
                st.warning("⚠️ Büyük dosya! İşlem biraz uzun sürebilir.")
            
            if st.button("🚀 İşle", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    from langchain_community.vectorstores import FAISS
                    from langchain_groq import ChatGroq
                    from langchain.chains import RetrievalQA
                    from langchain.prompts import PromptTemplate
                    
                    # PDF kaydet
                    status.text("1/5 PDF kaydediliyor...")
                    progress.progress(20)
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # PDF oku
                    status.text("2/5 PDF okunuyor...")
                    progress.progress(40)
                    loader = PyPDFLoader("temp.pdf")
                    documents = loader.load()
                    st.info(f"✅ {len(documents)} sayfa okundu")
                    
                    # Parçala
                    status.text("3/5 Parçalanıyor...")
                    progress.progress(60)
                    
                    if file_size_mb > 20:
                        chunk_size = 1200
                        chunk_overlap = 150
                    elif file_size_mb > 10:
                        chunk_size = 1000
                        chunk_overlap = 100
                    else:
                        chunk_size = 800
                        chunk_overlap = 100
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = splitter.split_documents(documents)
                    st.info(f"✅ {len(chunks)} parça oluşturuldu (chunk size: {chunk_size})")
                    
                    # BURADA DEĞİŞİKLİK - Sentence Transformers dene, yoksa TF-IDF
                    status.text("4/5 Embedding modeli yükleniyor...")
                    progress.progress(75)
                    
                    try:
                        from sentence_transformers import SentenceTransformer
                        from langchain.embeddings.base import Embeddings
                        
                        class SentenceTransformerEmbeddings(Embeddings):
                            _model = None
                            
                            def __init__(self, model_name="all-MiniLM-L6-v2"):
                                if SentenceTransformerEmbeddings._model is None:
                                    SentenceTransformerEmbeddings._model = SentenceTransformer(
                                        model_name,
                                        device='cpu'
                                    )
                                self.model = SentenceTransformerEmbeddings._model
                            
                            def embed_documents(self, texts):
                                embeddings = self.model.encode(
                                    texts, 
                                    show_progress_bar=False,
                                    convert_to_numpy=True
                                )
                                return embeddings.tolist()
                            
                            def embed_query(self, text):
                                embedding = self.model.encode(
                                    [text], 
                                    show_progress_bar=False,
                                    convert_to_numpy=True
                                )
                                return embedding[0].tolist()
                        
                        embeddings = SentenceTransformerEmbeddings()
                        st.info("✅ Sentence Transformers (semantic search)")
                        
                    except Exception as e:
                        st.warning("Sentence-transformers yüklenemedi, TF-IDF kullanılıyor")
                        
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from langchain.embeddings.base import Embeddings
                        
                        class SimpleEmbeddings(Embeddings):
                            def __init__(self):
                                self.vectorizer = TfidfVectorizer(max_features=384)
                                self.fitted = False
                            
                            def embed_documents(self, texts):
                                if not self.fitted:
                                    self.vectorizer.fit(texts)
                                    self.fitted = True
                                vectors = self.vectorizer.transform(texts).toarray()
                                return vectors.tolist()
                            
                            def embed_query(self, text):
                                if not self.fitted:
                                    return [0.0] * 384
                                vector = self.vectorizer.transform([text]).toarray()
                                return vector[0].tolist()
                        
                        embeddings = SimpleEmbeddings()
                        st.info("✅ TF-IDF (yedek)")
                    
                    # Vector store
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vectorstore = vectorstore
                    
                    # LLM
                    status.text("5/5 LLM hazırlanıyor...")
                    progress.progress(90)
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.2
                    )
                    
                    prompt = PromptTemplate(
                        template="""Sen bir uzman asistansın. Sadece verilen bağlam içindeki bilgileri kullan.

BAĞLAM:
{context}

SORU: {question}

KURALLAR:
- Sadece bağlamdaki bilgileri kullan
- Bilgi yoksa "Bu bilgi dokümanda bulunmuyor" de
- Türkçe, net ve anlaşılır cevap ver
- Kaynak belirt

CEVAP:""",
                        input_variables=["context", "question"]
                    )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt}
                    )
                    st.session_state.qa_chain = qa_chain
                    
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")
                    
                    progress.progress(100)
                    status.empty()
                    progress.empty()
                    
                    st.success(f"✅ Hazır! {len(chunks)} parça")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Hata: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    if st.session_state.qa_chain:
        st.markdown("---")
        st.success("✅ Sistem Aktif")

# Ana alan
if not groq_api_key:
    st.info("👈 Groq API Key girin")
elif not st.session_state.qa_chain:
    st.info("👈 PDF yükleyip işleyin")
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Soru")
        
        question = st.text_area(
            "Sorunuz:",
            height=100,
            placeholder="Bu döküman ne hakkında?"
        )
        
        if st.button("🔍 Cevap", type="primary"):
            if question:
                with st.spinner("Cevap hazırlanıyor..."):
                    try:
                        result = st.session_state.qa_chain.invoke({"query": question})
                        
                        st.markdown("### 💡 Cevap")
                        st.success(result['result'])
                        
                        with st.expander("📄 Kaynaklar"):
                            for i, doc in enumerate(result['source_documents'], 1):
                                st.text(f"Kaynak {i}:\n{doc.page_content[:200]}...")
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"Hata: {e}")
            else:
                st.warning("Soru yazın")
    
    with col2:
        st.subheader("📝 Örnekler")
        examples = [
            "Ne hakkında?",
            "Özet çıkar",
            "Ana konular?"
        ]
        for ex in examples:
            st.button(ex, key=ex)

st.markdown("---")
st.caption("RAG Sistemi")