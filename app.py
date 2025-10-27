import streamlit as st
import os
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RAG Soru-Cevap",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .source-box {
        background-color: #fff9e6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 RAG Soru-Cevap Sistemi")
st.markdown("*PDF dokümanlarınızı yükleyin ve sorular sorun*")
st.markdown("---")

# Session state initialization
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Groq API anahtarınızı girin"
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ API Key ayarlandı")
    
    st.markdown("---")
    
    # PDF Upload Section
    if groq_api_key:
        st.subheader("📄 PDF Yükle")
        
        st.caption("Maksimum: 500MB | Önerilen: 50MB altı")
        
        uploaded_file = st.file_uploader("PDF seçin", type=['pdf'])
        
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📊 Dosya: {uploaded_file.name}")
            st.info(f"📏 Boyut: {file_size_mb:.2f} MB")
            
            if file_size_mb > 50:
                st.warning("⚠️ Büyük dosya! İşlem uzun sürebilir.")
            
            process_btn = st.button("🚀 PDF'i İşle", type="primary", use_container_width=True)
            
            if process_btn:
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    from langchain_community.vectorstores import FAISS
                    from langchain_groq import ChatGroq
                    from langchain.chains import RetrievalQA
                    from langchain.prompts import PromptTemplate
                    
                    # Step 1: Save PDF
                    status.text("1/5 📥 PDF kaydediliyor...")
                    progress.progress(20)
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Step 2: Load PDF
                    status.text("2/5 📖 PDF okunuyor...")
                    progress.progress(40)
                    loader = PyPDFLoader("temp.pdf")
                    documents = loader.load()
                    st.info(f"✅ {len(documents)} sayfa okundu")
                    
                    # Step 3: Split documents
                    status.text("3/5 ✂️ Parçalara ayrılıyor...")
                    progress.progress(60)
                    
                    # Dynamic chunk size based on file size
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
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = splitter.split_documents(documents)
                    st.info(f"✅ {len(chunks)} parça oluşturuldu")
                    
                    # Step 4: Create embeddings
                    status.text("4/5 🧠 Embedding modeli yükleniyor...")
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
                        st.success("✅ Sentence Transformers yüklendi (Semantic Search)")
                        
                    except Exception as e:
                        st.warning("⚠️ Sentence-transformers yüklenemedi, TF-IDF kullanılıyor")
                        
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
                        st.info("✅ TF-IDF kullanılıyor")
                    
                    # Create vector store
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vectorstore = vectorstore
                    
                    # Step 5: Setup LLM and QA Chain
                    status.text("5/5 🤖 LLM hazırlanıyor...")
                    progress.progress(90)
                    
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.2
                    )
                    
                    prompt = PromptTemplate(
                        template="""Sen bir uzman asistansın. Sadece verilen bağlam içindeki bilgileri kullanarak cevap ver.

BAĞLAM:
{context}

SORU: {question}

KURALLAR:
- Sadece bağlamdaki bilgileri kullan
- Bilgi yoksa "Bu bilgi dokümanda bulunmuyor" de
- Türkçe, net ve anlaşılır cevap ver
- Mümkünse sayfa numarası belirt

CEVAP:""",
                        input_variables=["context", "question"]
                    )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt}
                    )
                    st.session_state.qa_chain = qa_chain
                    st.session_state.pdf_name = uploaded_file.name
                    
                    # Cleanup
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")
                    
                    progress.progress(100)
                    status.empty()
                    progress.empty()
                    
                    st.success(f"✅ Sistem hazır! {len(chunks)} parça oluşturuldu")
                    st.balloons()
                    
                    # Clear chat history on new PDF
                    st.session_state.chat_history = []
                    
                except Exception as e:
                    st.error(f"❌ Hata: {str(e)}")
                    import traceback
                    with st.expander("Hata detayları"):
                        st.code(traceback.format_exc())
    
    # System status
    st.markdown("---")
    if st.session_state.qa_chain:
        st.success("✅ Sistem Aktif")
        if st.session_state.pdf_name:
            st.info(f"📄 {st.session_state.pdf_name}")
        
        # Clear chat button
        if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Stats
        if st.session_state.chat_history:
            st.markdown("---")
            st.metric("Soru Sayısı", len(st.session_state.chat_history))

# Main area
if not groq_api_key:
    st.info("👈 Lütfen Groq API Key girin")
    st.markdown("""
    ### Nasıl API Key alınır?
    1. [console.groq.com](https://console.groq.com) adresine gidin
    2. Ücretsiz hesap oluşturun
    3. API Keys bölümünden yeni key oluşturun
    """)
    
elif not st.session_state.qa_chain:
    st.info("👈 Lütfen PDF yükleyip işleyin")
    st.markdown("""
    ### PDF formatında belge yükleyin
    - Maksimum dosya boyutu: 500MB
    - Önerilen: 50MB altı dosyalar daha hızlı işlenir
    """)
    
else:
    # Chat interface
    st.subheader("💬 Sohbet")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(
                    f"""<div class="user-message">
                    <strong>👤 Sen:</strong><br>{chat['question']}
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Assistant message
                st.markdown(
                    f"""<div class="assistant-message">
                    <strong>🤖 Asistan:</strong><br>{chat['answer']}
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Sources in expander
                if chat.get('sources'):
                    with st.expander(f"📚 Kaynaklar ({len(chat['sources'])} adet)"):
                        for j, source in enumerate(chat['sources'], 1):
                            page_num = source.metadata.get('page', 'N/A')
                            st.markdown(f"""
                            <div class="source-box">
                            <strong>Kaynak {j} (Sayfa: {page_num})</strong><br>
                            {source.page_content[:300]}...
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
        else:
            st.info("👋 Merhaba! PDF'iniz hakkında soru sorabilirsiniz.")
    
    # Question input area
    st.markdown("### 💭 Yeni Soru")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Sorunuz:",
            placeholder="Bu döküman ne hakkında?",
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🔍 Sor", type="primary", use_container_width=True)
    
    # Example questions
    st.markdown("**💡 Örnek Sorular:**")
    example_cols = st.columns(3)
    
    examples = [
        "Bu döküman ne hakkında?",
        "Ana konuları özetle",
        "En önemli noktalar neler?"
    ]
    
    example_clicked = None
    for idx, ex in enumerate(examples):
        with example_cols[idx]:
            if st.button(ex, key=f"example_{idx}", use_container_width=True):
                example_clicked = ex
    
    # Handle question (from input or example)
    query = None
    if ask_button and question:
        query = question
    elif example_clicked:
        query = example_clicked
    
    if query:
        with st.spinner("🤔 Düşünüyorum..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": query})
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': query,
                    'answer': result['result'],
                    'sources': result.get('source_documents', []),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Rerun to show new message
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Hata oluştu: {str(e)}")

# Footer
st.markdown("---")
st.caption("📚 RAG Soru-Cevap Sistemi | Powered by Groq + LangChain")