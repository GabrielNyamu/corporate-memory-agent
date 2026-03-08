import streamlit as st
import os
import shutil
import base64
import re
from PIL import Image
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# --- 1. CONFIGURATION & CSS ---
st.set_page_config(page_title="Corporate Memory & Secretary", page_icon="", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 15px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #eee; }
    section[data-testid="stSidebar"] { background-color: #2c3e50; color: white; }
    .source-card { background-color: white; border-left: 5px solid #28a745; padding: 15px; margin-top: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
    st.markdown("###  Admin Panel")
    api_key = st.text_input(" Gemini API Key", type="password")
    
    st.divider()
    if st.button(" Wipe Memory"):
        if os.path.exists("./chroma_db"): shutil.rmtree("./chroma_db")
        if os.path.exists("temp_data"): shutil.rmtree("temp_data")
        st.session_state.clear()
        st.success("System wiped clean.")
        st.rerun()

# --- 3. DYNAMIC AI LOADER ---
@st.cache_resource
def load_ai_engine(api_key_input):
    genai.configure(api_key=api_key_input)
    
    # 1. Find the valid model name dynamically
    valid_model_name = None
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name or 'pro' in m.name:
                    valid_model_name = m.name
                    break
        if not valid_model_name:
             for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    valid_model_name = m.name
                    break
    except: return None
        
    if not valid_model_name: return None

    # 2. Setup LlamaIndex
    llm = Gemini(model=valid_model_name, api_key=api_key_input)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 3. Return the name so we can use it elsewhere
    return valid_model_name

# --- 4. MAIN APP LOGIC ---
st.title(" Corporate Memory System")

if not api_key:
    st.warning(" Please enter your API Key in the sidebar to start.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# Load Engine & GET THE MODEL NAME
active_model_name = load_ai_engine(api_key)

if not active_model_name:
    st.error(" Could not connect to Google Gemini. Please check your API Key.")
    st.stop()

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs([" Search Archives (RAG)", " Generate Minutes (AI Secretary)"])

# ==========================================
# TAB 1: SEARCH ARCHIVES (RAG)
# ==========================================
with tab1:
    if "last_uploaded_files" not in st.session_state: st.session_state.last_uploaded_files = []
    if "query_engine" not in st.session_state: st.session_state.query_engine = None
    if "messages" not in st.session_state: 
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about past decisions."}]

    with st.expander("📂 Document Management", expanded=not st.session_state.query_engine):
        uploaded_files = st.file_uploader("Drop PDF Minutes Here", type=['pdf'], accept_multiple_files=True, key="rag_upload")

        if uploaded_files:
            current_file_signature = sorted([(f.name, f.size) for f in uploaded_files])
            if st.session_state.last_uploaded_files != current_file_signature:
                with st.status(" Reading & Indexing...", expanded=True) as status:
                    st.session_state.query_engine = None
                    if os.path.exists("temp_data"): shutil.rmtree("temp_data")
                    os.makedirs("temp_data")
                    
                    for file in uploaded_files:
                        with open(f"temp_data/{file.name}", "wb") as f:
                            f.write(file.getbuffer())
                    
                    documents = SimpleDirectoryReader("temp_data").load_data()
                    db = chromadb.PersistentClient(path="./chroma_db")
                    try: db.delete_collection("corporate_memory")
                    except: pass
                    
                    chroma_collection = db.get_or_create_collection("corporate_memory")
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=vector_store))
                    st.session_state.query_engine = index.as_query_engine(similarity_top_k=10)
                    st.session_state.last_uploaded_files = current_file_signature
                    status.update(label=" Memory Updated!", state="complete", expanded=False)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Was the budget approved?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.query_engine:
            with st.chat_message("assistant"):
                with st.spinner("Locating exact page..."):
                    try:
                        citation_prompt = f"Question: '{prompt}'. Answer strictly from documents. Cite the specific Page Number in brackets like (Page 5)."
                        response = st.session_state.query_engine.query(citation_prompt)
                        st.markdown(response.response)
                        
                        match = re.search(r"Page\s+(\d+)", response.response, re.IGNORECASE)
                        target_page = match.group(1) if match else None
                        best_node = None
                        if response.source_nodes:
                            if target_page:
                                for node in response.source_nodes:
                                    if node.metadata.get('page_label') == target_page:
                                        best_node = node
                                        break
                                if not best_node: best_node = response.source_nodes[0]
                            else:
                                best_node = response.source_nodes[0]

                        if best_node:
                            file_name = best_node.metadata.get('file_name', 'Unknown')
                            page_label = best_node.metadata.get('page_label', '?')
                            link_html = ""
                            pdf_path = f"temp_data/{file_name}"
                            if os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode()
                                link_html = f'<a href="data:application/pdf;base64,{b64}#page={page_label}" target="_blank" style="text-decoration:none; color:#28a745; font-weight:bold; font-size:1.1em;">📄 Click to Open Document (Page {page_label}) ↗</a>'

                            st.markdown(f"""<div class="source-card"><h4 style="margin:0; color:#333;">🔍 Verified Source</h4><p style="color:#555; margin: 5px 0;">Found in <strong>{file_name}</strong> on <strong>Page {page_label}</strong>.</p><div style="margin-top:10px;">{link_html}</div></div>""", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": response.response})
                    except Exception as e:
                        st.error(f"Error: {e}")

# ==========================================
# TAB 2: GENERATE MINUTES (FIXED)
# ==========================================
with tab2:
    st.header(" AI Minute Generator")
    st.caption(f"Connected Model: `{active_model_name}`") # Debug info
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Notes", type=['txt', 'png', 'jpg', 'jpeg'], key="notes_upload")
    
    ai_input = None
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        with col2:
            if file_type == 'txt':
                raw_text = uploaded_file.read().decode("utf-8")
                st.text_area("Preview", raw_text[:500] + "...", height=300)
                ai_input = raw_text
            elif file_type in ['png', 'jpg', 'jpeg']:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Notes", use_container_width=True)
                ai_input = image
            
        if st.button("✨ Generate Formal Minutes", type="primary"):
            with st.spinner("Drafting minutes..."):
                try:
                    # FIX: Use the 'active_model_name' we found earlier!
                    model = genai.GenerativeModel(active_model_name)
                    
                    prompt = """
                    You are an expert Corporate Secretary. 
                    TASK: Analyze the input (transcript or image) and write formal Meeting Minutes.
                    FORMAT: Meeting Details, Agenda, Resolutions (Bold), Action Items (Bullets).
                    Tone: Professional.
                    """
                    
                    response = model.generate_content([prompt, ai_input])
                    generated_minutes = response.text
                    
                    st.divider()
                    st.subheader(" Draft Minutes")
                    st.markdown(generated_minutes)
                    st.download_button("📥 Download Minutes (.txt)", generated_minutes, "Minutes_Draft.txt", "text/plain")
                    
                except Exception as e:
                    st.error(f"Error: {e}")