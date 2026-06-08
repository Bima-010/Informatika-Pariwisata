import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gensim.models import LdaModel
from gensim import corpora

st.set_page_config(
    page_title="Sentimen Kuliner Lamongan",
    page_icon="🍲",
    layout="centered"
)

st.title("🍲 Analisis Sentimen & Aspek Ekosistem Kuliner Khas Lamongan")
st.write("Masukkan ulasan untuk memprediksi aspek dan sentimennya secara riil menggunakan model Anda.")

PATH_INDOBERT = "./indoBERT_final_model"  
PATH_LDA_MODEL = "./lda_model.model"       
PATH_LDA_DICT = "./id2word.dict"           

# Mapping Label IndoBERT (0=Negatif, 1=Netral, 2=Positif)
MAPPING_LABEL = {
    0: "NEGATIF",
    1: "NETRAL",
    2: "POSITIF"
}

MAPPING_ASPEK = {
    0: "Aspek 1",
    1: "Aspek 2",
    2: "Aspek 3"
}

# ==========================================
# 2. LOAD SEMUA ARTIFAK MODEL (DENGAN CACHING)
# ==========================================
@st.cache_resource
def load_all_models():
    # Memuat tokenizer dan model IndoBERT dari folder lokal
    tokenizer = AutoTokenizer.from_pretrained(PATH_INDOBERT)
    bert_model = AutoModelForSequenceClassification.from_pretrained(PATH_INDOBERT)
    
    # Memuat model LDA dan kamus kata dari berkas lokal
    lda_model = LdaModel.load(PATH_LDA_MODEL)
    id2word = corpora.Dictionary.load(PATH_LDA_DICT)
    
    return tokenizer, bert_model, lda_model, id2word

try:
    tokenizer, bert_model, lda_model, id2word = load_all_models()
    st.sidebar.success("🟢 Semua Model Berhasil Dimuat!")
    models_ready = True
except Exception as e:
    st.sidebar.error("🔴 Gagal Memuat Model!")
    st.sidebar.write(f"Detail Error: {e}")
    models_ready = False

def analisa_ulasan_baru(teks):
    # --- PROSES EKSTRAKSI ASPEK (LDA) ---
    tokens = teks.lower().split() 
    bow = id2word.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow)
    
    if topics:
        dominant_topic_idx = max(topics, key=lambda x: x[1])[0]
        aspek_terprediksi = MAPPING_ASPEK.get(dominant_topic_idx, "Lainnya")
    else:
        aspek_terprediksi = "Lainnya"

    # --- PROSES PREDIKSI SENTIMEN (INDOBERT) ---
    inputs = tokenizer(teks, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediksi_indeks_kelas = torch.argmax(probs, dim=-1).item()
    confidence_score = probs[0][prediksi_indeks_kelas].item()
    
    sentimen_terprediksi = MAPPING_LABEL.get(prediksi_indeks_kelas, "TIDAK DIKETAHUI")
    
    return aspek_terprediksi, sentimen_terprediksi, confidence_score

st.header("🔍 Uji Nilai Ulasan")

input_ulasan = st.text_area(
    label="Tulis ulasan kuliner Lamongan di sini:",
    value="Soto Lamongan di sini rasanya enak sekali, kuah koya-nya melimpah dan gurih!",
    placeholder="Ketik ulasan..."
)

if st.button("Jalankan Analisis", disabled=not models_ready):
    if input_ulasan.strip() == "":
        st.warning("Silakan isi teks ulasan terlebih dahulu.")
    else:
        aspek, sentimen, skor = analisa_ulasan_baru(input_ulasan)
        
        st.subheader("📋 Hasil Analisis Sistem:")
        st.write(f"📍 **Kategori Tempat / Topik:** `{aspek}`")
        
        if sentimen == "POSITIF":
            st.success(f"**Sentimen:** {sentimen} (Confidence: {skor:.2%})")
        elif sentimen == "NEGATIF":
            st.error(f"**Sentimen:** {sentimen} (Confidence: {skor:.2%})")
        else:
            st.warning(f"**Sentimen:** {sentimen} (Confidence: {skor:.2%})")

st.markdown("---")

st.header("📊 Ringkasan Kata Kunci Model LDA")
st.write("Representasi sebaran kata kunci teratas untuk ke-3 aspek berdasarkan nilai bobot komponen:")

data_lda = {
    "Nama Aspek": ["Aspek 1", "Aspek 2", "Aspek 3"],
    "Kata Kunci Utama & Bobot Distribusi": [
        '0.028*"enak" + 0.027*"lamongan" + 0.024*"dan" + 0.024*"soto" + 0.022*"di"',
        '0.034*"soto" + 0.023*"lamongan" + 0.014*"the" + 0.011*"ke" + 0.008*"kalo"',
        '0.020*"wingko" + 0.019*"enak" + 0.016*"dan" + 0.016*"rasa" + 0.015*"yg"'
    ]
}
df_lda = pd.DataFrame(data_lda)
st.table(df_lda.set_index("Nama Aspek"))

st.caption("Aplikasi ini menggunakan model final IndoBERT-base-p2 & Latent Dirichlet Allocation (LDA) untuk keperluan demonstrasi.")
