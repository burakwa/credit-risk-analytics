import streamlit as st
import requests
import pandas as pd
import json

# Page Config
st.set_page_config(
    page_title="Credit Risk Analytics - Risk Tahmin Paneli",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a modern, premium look
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply font to elements */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* App Title Gradient Banner */
    .title-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle-text {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 400;
    }
    
    /* Result Box Styling */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .result-approved {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
        color: #14532d;
    }
    
    .result-rejected {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        color: #7f1d1d;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .result-description {
        font-size: 1rem;
        line-height: 1.6;
        opacity: 0.9;
    }
    
    /* Recommendation Card */
    .rec-card {
        background-color: #fafafa;
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 4px solid #6366f1;
        margin-top: 1rem;
    }
    
    .rec-title {
        font-weight: 600;
        color: #4f46e5;
        margin-bottom: 0.5rem;
    }
    
    /* Connection Indicator */
    .indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .indicator-online {
        background-color: #22c55e;
    }
    .indicator-offline {
        background-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Sample Profiles for testing
SAMPLES = {
    "Seçim Yapılmadı (Boş veya Özel Giriş)": None,
    "🛡️ Düşük Riskli Müşteri (Yüksek Tasarruf, Kısa Vade)": {
        "age": 45,
        "sex": "male",
        "job": 2, # Skilled
        "housing": "own",
        "saving_accounts": "rich",
        "checking_account": "rich",
        "credit_amount": 2500,
        "duration": 12,
        "purpose": "car"
    },
    "⚠️ Yüksek Riskli Müşteri (Düşük Tasarruf, Büyük Tutar, Genç)": {
        "age": 20,
        "sex": "female",
        "job": 0, # Unskilled non-resident
        "housing": "rent",
        "saving_accounts": "little",
        "checking_account": "little",
        "credit_amount": 15000,
        "duration": 48,
        "purpose": "business"
    },
    "👥 Ortalama Profil (Orta Risk, Konut Sahibi)": {
        "age": 32,
        "sex": "male",
        "job": 2, # Skilled
        "housing": "own",
        "saving_accounts": "moderate",
        "checking_account": "moderate",
        "credit_amount": 4500,
        "duration": 24,
        "purpose": "furniture/equipment"
    }
}

# Sidebar - API configuration
st.sidebar.image("assets/logo.png", width=340 )
st.sidebar.title("Kontrol Paneli")

api_url = st.sidebar.text_input("API Sunucu Adresi", value="http://localhost:8000")

# Test API connection
api_online = False
try:
    health_check = requests.get(f"{api_url}/", timeout=2)
    if health_check.status_code == 200:
        api_online = True
except Exception:
    pass

# Show connection status
if api_online:
    st.sidebar.markdown('<span class="indicator indicator-online"></span>**API Sunucusu Aktif**', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="indicator indicator-offline"></span>**API Bağlantısı Yok**', unsafe_allow_html=True)
    st.sidebar.warning("Uygulamanın çalışması için FastAPI sunucusunun aktif olması gerekir. Başlatmak için terminalden çalıştırın:\n\n`poetry run uvicorn app.main:app --reload`")

st.sidebar.markdown("---")
st.sidebar.subheader("Hızlı Test Profili")

# Callback to load sample into session state
def load_sample():
    selected = st.session_state.sample_selector
    if selected != "Seçim Yapılmadı (Boş veya Özel Giriş)":
        data = SAMPLES[selected]
        for key, val in data.items():
            st.session_state[f"input_{key}"] = val

selected_sample = st.sidebar.selectbox(
    "Hazır Profil Seçin",
    options=list(SAMPLES.keys()),
    key="sample_selector",
    on_change=load_sample
)

# Initialize Session State values if not present
DEFAULTS = {
    "age": 30,
    "sex": "male",
    "job": 2,
    "housing": "own",
    "saving_accounts": "little",
    "checking_account": "moderate",
    "credit_amount": 5000,
    "duration": 24,
    "purpose": "car"
}

for k, v in DEFAULTS.items():
    if f"input_{k}" not in st.session_state:
        st.session_state[f"input_{k}"] = v

# Header
st.markdown("""
<div class="title-banner">
    <h1 class="title-text">🛡️ Credit Risk Analytics</h1>
    <p class="subtitle-text">FastAPI + XGBoost tabanlı yapay zeka destekli kredi risk analizi ve karar destek sistemi.</p>
</div>
""", unsafe_allow_html=True)

# Forms and details
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="section-header">Müşteri ve Kredi Başvuru Bilgileri</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["👤 Demografi & İş Bilgileri", "💰 Finansal Durum & Kredi Detayları"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider(
                "Yaş", 
                min_value=18, 
                max_value=100, 
                value=int(st.session_state.input_age), 
                key="input_age",
                help="Müşterinin yaşı (18 - 100 arası olmalıdır)"
            )
            
            sex = st.selectbox(
                "Cinsiyet",
                options=["male", "female"],
                format_func=lambda x: "Erkek (Male)" if x == "male" else "Kadın (Female)",
                index=0 if st.session_state.input_sex == "male" else 1,
                key="input_sex"
            )
        
        with c2:
            job = st.selectbox(
                "Meslek Durumu / Nitelik",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0: Vasıfsız & Yerleşik Olmayan (Unskilled & Non-Resident)",
                    1: "1: Vasıfsız & Yerleşik (Unskilled & Resident)",
                    2: "2: Vasıflı Çalışan (Skilled)",
                    3: "3: Yüksek Nitelikli / Yönetici (Highly Skilled)"
                }[x],
                index=st.session_state.input_job,
                key="input_job",
                help="Müşterinin mesleki nitelik seviyesi"
            )
            
            housing = st.selectbox(
                "Konut Durumu",
                options=["own", "free", "rent"],
                format_func=lambda x: {
                    "own": "Ev Sahibi (Own)",
                    "free": "Ücretsiz Oturuyor (Free)",
                    "rent": "Kiracı (Rent)"
                }[x],
                index=["own", "free", "rent"].index(st.session_state.input_housing),
                key="input_housing"
            )
            
    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            saving_accounts = st.selectbox(
                "Tasarruf Hesabı Durumu",
                options=["little", "moderate", "quite rich", "rich", "unknown"],
                format_func=lambda x: {
                    "little": "Düşük Bakiye (< 100 DM)",
                    "moderate": "Orta Bakiye (100 - 500 DM)",
                    "quite rich": "Yüksek Bakiye (500 - 1000 DM)",
                    "rich": "Çok Yüksek Bakiye (>= 1000 DM)",
                    "unknown": "Bilinmiyor / Yok"
                }[x],
                index=["little", "moderate", "quite rich", "rich", "unknown"].index(st.session_state.input_saving_accounts),
                key="input_saving_accounts"
            )
            
            checking_account = st.selectbox(
                "Vadesiz Hesap Durumu",
                options=["little", "moderate", "rich", "unknown"],
                format_func=lambda x: {
                    "little": "Düşük Bakiye (< 0 / Borçlu)",
                    "moderate": "Orta Bakiye (0 - 200 DM)",
                    "rich": "Yüksek Bakiye (>= 200 DM)",
                    "unknown": "Bilinmiyor / Yok"
                }[x],
                index=["little", "moderate", "rich", "unknown"].index(st.session_state.input_checking_account),
                key="input_checking_account"
            )
            
            purpose = st.selectbox(
                "Kredi Amacı",
                options=["car", "furniture/equipment", "radio/TV", "domestic appliances", "repairs", "education", "business", "vacation/others"],
                format_func=lambda x: {
                    "car": "Otomobil",
                    "furniture/equipment": "Mobilya / Ekipman",
                    "radio/TV": "Radyo / TV",
                    "domestic appliances": "Ev Aletleri",
                    "repairs": "Tamirat",
                    "education": "Eğitim",
                    "business": "İş / Girişim",
                    "vacation/others": "Tatil & Diğer"
                }[x],
                index=["car", "furniture/equipment", "radio/TV", "domestic appliances", "repairs", "education", "business", "vacation/others"].index(st.session_state.input_purpose),
                key="input_purpose"
            )
            
        with c4:
            credit_amount = st.number_input(
                "Kredi Tutarı (EUR)",
                min_value=100,
                max_value=100000,
                value=int(st.session_state.input_credit_amount),
                step=500,
                key="input_credit_amount",
                help="Talep edilen kredi miktarı"
            )
            
            duration = st.slider(
                "Vade (Ay)",
                min_value=1,
                max_value=72,
                value=int(st.session_state.input_duration),
                key="input_duration",
                help="Geri ödeme vadesi (ay cinsinden)"
            )

    submit_button = st.button("📊 Kredi Riskini Hesapla", type="primary", use_container_width=True)

with col2:
    st.markdown('<div class="section-header">Analiz Sonuçları</div>', unsafe_allow_html=True)
    
    if submit_button:
        if not api_online:
            st.error("API sunucusuna bağlanılamadı. Lütfen sunucunun aktif olduğundan emin olun.")
        else:
            # Build payload
            feature_data = {
                "age": int(age),
                "sex": sex,
                "job": int(job),
                "housing": housing,
                "saving_accounts": saving_accounts,
                "checking_account": checking_account,
                "credit_amount": int(credit_amount),
                "duration": int(duration),
                "purpose": purpose
            }
            
            payload = {"features": [feature_data]}
            
            with st.spinner("Kredi riski hesaplanıyor..."):
                try:
                    # Request prediction
                    pred_res = requests.post(f"{api_url}/api/v1/predict", json=payload, timeout=5)
                    # Request probability
                    proba_res = requests.post(f"{api_url}/api/v1/predict-proba", json=payload, timeout=5)
                    
                    if pred_res.status_code == 200 and proba_res.status_code == 200:
                        prediction = pred_res.json()["predictions"][0]
                        # proba is [P(bad), P(good)]
                        probabilities = proba_res.json()["probabilities"][0]
                        p_bad = probabilities[0]
                        p_good = probabilities[1]
                        
                        # Show prediction visual output
                        if prediction == 1: # Good Risk (Low Risk)
                            st.markdown(f"""
                            <div class="result-card result-approved">
                                <div class="result-title">🛡️ ONAYLANABİLİR - DÜŞÜK RİSK</div>
                                <div class="result-description">
                                    Müşteri profili güvenli kredi ödeme sınırları içerisindedir.<br>
                                    <b>Kredi Uygunluk Derecesi:</b> Yüksek / Güvenilir<br>
                                    <b>Geri Ödeme Olasılığı:</b> %{p_good*100:.1f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else: # Bad Risk (High Risk)
                            st.markdown(f"""
                            <div class="result-card result-rejected">
                                <div class="result-title">⚠️ ONAYLANAMAZ - YÜKSEK RİSK</div>
                                <div class="result-description">
                                    Müşteri profili yüksek geri ödememe (temerrüt) riski taşımaktadır.<br>
                                    <b>Kredi Uygunluk Derecesi:</b> Riskli / Red Önerilir<br>
                                    <b>Temerrüt Olasılığı (Risk):</b> %{p_bad*100:.1f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed probability analysis
                        st.subheader("Risk Dağılım Grafiği")
                        chart_data = pd.DataFrame({
                            "Kategori": ["Ödeme Olasılığı (Good)", "Temerrüt Riski (Bad)"],
                            "Oran (%)": [p_good * 100, p_bad * 100]
                        })
                        
                        # Bar chart for probabilities
                        st.bar_chart(chart_data, x="Kategori", y="Oran (%)", color="#16a34a" if prediction == 1 else "#ef4444")
                        
                        # Metrics
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Temerrüt Riski (Bad)", f"%{p_bad*100:.1f}", delta=f"{'+' if p_bad > 0.5 else ''}{(p_bad - 0.5)*100:.1f}%", delta_color="inverse")
                        with m2:
                            st.metric("Geri Ödeme Gücü (Good)", f"%{p_good*100:.1f}", delta=f"{'+' if p_good > 0.5 else ''}{(p_good - 0.5)*100:.1f}%", delta_color="normal")
                            
                        # Recommendations based on data
                        st.markdown('<div class="rec-card">', unsafe_allow_html=True)
                        st.markdown('<div class="rec-title">Karar Destek Önerileri</div>', unsafe_allow_html=True)
                        
                        if prediction == 1:
                            st.markdown("✅ Bu profil için standart kredi koşullarıyla onay verilebilir.")
                            if p_good < 0.7:
                                st.markdown("⚠️ Geri ödeme olasılığı sınırda (%70'in altında). Teminat veya kefil istenmesi yararlı olabilir.")
                            if duration > 36:
                                st.markdown("ℹ️ Kredi vadesi oldukça uzun. İsteğe bağlı olarak faiz oranı marjı artırılabilir.")
                        else:
                            st.markdown("🚨 **Yüksek Risk Uyarısı:** Aşağıdaki düzenlemeleri talep etmeniz önerilir:")
                            if credit_amount > 5000:
                                st.markdown(f"- 📉 Kredi miktarını **{credit_amount * 0.6:.0f} EUR** veya altına çekmeyi talep edin.")
                            if duration > 18:
                                st.markdown("- ⏳ Geri ödeme vadesini **12 veya 18 aya** düşürmesini önerin.")
                            if saving_accounts in ["little", "unknown"]:
                                st.markdown("- 💰 Ek nakit mevduat teminatı veya güçlü bir garantör kefil talep edin.")
                                
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    else:
                        st.error(f"API sunucusundan geçersiz yanıt döndü. Hata Kodu: {pred_res.status_code}")
                except Exception as e:
                    st.error(f"Tahmin işlemi sırasında bir hata oluştu: {str(e)}")
    else:
        st.info("Kredi riskini hesaplamak için soldaki formu doldurun ve 'Kredi Riskini Hesapla' butonuna tıklayın.")
