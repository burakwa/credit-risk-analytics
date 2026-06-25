# Credit Risk Analytics 🚀
![Credit Risk Analytics Logo](app/ui/assets/logo.png)

Bu proje, makine öğrenmesi teknikleri kullanarak kredi riskini analiz eden ve tahminleyen uçtan uca (end-to-end) bir sistemdir. **FastAPI** tabanlı güçlü bir backend mimarisi ve kullanıcı dostu bir **Streamlit** arayüzü (frontend) barındırır. Projede bağımlılık yönetimi için **Poetry** kullanılmıştır.

---
### ⚖️ Sorumluluk Reddi (Disclaimer)
Yasal Uyarı: Bu proje tamamen eğitim ve portfolyo amacıyla geliştirilmiştir. Sistem tarafından üretilen kredi riski tahminleri ve analiz sonuçları yatırım, finans veya kesin karar tavsiyesi niteliği taşımamaktadır. Proje yazarının, bu yazılımın kullanımından doğabilecek doğrudan veya dolaylı hiçbir zarardan, veri kaybından ya da finansal kayıptan ötürü herhangi bir sorumluluğu bulunmamaktadır. Kullanıcılar tüm riski kendileri üstlenir.

## 🛠️ Kullanılan Teknolojiler ve Bağımlılıklar

Proje mimarisi üç ana grupta toplanan modern kütüphaneler üzerine inşa edilmiştir:

* **Çekirdek & Web (Dependencies):** FastAPI, Uvicorn, Streamlit, Requests, Pytest
* **Makine Öğrenmesi (ML):** Scikit-learn, XGBoost, Pandas, NumPy, Joblib, Matplotlib, Seaborn
* **Geliştirme (Dev):** HTTPX

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla takip edin.

### 1. Ön Gereksinimler
Sisteminizde **Python** ve **Poetry**'nin kurulu olduğundan emin olun. Eğer Poetry kurulu değilse, resmi sitesinden yükleyebilirsiniz.

### 2. Bağımlılıkların Yüklenmesi
Proje kök dizinine gidin ve Poetry kullanarak gerekli tüm paketleri yükleyin:

```bash
poetry install
```

### 3. Backend (API) Servisinin Başlatılması
FastAPI sunucusunu başlatmak için ana dizinde aşağıdaki komutu çalıştırın:
```bash
poetry run uvicorn app.main:app --reload
```
### 4. Frontend (Kullanıcı Arayüzü) Başlatılması
Streamlit arayüzünü çalıştırmak için app/ui dizinine geçiş yapın veya ana dizinden şu komutu yürütün:

```bash
poetry run streamlit run app/ui/ui.py
```

### Testlerin Çalıştırılması
```bash
poetry run pytest
```