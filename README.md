**Sentiment & Aspektová Analýza**

Webová aplikácia na analýzu sentimentu zákazníckych online hodnotení v slovenskom jazyku s podporou aspektovej analýzy. Vyvinutá ako súčasť diplomovej práce

**O projekte**

Systém automaticky klasifikuje zákaznícke hodnotenia do troch kategórií — pozitívny, neutrálny, negatívny — a identifikuje kľúčové aspekty (personál, cena, produkty, predajňa, reklamácie, servis, e-shop) s ich samostatným sentimentom.
**Klasifikačný model**: kinit/slovakbert-sentiment-twitter — SlovakBERT 

**Funkcie**

**📊 Porovnanie pobočiek** — porovnanie sentimentu naprieč pobočkami, KPI karty, grafy, sumarizačné tabuľky


**🔬 Detailná analýza**— manuálne zadanie ľubovoľného textu a zobrazenie klasifikácie s aspektmi


**🗄️ Databáza**— prehľad uložených dát, export CSV, import nových hodnotení

**Inštalácia**
Požiadavky: Python 3.x, terminál

# 1. Stiahni repozitár (Code → Download ZIP) a rozbali

# 2. Inštaluj závislosti
pip install -r requirements.txt

# 3. Importuj hodnotenia (xlsx so stĺpcami: review_id, pobocka, text, hviezdy, date)
V module Databáza

# 4. Spusti aplikáciu
streamlit run sentiment_app.py

**Technológie**

**Streamlit** — webové rozhranie
**Transformers** — SlovakBERT model
**Plotly** — vizualizácie
**SQLite** — databáza
**simplemma **— lematizácia

