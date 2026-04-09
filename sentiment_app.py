from PIL import BmpImagePlugin
import streamlit as st
from transformers import pipeline
import re
import pandas as pd
import plotly.express as px
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import simplemma
from unidecode import unidecode
from thefuzz import fuzz
import sqlite3
from datetime import datetime
from streamlit_option_menu import option_menu

# ==========================================
# ZÁKLADNÉ NASTAVENIA
# ==========================================
st.set_page_config(
    page_title="Sentiment & Aspektová Analýza",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# LEXIKÓN — 7 ASPEKTOV
# ==========================================
ASPEKTOVE_SADY = {

    # Fyzický obchod: priestory, atmosféra, orientácia, dostupnosť
    'Predajňa': [
        'predajňa', 'predajn',
        'obchod',
        'pobočka', 'pobočk',
        'prevádzka', 'prevádzk',
        'prodejna',
        'prostredie',
        'priestor',
        'čistota',
        'atmosféra',
        'usporiadanie',
        'prehľadnosť', 'prehľadn',
        'regál',
        'kabínka',
        'šatňa',
        'pokladňa',
        'parkovanie', 'parkovisk',
        'poloha',
        'lokalita',
        'otváracie',
        'oddelenie',
        'sekcia',
        'poschodie',
        'prízemie',
    ],

    # Fyzické vlastnosti tovaru: kvalita, výber, sortiment, veľkosti
    'Produkty': [
        'produkt',
        'tovar',
        'výrobok',
        'artikel',
        'položka',
        'materiál',
        'kvalita',
        'výdrž', 'vydržať',
        'trvanlivosť',
        'odolnosť',
        'spracovanie',
        'šitie',
        'strih',
        'rozpadnúť', 'rozpadá',
        'pokaziť', 'pokazen',
        'chybný', 'chybn',
        'poškodený', 'poškoden',
        'nekvalitný',
        'oblečenie',
        'bunda',
        'tričko',
        'nohavice',
        'šortky',
        'dres',
        'ponožka',
        'plavky',
        'obuv',
        'topánka', 'topánk',
        'tenisky', 'tenisk',
        'botasky',
        'čižmy',
        'sandále',
        'kopačky',
        'bicykel',
        'kolobežka',
        'korčule',
        'lyže',
        'snowboard',
        'paddleboard',
        'kajak',
        'stan',
        'karimatka',
        'ruksak', 'batoh',
        'helma',
        'chrániče',
        'lopta',
        'raketa',
        'veľkosť',
        'výber',
        'sortiment',
        'ponuka',
        'kolekcia',
        'model',
        'variant',
    ],

    # Správanie, ochota, odbornosť zamestnancov
    # Pozn: sentiment adjektíva (milý, arogantný...) sú tu zámerne —
    # bez nich by sme nenašli "predavačka bola milá" kde nie je iné aspektové slovo
    'Personál': [
        'personál',
        'zamestnanec', 'zamestnanc',
        'obsluha',
        'predavač', 'predavačka',
        'poradca',
        'vedúci', 'vedúca',
        'ochranka',
        'pracovník', 'pracovníčka',
        'poradiť', 'poradil', 'poradila',
        'pomôcť', 'pomohol', 'pomohla',
        'obsluhovať',
        'osloviť',
        'privítať',
        'prístup',
        'ochota',
        'správanie',
        'vystupovanie',
        'ochotný', 'ochotn',
        'neochotný', 'neochotn',
        'milý', 'príjemný',
        'nepríjemný',
        'arogantný',
        'profesionálny',
        'odborný',
        'kompetentný',
        'nápomocný',
        'ústretový', 'ústretov',
        'nevrlý',
    ],

    # Výška ceny, hodnota za peniaze, zľavy
    'Cena': [
        'cena', 'cien',
        'cenový',
        'lacný', 'lacn',
        'drahý', 'drahn',
        'predražený',
        'cenovo',
        'finančne',
        'peniaze', 'peniaz',
        'zaplatiť', 'zaplatil',
        'zľava', 'zľav',
        'výpredaj',
        'akcia',
        'pomer',
        'hodnota',
        'rozpočet',
        'voucher',
        'poukaz',
        'bod',
        'vernostný',
        'poštovné',
        'dopravné',
    ],

    # Proces reklamácie, vrátenie tovaru, výmena, záručné podmienky
    'Reklamácie': [
        'reklamácia', 'reklamáci',
        'reklamovať', 'reklamoval',
        'záručný', 'záruk',
        'záruka',
        'garancia', 'garančn',
        'vrátenie', 'vrátiť',
        'výmena', 'vymeniť',
        'refundácia', 'refundovať',
        'uznať', 'uznaný',
        'zamietnuť', 'zamietnut',
        'lehota',
        'spotrebiteľ',
        'chybný tovar',
        'poškodený tovar',
    ],

    # Fyzický servis v predajni: oprava bicyklov, nastavenie, požičovňa
    'Servis': [
        'servis',
        'servisný',
        'oprava', 'opraviť',
        'nastavenie', 'nastaviť',
        'údržba',
        'technik',
        'mechanik',
        'požičovňa', 'požičať',
        'garančná kontrola',
        'garančný servis',
        'dielňa',
        'premazať',
        'nafúkať',
        'dotiahnuť',
        'kalibrovať',
        'namontovať',
        'brzda',
        'reťaz',
        'prehadzovačka',
        'pneumatika',
    ],

    # Online nákup, doručenie, zákaznícka linka, aplikácia
    'E-shop': [
        'eshop', 'e-shop',
        'online',
        'internet',
        'stránka', 'web', 'webe',
        'aplikácia', 'aplikáci', 'appka', 'apka',
        'objednávka', 'objednáv',
        'objednať',
        'doručenie', 'doručiť',
        'doprava',
        'kuriér',
        'balík',
        'dodanie', 'dodať',
        'zasielkovňa', 'zasielkovn',
        'zásielka',
        'sledovanie',
        'sklad',
        'odoslané',
        'vyzdvihnutie',
        'osobný odber',
        'linka',
        'infolinka',
        'telefonicky',
        'volať', 'volal',
        'mail', 'email',
        'dobierka',
    ],
}

# Spojky na delenie viet na klauzuly
CONJUNCTIONS_PATTERN = (
    r'\b(ale|avšak|no|napriek|hoci|pritom|lenže|zatiaľ čo|'
    r'na druhej strane|bohužiaľ|žiaľ|nanešťastie|'
    r'naopak|aspoň|aj keď|aj napriek)\b'
)

# Fuzzy prahy — kratšie slová potrebujú vyšší prah
FUZZY_THRESHOLD_DEFAULT = 82
FUZZY_THRESHOLD_SHORT   = 90   # pre slová kratšie ako 6 znakov


# ==========================================
# MODEL — kinit/slovakbert-sentiment-twitter
# Labels: LABEL_0 = negatívny (-1)
#         LABEL_1 = neutrálny (0)
#         LABEL_2 = pozitívny (+1)
# ==========================================
@st.cache_resource
def load_nlp_tools():
    model_path = "kinit/slovakbert-sentiment-twitter"
    sentiment_model = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path,
        top_k=None,
        truncation=True,
        max_length=512,
    )
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    return sentiment_model


def lemmatize_text(text):
    words = word_tokenize(text, language='slovene')
    lemmatized = [simplemma.lemmatize(w, lang='sk') for w in words]
    return " ".join(lemmatized).lower()


def clean_text(text):
    """
    Vyčistí text pred sentiment analýzou:
    - Odstráni emojis a Unicode symboly ktoré model nevie spracovať
    - Odstráni nadbytočné medzery
    Pôvodný text ostáva nezmenený pre zobrazenie v UI.
    """
    # Odstráni všetky znaky mimo základnej latinky, diakritiky a bežnej interpunkcie
    cleaned = re.sub(
        r'[^\w\s\.,!?;:\-\(\)\'\"áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ]',
        ' ', text
    )
    # Zredukuj viaceré medzery na jednu
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ==========================================
# SPRACOVANIE VÝSLEDKOV — SlovakBERT kinit
# ==========================================
def process_sentiment_results(results):
    """
    kinit/slovakbert-sentiment-twitter vracia:
      1  → Pozitívny
      0  → Neutrálny
      -1 → Negatívny
    """
    scores = {"Pozitívny": 0.0, "Neutrálny": 0.0, "Negatívny": 0.0}
    label_map = {
        "1":  "Pozitívny",
        "0":  "Neutrálny",
        "-1": "Negatívny",
    }
    for res in results:
        label = str(res['label']).strip()
        slovak_label = label_map.get(label)
        if slovak_label:
            scores[slovak_label] = res['score']

    polarity  = scores["Pozitívny"] - scores["Negatívny"]
    max_label = max(scores, key=scores.get)
    return {
        "scores":    scores,
        "label":     max_label,
        "max_score": scores[max_label],
        "polarity":  polarity,
    }



# ==========================================
# EXTRAKCIA ASPEKTOV
# ==========================================
def extract_aspects_ultimate(text, model):
    found_aspects = []
    debug_info    = []

    sentences = sent_tokenize(text, language='slovene')

    for sentence in sentences:
        clauses = re.split(CONJUNCTIONS_PATTERN, sentence, flags=re.IGNORECASE)

        # Zlúčiť spojku späť s nasledujúcou klauzulou
        processed_clauses = [clauses[0]]
        if len(clauses) > 1:
            for i in range(1, len(clauses) - 1, 2):
                processed_clauses.append(clauses[i] + clauses[i + 1])

        for clause in processed_clauses:
            clause = clause.strip()
            if not clause:
                continue

            lemmatized_clause  = lemmatize_text(clause)
            clause_normalized  = unidecode(lemmatized_clause)
            clause_words       = clause_normalized.split()

            # Deduplikácia aspektov na úrovni klauzuly
            found_in_clause = set()

            for aspect_name, keywords in ASPEKTOVE_SADY.items():
                if aspect_name in found_in_clause:
                    continue

                aspect_found     = False
                matched_word_info = ""

                for keyword in keywords:
                    keyword_norm = unidecode(keyword)

                    # 1. Presná zhoda (substring)
                    if keyword_norm in clause_normalized:
                        aspect_found      = True
                        matched_word_info = f"Presná zhoda: '{keyword_norm}'"
                        break

                    # 2. Fuzzy matching — prah závisí od dĺžky kľúčového slova
                    threshold = FUZZY_THRESHOLD_SHORT if len(keyword_norm) < 6 else FUZZY_THRESHOLD_DEFAULT
                    for word in clause_words:
                        if fuzz.ratio(keyword_norm, word) >= threshold:
                            aspect_found      = True
                            matched_word_info = (
                                f"Fuzzy zhoda ({fuzz.ratio(keyword_norm, word)}%): "
                                f"'{word}' ≈ '{keyword_norm}'"
                            )
                            break
                    if aspect_found:
                        break

                if aspect_found:
                    found_in_clause.add(aspect_name)
                    raw_result    = model(clause)[0]
                    sentiment_data = process_sentiment_results(raw_result)

                    debug_info.append({
                        "Časť vety":       clause,
                        "Nájdené cez":     matched_word_info,
                        "Priradený aspekt": aspect_name,
                    })
                    found_aspects.append({
                        "Aspekt":      aspect_name,
                        "Časť vety":   clause,
                        "Zistený stav": sentiment_data['label'],
                        "Polarita":    sentiment_data['polarity'],
                    })

    return found_aspects, debug_info


# ==========================================
# TABUĽKY — MANAŽÉRSKY SÚHRN
# ==========================================
def draw_summary_tables(df_overall, df_aspects):
    st.markdown("## Manažérsky Súhrn")
    st.markdown("Prehľadné tabuľky s výsledkami.")

    st.subheader("Celková úspešnosť pobočiek")
    summary = (
        df_overall.groupby('pobocka')['Sentiment']
        .value_counts(normalize=True)
        .unstack()
        .fillna(0) * 100
    )
    summary = summary.rename(columns={
        'Pozitívny': 'Pozitívne %',
        'Negatívny': 'Negatívne %',
        'Neutrálny': 'Neutrálne %',
    })
    counts = df_overall['pobocka'].value_counts()
    summary['Počet hodnotení'] = counts

    cols = ['Počet hodnotení', 'Pozitívne %', 'Neutrálne %', 'Negatívne %']
    existing_cols = [c for c in cols if c in summary.columns]
    summary = summary[existing_cols].sort_values(by='Pozitívne %', ascending=False)

    column_config = {
        "Počet hodnotení": st.column_config.NumberColumn(
            "Počet hodnotení", format="%d 👤",
        ),
        "Pozitívne %": st.column_config.ProgressColumn(
            "Pozitívne %", format="%.1f%%", min_value=0, max_value=100,
        ),
        "Negatívne %": st.column_config.ProgressColumn(
            "Negatívne %", format="%.1f%%", min_value=0, max_value=100,
        ),
        "Neutrálne %": st.column_config.NumberColumn(
            "Neutrálne %", format="%.1f%%",
        ),
    }
    st.dataframe(summary, column_config=column_config, width="stretch")

    if not df_aspects.empty:
        st.subheader("Detailný súhrn spokojnosti podľa aspektov")
        st.markdown("Tabuľka ukazuje **% pozitívnych hodnotení pre daný aspekt**")

        aspect_totals    = df_aspects.groupby(['pobocka', 'Aspekt']).size()
        aspect_positives = (
            df_aspects[df_aspects['Sentiment'] == 'Pozitívny']
            .groupby(['pobocka', 'Aspekt'])
            .size()
        )
        aspect_matrix = (aspect_positives / aspect_totals * 100).unstack().fillna(0)

        st.dataframe(
            aspect_matrix.style
            .highlight_max(axis=1, color='#D1E7DD', props='font-weight:bold;')
            .format("{:.0f}%"),
            width="stretch",
        )
    else:
        st.info("Pre zvolené kritériá neboli nájdené žiadne detailné aspekty.")


# ==========================================
# MODUL 1: ANALÝZA POBOČIEK (DASHBOARD)
# ==========================================
def run_dashboard_module(sentiment_model, db_path):
    st.title("📊 Sentiment porovnávač pobočiek")
    st.markdown("Nastavte si filtre a porovnajte sentiment pobočiek.")

    with sqlite3.connect(db_path) as conn:
        try:
            df_info = pd.read_sql_query(
                "SELECT DISTINCT pobocka FROM raw_reviews WHERE pobocka IS NOT NULL AND pobocka != ''",
                conn,
            )
            vsetky_pobocky = df_info['pobocka'].tolist()

            df_dates   = pd.read_sql_query(
                "SELECT date FROM raw_reviews WHERE date IS NOT NULL AND date != ''", conn
            )
            valid_dates = pd.to_datetime(df_dates['date'], errors='coerce').dropna()

            if not valid_dates.empty:
                db_min_date = valid_dates.min().date()
                db_max_date = valid_dates.max().date()
            else:
                db_min_date = db_max_date = datetime.today().date()
        except Exception as e:
            vsetky_pobocky = []
            db_min_date = db_max_date = datetime.today().date()
            st.warning(f"Chyba pri čítaní databázy pre filtre: {e}")

    if not vsetky_pobocky:
        st.warning("⚠️ Databáza je prázdna, nie sú k dispozícii žiadne hodnotenia.")
        return

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        vybrane_pobocky = st.multiselect("📍 Vyberte pobočky na analýzu:", options=vsetky_pobocky)
        vybrat_vsetky = st.checkbox("Vybrať všetky pobočky", value=False)
        if vybrat_vsetky:
            vybrane_pobocky = vsetky_pobocky
    with col_filter2:
        vybrany_datum = st.date_input("📅 Vyberte obdobie (Od - Do):", value=(db_min_date, db_max_date))
        analyzovat_vsetko = st.checkbox("Analyzovať celé dostupné obdobie", value=True)

    if analyzovat_vsetko:
        start_date, end_date = db_min_date, db_max_date
    else:
        if len(vybrany_datum) == 2:
            start_date, end_date = vybrany_datum
        else:
            start_date = end_date = vybrany_datum[0]

    str_start = start_date.strftime("%Y-%m-%d")
    str_end   = end_date.strftime("%Y-%m-%d")

    st.write("")

    if st.button("Spustiť Analýzu", type="primary", use_container_width=True):
        if not vybrane_pobocky:
            st.error("⚠️ Vyberte aspoň jednu pobočku!")
            return

        st.markdown("---")
        placeholders = ', '.join(['?'] * len(vybrane_pobocky))
        sql_params   = tuple(vybrane_pobocky) + (str_start, str_end)

        with st.spinner("Hľadám nové hodnotenia pre zvolené pobočky a obdobie..."):
            with sqlite3.connect(db_path) as conn:
                query_new = f"""
                    SELECT r.review_id, r.text
                    FROM raw_reviews r
                    LEFT JOIN processed_sentiment p ON r.review_id = p.review_id
                    WHERE p.review_id IS NULL
                      AND r.text IS NOT NULL AND r.text != ''
                      AND r.pobocka IN ({placeholders})
                      AND r.date BETWEEN ? AND ?
                """
                df_new = pd.read_sql_query(query_new, conn, params=sql_params)

                if not df_new.empty:
                    st.info(f"💡 Bolo nájdených {len(df_new)} nových hodnotení. Spúšťam analýzu.")
                    cursor       = conn.cursor()
                    progress_bar = st.progress(0)
                    total        = len(df_new)

                    # OPRAVA: enumerate namiesto iterrows index
                    for idx, (_, row) in enumerate(df_new.iterrows()):
                        r_id = row['review_id']
                        text = row['text']

                        raw_sent   = sentiment_model(text)[0]
                        sent_data  = process_sentiment_results(raw_sent)
                        cursor.execute(
                            "INSERT INTO processed_sentiment (review_id, sentiment_label, sentiment_score) VALUES (?, ?, ?)",
                            (r_id, sent_data['label'], sent_data['max_score']),
                        )

                        aspects, _ = extract_aspects_ultimate(text, sentiment_model)
                        for asp in aspects:
                            cursor.execute(
                                "INSERT INTO aspect_analysis (review_id, aspekt, veta, sentiment) VALUES (?, ?, ?, ?)",
                                (r_id, asp['Aspekt'], asp['Časť vety'], asp['Zistený stav']),
                            )

                        progress_bar.progress((idx + 1) / total)

                    conn.commit()
                    st.success("✅ Nové dáta boli úspešne analyzované a uložené do databázy.")
                else:
                    st.success("✨ Tento výber je už plne zanalyzovaný. Načítavam dáta z databázy.")

        with sqlite3.connect(db_path) as conn:
            query_overall = f"""
                SELECT r.pobocka, r.text, p.sentiment_label AS Sentiment
                FROM raw_reviews r
                JOIN processed_sentiment p ON r.review_id = p.review_id
                WHERE r.pobocka IN ({placeholders})
                  AND r.date BETWEEN ? AND ?
            """
            df_overall_data = pd.read_sql_query(query_overall, conn, params=sql_params)

            query_aspects = f"""
                SELECT r.pobocka, a.aspekt AS Aspekt, a.sentiment AS Sentiment
                FROM raw_reviews r
                JOIN aspect_analysis a ON r.review_id = a.review_id
                WHERE r.pobocka IN ({placeholders})
                  AND r.date BETWEEN ? AND ?
            """
            df_aspect_data = pd.read_sql_query(query_aspects, conn, params=sql_params)

        if df_overall_data.empty:
            st.warning("⚠️ V databáze sa pre tento výber nenachádzajú žiadne spracované dáta.")
            return

        color_map = {'Pozitívny': '#2ecc71', 'Neutrálny': '#f1c40f', 'Negatívny': '#e74c3c'}

        for pobocka in df_overall_data['pobocka'].unique():
            st.subheader(f"📍 Pobočka: {pobocka}")
            df_branch_overall = df_overall_data[df_overall_data['pobocka'] == pobocka]
            df_branch_aspects = df_aspect_data[df_aspect_data['pobocka'] == pobocka]

            sent_counts = df_branch_overall['Sentiment'].value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Počet']

            # ── KPI CARDS ──────────────────────────────────────────────
            total       = len(df_branch_overall)
            poc_poz     = (df_branch_overall['Sentiment'] == 'Pozitívny').sum()
            poc_neg     = (df_branch_overall['Sentiment'] == 'Negatívny').sum()
            pct_poz     = poc_poz / total * 100 if total else 0
            pct_neg     = poc_neg / total * 100 if total else 0

            najlepsi_aspekt = "—"
            najhorsi_aspekt = "—"
            if not df_branch_aspects.empty:
                asp_totals    = df_branch_aspects.groupby('Aspekt').size()
                asp_positives = (
                    df_branch_aspects[df_branch_aspects['Sentiment'] == 'Pozitívny']
                    .groupby('Aspekt').size()
                )
                asp_score = (asp_positives / asp_totals).fillna(0)
                najlepsi_aspekt = asp_score.idxmax()
                najhorsi_aspekt = asp_score.idxmin()

            st.markdown(f"""
            <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:16px;">
                <div style="background:#f0f2f6; border-radius:10px; padding:14px 16px; text-align:center;">
                    <div style="font-size:12px; color:#666; margin-bottom:4px;">📝 Hodnotenia</div>
                    <div style="font-size:22px; font-weight:700; color:#1a1a1a;">{total}</div>
                </div>
                <div style="background:#f0f2f6; border-radius:10px; padding:14px 16px; text-align:center;">
                    <div style="font-size:12px; color:#666; margin-bottom:4px;">✅ Pozitívne</div>
                    <div style="font-size:22px; font-weight:700; color:#2ecc71;">{pct_poz:.1f}%</div>
                </div>
                <div style="background:#f0f2f6; border-radius:10px; padding:14px 16px; text-align:center;">
                    <div style="font-size:12px; color:#666; margin-bottom:4px;">❌ Negatívne</div>
                    <div style="font-size:22px; font-weight:700; color:#e74c3c;">{pct_neg:.1f}%</div>
                </div>
                <div style="background:#f0f2f6; border-radius:10px; padding:14px 16px; text-align:center;">
                    <div style="font-size:12px; color:#666; margin-bottom:4px;">🏆 Najlepší aspekt</div>
                    <div style="font-size:16px; font-weight:700; color:#1a1a1a;">{najlepsi_aspekt}</div>
                </div>
                <div style="background:#f0f2f6; border-radius:10px; padding:14px 16px; text-align:center;">
                    <div style="font-size:12px; color:#666; margin-bottom:4px;">⚠️ Najhorší aspekt</div>
                    <div style="font-size:16px; font-weight:700; color:#1a1a1a;">{najhorsi_aspekt}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # ────────────────────────────────────────────────────────────

            fig_pie = None
            if not sent_counts.empty:
                fig_pie = px.pie(
                    sent_counts, values='Počet', names='Sentiment', hole=0.4,
                    title="Celkový sentiment pobočky",
                    color='Sentiment', color_discrete_map=color_map,
                )
                fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300)

            fig_bar = None
            if not df_branch_aspects.empty:
                asp_counts = df_branch_aspects.groupby(['Aspekt', 'Sentiment']).size().reset_index(name='Počet')

             
                asp_pivot = asp_counts.pivot(index='Aspekt', columns='Sentiment', values='Počet').fillna(0)
                for col in ['Pozitívny', 'Neutrálny', 'Negatívny']:
                    if col not in asp_pivot.columns:
                        asp_pivot[col] = 0
                asp_pivot = asp_pivot[['Pozitívny', 'Neutrálny', 'Negatívny']].reset_index()

                fig_bar = px.bar(
                    asp_pivot,
                    y='Aspekt',
                    x=['Pozitívny', 'Neutrálny', 'Negatívny'],
                    title='Analýza konkrétnych aspektov (ABSA)',
                    orientation='h',
                    color_discrete_map=color_map,
                    barmode='stack',
                )
                fig_bar.update_layout(
                    margin=dict(t=40, b=0, l=0, r=0),
                    height=350,
                    legend_title_text='Sentiment',
                    xaxis_title='Počet hodnotení',
                    yaxis_title='',
                )

            col1, col2 = st.columns([1, 1.5])
            with col1:
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{pobocka}")
            with col2:
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{pobocka}")
                else:
                    st.info("Žiadne zachytené aspekty pre túto pobočku.")

            with st.expander(f"Zobraziť hodnotenia pobočky — {pobocka} ({len(df_branch_overall)} záznamov)"):
                st.dataframe(df_branch_overall[['text', 'Sentiment']], width="stretch")

            st.divider()

        st.markdown("---")
        draw_summary_tables(df_overall_data, df_aspect_data)



# ==========================================
# MODUL 2: DETAILNÁ ANALÝZA 1 RECENZIE
# ==========================================
def run_laboratory_module(sentiment_model):
    st.title("Detailná Analýza")

    user_text = st.text_area("Vložte text, ktorý chcete analyzovať:", height=120, value="")

    if st.button("Analyzovať text", type="primary"):
        if not user_text.strip():
            st.warning("Zadajte text na analýzu.")
            return

        st.markdown("---")

        # 1. CELKOVÝ SENTIMENT
        st.subheader("Celkový Sentiment Textu")
        raw_sent_results = sentiment_model(user_text)[0]
        sent_data        = process_sentiment_results(raw_sent_results)

        col1, col2 = st.columns([1, 2])
        with col1:
            if sent_data['label'] == "Pozitívny":
                st.success(f"**Dominantný sentiment:** {sent_data['label']}")
            elif sent_data['label'] == "Negatívny":
                st.error(f"**Dominantný sentiment:** {sent_data['label']}")
            else:
                st.info(f"**Dominantný sentiment:** {sent_data['label']}")
            st.metric(label="Vypočítaná Polarita (-1 až +1)", value=f"{sent_data['polarity']:.2f}")
        with col2:
            st.progress(sent_data['scores']['Pozitívny'], text=f"Pozitívny ({sent_data['scores']['Pozitívny']:.1%})")
            st.progress(sent_data['scores']['Neutrálny'], text=f"Neutrálny ({sent_data['scores']['Neutrálny']:.1%})")
            st.progress(sent_data['scores']['Negatívny'], text=f"Negatívny ({sent_data['scores']['Negatívny']:.1%})")

        st.markdown("---")

        # 2. ASPEKTY
        st.subheader("Detailný rozbor nájdených aspektov")
        with st.spinner("Hľadám aspekty cez Fuzzy Matching..."):
            extracted_aspects, debug_info = extract_aspects_ultimate(user_text, sentiment_model)

        if not extracted_aspects:
            st.warning("V texte neboli nájdené žiadne kľúčové slová z lexikónu.")
        else:
            for item in extracted_aspects:
                status = item['Zistený stav']
                if status == "Pozitívny":
                    color, emoji = "green", "🟢"
                elif status == "Negatívny":
                    color, emoji = "red", "🔴"
                else:
                    color, emoji = "orange", "🟡"

                st.markdown(f"**Aspekt: `{item['Aspekt']}`**")
                st.markdown(f"> *\"{item['Časť vety']}\"*")
                st.markdown(
                    f"**{emoji} {status}** (Skóre polarity: "
                    f"<span style='color:{color}'>{item['Polarita']:.2f}</span>)",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

        # 3. DEBUG
        with st.expander("🔍 Debug — ako boli aspekty nájdené"):
            if debug_info:
                st.dataframe(pd.DataFrame(debug_info), width="stretch")
            else:
                st.info("Žiadne aspekty nenájdené.")


# ==========================================
# MODUL 3: DATA WAREHOUSE
# ==========================================
def run_data_warehouse_module(db_path):
    st.title("🗄️Databáza")
    st.markdown("Prehľad uložených dát.")
 
    with sqlite3.connect(db_path) as conn:
        try:
            raw_df  = pd.read_sql_query("SELECT * FROM raw_reviews", conn)
            sent_df = pd.read_sql_query(
                "SELECT  p.review_id, r.pobocka, r.text, p.sentiment_label, p.sentiment_score "
                "FROM processed_sentiment p "
                "JOIN raw_reviews r ON p.review_id = r.review_id",
                conn
            )
            asp_df  = pd.read_sql_query(
                "SELECT a.review_id, r.pobocka, a.aspekt, a.veta, a.sentiment "
                "FROM aspect_analysis a "
                "JOIN raw_reviews r ON a.review_id = r.review_id",
                conn
            )
        except Exception as e:
            st.error(f"❌ Chyba pri načítaní databázy: {e}")
            return
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Online hodnotenia",   len(raw_df))
    col2.metric("Spracovaný sentiment", len(sent_df))
    col3.metric("Nájdené aspekty",      len(asp_df))
    st.divider()
 
    tab1, tab2, tab3 = st.tabs(["Online hodnotenia", "Sentiment", "Aspekty"])
    with tab1:
        st.dataframe(raw_df, width="stretch", height=400)
        st.download_button("Stiahnuť raw_reviews (.csv)", raw_df.to_csv(index=False).encode('utf-8'), 'raw_reviews.csv', 'text/csv')
    with tab2:
        st.dataframe(sent_df, width="stretch", height=400)
        st.download_button("Stiahnuť processed_sentiment (.csv)", sent_df.to_csv(index=False).encode('utf-8'), 'processed_sentiment.csv', 'text/csv')
    with tab3:
        st.dataframe(asp_df, width="stretch", height=400)
        st.download_button("Stiahnuť aspect_analysis (.csv)", asp_df.to_csv(index=False).encode('utf-8'), 'aspect_analysis.csv', 'text/csv')
 
    st.divider()
    st.subheader("📥 Import nových hodnotení")
    st.markdown("Nahrajte súbor vo formáte `.xlsx` alebo `.csv` so stĺpcami: `review_id`, `pobocka`, `text`, `hviezdy`, `date`")
 
    uploaded_file = st.file_uploader("Vybrať súbor", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
 
            df_upload = df_upload[["review_id", "pobocka", "text", "hviezdy", "date"]].copy()
            df_upload = df_upload[df_upload["text"].notna() & (df_upload["text"].str.strip() != "")].copy()
            df_upload["review_id"] = df_upload["pobocka"].str.replace(" ", "_") + "_" + df_upload["review_id"].astype(str)
            df_upload["pobocka"]   = df_upload["pobocka"].astype(str)
            df_upload["text"]      = df_upload["text"].astype(str)
            df_upload["hviezdy"]   = pd.to_numeric(df_upload["hviezdy"], errors="coerce").fillna(0).astype(int)
            df_upload["date"]      = df_upload["date"].astype(str)
 
            st.info(f"Súbor obsahuje {len(df_upload)} hodnotení. Kliknite na tlačidlo pre import.")
 
            if st.button("Importovať do databázy", type="primary"):
                with sqlite3.connect(db_path) as conn:
                    existing_ids = set(pd.read_sql_query("SELECT review_id FROM raw_reviews", conn)["review_id"].astype(str))
                    new_rows = df_upload[~df_upload["review_id"].isin(existing_ids)].copy()
 
                    if new_rows.empty:
                        st.warning("Žiadne nové záznamy — všetky už existujú v databáze.")
                    else:
                        cursor = conn.cursor()
                        for _, row in new_rows.iterrows():
                            cursor.execute(
                                "INSERT OR IGNORE INTO raw_reviews (review_id, pobocka, text, hviezdy, date) VALUES (?, ?, ?, ?, ?)",
                                (str(row["review_id"]), str(row["pobocka"]), str(row["text"]), int(row["hviezdy"]), str(row["date"]))
                            )
                        conn.commit()
                        st.success(f"✅ Importovaných {len(new_rows)} nových hodnotení. Preskočených (duplicity): {len(df_upload) - len(new_rows)}")
                        st.rerun()
 
        except Exception as e:
            st.error(f"❌ Chyba pri spracovaní súboru: {e}")
            st.info("Uisti sa že súbor obsahuje stĺpce: review_id, pobocka, text, hviezdy, date")
 

# ==========================================
# INICIALIZÁCIA DATABÁZY
# ==========================================
def init_db(db_path):
    """
    Vytvorí tabuľky databázy ak ešte neexistujú.
    Schéma:
      raw_reviews        — online hodnotenia so hviezdičkami
      processed_sentiment — sentiment priradený k hodnoteniu (FK → raw_reviews)
      aspect_analysis    — aspekty nájdené v hodnotení (FK → raw_reviews)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS raw_reviews (
                review_id   TEXT    PRIMARY KEY,
                pobocka     TEXT    NOT NULL,
                text        TEXT    NOT NULL,
                hviezdy      INTEGER,
                date        TEXT
            );

            CREATE TABLE IF NOT EXISTS processed_sentiment (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id        TEXT    NOT NULL,
                sentiment_label  TEXT    NOT NULL,
                sentiment_score  REAL    NOT NULL,
                FOREIGN KEY (review_id) REFERENCES raw_reviews (review_id)
            );

            CREATE TABLE IF NOT EXISTS aspect_analysis (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT    NOT NULL,
                aspekt    TEXT    NOT NULL,
                veta      TEXT,
                sentiment TEXT,
                FOREIGN KEY (review_id) REFERENCES raw_reviews (review_id)
            );
        """)
        conn.commit()


# ==========================================
# MASTER ROUTER
# ==========================================
def main():
    with st.spinner("Systém inicializuje jazykové a AI modely..."):
        sentiment_model = load_nlp_tools()

    db_path = 'databaza.db'
    init_db(db_path)

    selected_module = option_menu(
        menu_title=None,
        options=["Porovnávanie pobočiek", "Detailná Analýza", "Databáza"],
        icons=["bar-chart-line", "microscope", "database"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container":        {"padding": "0!important", "background-color": "#fafafa", "margin-bottom": "20px"},
            "icon":             {"color": "orange", "font-size": "18px"},
            "nav-link":         {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0047AB", "color": "white"},
        },
    )

    if selected_module == "Porovnávanie pobočiek":
        run_dashboard_module(sentiment_model, db_path)
    elif selected_module == "Detailná Analýza":
        run_laboratory_module(sentiment_model)
    elif selected_module == "Databáza":
        run_data_warehouse_module(db_path)


if __name__ == "__main__":
    main()