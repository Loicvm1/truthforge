import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random
import os
import json
from datetime import datetime

# --- OPTIONAL DEPENDENCIES (graceful fallback) ---
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

# --- CONFIGURATION ---
LOCAL_MODEL_PATH = "./model/roberta-fake-news-classification"

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.theguardian.com/world/rss",
]

DATASET_IDS = [
    "GonzaloA/fake_news",
    "jdpressman/fake_news",
]

st.set_page_config(page_title="TruthForge — AI News Verification", page_icon="📰", layout="wide")


# =============================================
#  FULL CSS — from website/styles.css
# =============================================
st.markdown("""
<style>
    /* --- Google Fonts --- */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Lora:ital,wght@0,400;0,600;1,400&family=UnifrakturMaguntia&display=swap');

    /* --- Root variables (from styles.css) --- */
    :root {
        --paper-bg: #fdf6e3;
        --paper-dark: #f5ecd4;
        --ink: #1a1a1a;
        --ink-light: #3d3d3d;
        --ink-muted: #6b6b6b;
        --accent-gold: #c9a84c;
        --accent-red: #c0392b;
        --accent-green: #27ae60;
        --accent-blue: #2c3e6b;
        --border-dark: #2c2c2c;
        --border-light: #c4b99a;
        --shadow-soft: 0 4px 24px rgba(0,0,0,0.08);
        --shadow-medium: 0 8px 32px rgba(0,0,0,0.12);
        --radius-sm: 4px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* --- Global overrides --- */
    .stApp {
        background: var(--paper-bg) !important;
        font-family: 'Lora', 'Georgia', serif !important;
        color: var(--ink) !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}

    /* --- Masthead --- */
    .masthead {
        text-align: center;
        padding: 2rem 0 1.5rem;
        border-bottom: 4px double var(--border-dark);
        margin-bottom: 2rem;
    }
    .masthead-ornament {
        font-size: 0.9rem;
        letter-spacing: 1.2em;
        color: var(--accent-gold);
        margin-bottom: 0.5rem;
    }
    .masthead-title {
        font-family: 'UnifrakturMaguntia', 'Playfair Display', serif !important;
        font-size: clamp(3rem, 8vw, 5.5rem);
        font-weight: 400;
        letter-spacing: 0.03em;
        color: var(--ink);
        line-height: 1.1;
        text-shadow: 2px 2px 0 rgba(0,0,0,0.05);
        margin: 0;
    }
    .masthead-subtitle {
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem;
        font-style: italic;
        color: var(--ink-muted);
        margin-top: 0.3rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    .masthead-meta {
        margin-top: 1rem;
        font-size: 0.82rem;
        color: var(--ink-muted);
        letter-spacing: 0.05em;
    }
    .masthead-meta .sep {
        margin: 0 0.6rem;
        color: var(--border-light);
    }
    .masthead-rule {
        width: 60%;
        margin: 1.2rem auto 0;
        border: none;
        border-top: 1px solid var(--border-light);
        height: 0;
    }

    /* --- Column labels --- */
    .column-header {
        border-bottom: 2px solid var(--border-dark);
        margin-bottom: 1.2rem;
        padding-bottom: 0.4rem;
    }
    .column-label {
        font-family: 'Playfair Display', serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--ink);
    }

    /* --- Intro --- */
    .intro-text {
        font-size: 1.05rem;
        color: var(--ink-light);
        line-height: 1.85;
        max-width: 800px;
    }
    .drop-cap {
        float: left;
        font-family: 'Playfair Display', serif;
        font-size: 4.2rem;
        font-weight: 900;
        line-height: 0.8;
        padding: 0.08em 0.12em 0 0;
        color: var(--accent-blue);
    }

    /* --- Panels --- */
    .tf-panel {
        background: white;
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1.8rem;
        box-shadow: var(--shadow-soft);
        transition: box-shadow var(--transition);
    }
    .tf-panel:hover {
        box-shadow: var(--shadow-medium);
    }

    /* --- Verdict badges --- */
    .verdict-badge {
        text-align: center;
        padding: 1.8rem 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.6s ease-out;
    }
    .verdict-real {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid var(--accent-green);
    }
    .verdict-fake {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid var(--accent-red);
    }
    .verdict-icon {
        font-size: 3.2rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .verdict-text {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .verdict-real .verdict-text { color: #155724; }
    .verdict-fake .verdict-text { color: #721c24; }

    @keyframes popIn {
        0% { transform: scale(0); }
        100% { transform: scale(1); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- Score bars --- */
    .score-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .score-label {
        font-size: 0.8rem;
        font-weight: 600;
        width: 110px;
        flex-shrink: 0;
        color: var(--ink-light);
    }
    .score-track {
        flex: 1;
        height: 14px;
        background: var(--paper-dark);
        border-radius: 7px;
        overflow: hidden;
        border: 1px solid var(--border-light);
    }
    .score-fill {
        height: 100%;
        border-radius: 7px;
        transition: width 1s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    .score-fill-real { background: linear-gradient(90deg, #27ae60, #2ecc71); }
    .score-fill-fake { background: linear-gradient(90deg, #c0392b, #e74c3c); }
    .score-value {
        font-family: 'Playfair Display', serif;
        font-size: 0.9rem;
        font-weight: 700;
        width: 55px;
        text-align: right;
        color: var(--ink);
    }

    /* --- Confidence --- */
    .confidence-note {
        text-align: center;
        font-size: 0.82rem;
        color: var(--ink-muted);
        font-style: italic;
        padding-top: 0.8rem;
        border-top: 1px dashed var(--border-light);
    }

    /* --- Placeholder --- */
    .result-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: var(--ink-muted);
        padding: 2rem;
        min-height: 280px;
    }
    .placeholder-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: gentleBounce 3s ease-in-out infinite;
    }
    @keyframes gentleBounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    .result-placeholder p {
        font-style: italic;
        font-size: 0.95rem;
    }

    /* --- History cards --- */
    .history-card {
        background: white;
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 1.2rem;
        box-shadow: var(--shadow-soft);
        transition: all var(--transition);
        animation: fadeInUp 0.3s ease-out;
        position: relative;
        overflow: hidden;
        margin-bottom: 0.8rem;
    }
    .history-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
    }
    .history-card.card-real::before { background: var(--accent-green); }
    .history-card.card-fake::before { background: var(--accent-red); }
    .history-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-medium);
    }
    .history-card-verdict {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .card-real .history-card-verdict { color: var(--accent-green); }
    .card-fake .history-card-verdict { color: var(--accent-red); }
    .history-card-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 0.95rem;
        line-height: 1.4;
        margin-bottom: 0.4rem;
        color: var(--ink);
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .history-card-scores {
        font-size: 0.78rem;
        color: var(--ink-muted);
    }

    /* --- Footer --- */
    .tf-footer {
        text-align: center;
        padding: 1rem 0 2rem;
    }
    .tf-footer-rule {
        border: none;
        border-top: 2px solid var(--border-dark);
        margin-bottom: 1rem;
    }
    .tf-footer p {
        font-size: 0.78rem;
        color: var(--ink-muted);
        letter-spacing: 0.05em;
    }

    /* --- Emoji background --- */
    .emoji-background {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .emoji-bg-item {
        position: absolute;
        opacity: 0.08;
        animation: floatEmoji linear infinite;
        will-change: transform;
    }
    @keyframes floatEmoji {
        0% { transform: translateY(110vh) rotate(0deg); }
        100% { transform: translateY(-10vh) rotate(360deg); }
    }

    /* --- Emoji rain --- */
    .emoji-rain {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 1000;
        overflow: hidden;
    }
    .rain-drop {
        position: absolute;
        top: -60px;
        font-size: 2rem;
        animation: rainFall linear forwards;
        will-change: transform, opacity;
    }
    @keyframes rainFall {
        0% { transform: translateY(0) rotate(0deg) scale(1); opacity: 1; }
        80% { opacity: 0.8; }
        100% { transform: translateY(110vh) rotate(720deg) scale(0.5); opacity: 0; }
    }

    /* --- Streamlit overrides --- */
    .stButton > button {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        border: 2px solid var(--border-dark) !important;
        border-radius: var(--radius-sm) !important;
        transition: all var(--transition) !important;
        background: var(--paper-bg) !important;
        color: var(--ink) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    }
    /* Primary button (Analyze) */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--accent-blue) !important;
        color: white !important;
        border-color: var(--accent-blue) !important;
        font-size: 1rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #1a2a4a !important;
        border-color: #1a2a4a !important;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        font-family: 'Lora', serif !important;
        background: var(--paper-bg) !important;
        border: 1.5px solid var(--border-light) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--ink) !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(44, 62, 107, 0.12) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Playfair Display', serif !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
#  EMOJI BACKGROUND (from script.js)
# =============================================
def render_emoji_background():
    emojis = ['📰', '🗞️', '📄', '🔍', '✍️', '📝']
    items_html = ""
    for i in range(18):
        emoji = random.choice(emojis)
        left = random.random() * 100
        font_size = 1.2 + random.random() * 1.6
        dur = 15 + random.random() * 25
        delay = -(random.random() * 30)
        items_html += f'<span class="emoji-bg-item" style="left:{left}%;font-size:{font_size}rem;animation-duration:{dur}s;animation-delay:{delay}s">{emoji}</span>'
    return f'<div class="emoji-background">{items_html}</div>'


# Render emojis once per session
if "emoji_bg_html" not in st.session_state:
    st.session_state.emoji_bg_html = render_emoji_background()

st.markdown(st.session_state.emoji_bg_html, unsafe_allow_html=True)


# =============================================
#  EMOJI RAIN (from script.js)
# =============================================
def render_emoji_rain(rain_type):
    emoji = '✅' if rain_type == 'real' else '❌'
    drops = ""
    for i in range(50):
        left = random.random() * 100
        fsize = 1.2 + random.random() * 2
        dur = 2 + random.random() * 3
        delay = random.random() * 2
        drops += f'<span class="rain-drop" style="left:{left}%;font-size:{fsize}rem;animation-duration:{dur}s;animation-delay:{delay}s">{emoji}</span>'
    return f'<div class="emoji-rain" id="emoji-rain">{drops}</div>'


# =============================================
#  LOAD MODEL
# =============================================
@st.cache_resource
def load_bert_resources():
    path = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "hamzab/roberta-fake-news-classification"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return tokenizer, model


# --- LOAD DATASET ---
@st.cache_resource
def load_news_dataset():
    if not DATASETS_AVAILABLE:
        return None, None
    for ds_id in DATASET_IDS:
        try:
            ds = load_dataset(ds_id, split="train")
            cols = ds.column_names
            title_col = next((c for c in cols if c.lower() in ["title"]), None)
            text_col = next((c for c in cols if c.lower() in ["text", "content", "body"]), None)
            label_col = next((c for c in cols if c.lower() in ["label"]), None)
            if not all([title_col, text_col, label_col]):
                continue
            reals, fakes = [], []
            for row in ds:
                title = row[title_col] or ""
                text = row[text_col] or ""
                label = row[label_col]
                if len(title.strip()) < 10 or len(text.strip()) < 100:
                    continue
                entry = {"t": title.strip(), "c": text.strip()[:2000]}
                if label in [1, "1", "Real", "real", "TRUE", "true"]:
                    reals.append(entry)
                elif label in [0, "0", "Fake", "fake", "FALSE", "false"]:
                    fakes.append(entry)
            if reals and fakes:
                random.shuffle(reals)
                random.shuffle(fakes)
                return reals[:500], fakes[:500]
        except Exception:
            continue
    return None, None


try:
    tokenizer, model = load_bert_resources()
except Exception as e:
    st.error(f"⚠️ Model Loading Error: {e}")
    st.stop()


# =============================================
#  BERT PREDICTION
# =============================================
def predict_bert(title, text):
    combined_text = f"<title>{title}<content>{text}<end>"
    inputs = tokenizer(
        combined_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )
    probabilities = torch.nn.Softmax(dim=1)(output.logits)[0]
    return {"Fake": probabilities[0].item(), "Real": probabilities[1].item()}


# =============================================
#  LIVE RSS REAL NEWS
# =============================================
def fetch_live_real_news():
    if not FEEDPARSER_AVAILABLE:
        return None, None
    feeds = RSS_FEEDS.copy()
    random.shuffle(feeds)
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            entries = [e for e in feed.entries if hasattr(e, 'link') and hasattr(e, 'title')]
            if not entries:
                continue
            random.shuffle(entries)
            for entry in entries[:5]:
                title = entry.title.strip()
                if NEWSPAPER_AVAILABLE:
                    try:
                        article = Article(entry.link)
                        article.download()
                        article.parse()
                        if len(article.text) > 200:
                            return title, article.text[:2000]
                    except Exception:
                        pass
                summary = getattr(entry, 'summary', '') or getattr(entry, 'description', '')
                if len(summary) > 100:
                    return title, summary[:2000]
        except Exception:
            continue
    return None, None


# =============================================
#  DATA SOURCE FUNCTIONS
# =============================================
def get_real_news(dataset_reals):
    if dataset_reals:
        return dataset_reals[random.randint(0, len(dataset_reals) - 1)]
    result = fetch_live_real_news()
    if result and result[0]:
        return {"t": result[0], "c": result[1]}
    return random.choice(FALLBACK_REAL)


def get_fake_news(dataset_fakes):
    if dataset_fakes:
        return dataset_fakes[random.randint(0, len(dataset_fakes) - 1)]
    return random.choice(FALLBACK_FAKE)


def get_confidence_label(score):
    if score > 90:
        return "Very High Confidence"
    elif score > 75:
        return "High Confidence"
    elif score > 60:
        return "Moderate Confidence"
    else:
        return "Low Confidence — exercise caution"


# =============================================
#  LOAD DATASET
# =============================================
with st.spinner("Loading news dataset (first run only)..."):
    dataset_reals, dataset_fakes = load_news_dataset()


# =============================================
#  SESSION STATE
# =============================================
if "history" not in st.session_state:
    st.session_state.history = []
if "headline_input" not in st.session_state:
    st.session_state.headline_input = ""
if "body_input" not in st.session_state:
    st.session_state.body_input = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "source" not in st.session_state:
    st.session_state.source = "unknown"
if "show_rain" not in st.session_state:
    st.session_state.show_rain = None


# =============================================
#  EMOJI RAIN OVERLAY
# =============================================
if st.session_state.show_rain:
    rain_html = render_emoji_rain(st.session_state.show_rain)
    # Auto-remove after animation via JS
    rain_html += """
    <script>
        setTimeout(function() {
            var rain = document.getElementById('emoji-rain');
            if (rain) rain.remove();
        }, 6000);
    </script>
    """
    components.html(rain_html, height=0)
    st.session_state.show_rain = None


# =============================================
#  MASTHEAD (from index.html)
# =============================================
today = datetime.now().strftime("%A, %B %d, %Y")

st.markdown(f"""
<div class="masthead">
    <div class="masthead-ornament">⚜ ⚜ ⚜</div>
    <div class="masthead-title">TruthForge</div>
    <p class="masthead-subtitle">AI-Powered News Authenticity Verification</p>
    <div class="masthead-meta">
        <span>{today}</span>
        <span class="sep">|</span>
        <span>🤖 RoBERTa Neural Network</span>
        <span class="sep">|</span>
        <span>📰 Edition №1</span>
    </div>
    <div class="masthead-rule"></div>
</div>
""", unsafe_allow_html=True)


# =============================================
#  ABOUT THE TOOL (from index.html)
# =============================================
st.markdown("""
<div class="column-header"><span class="column-label">ABOUT THE TOOL</span></div>
<p class="intro-text">
    <span class="drop-cap">T</span>ruthForge utilizes <strong>RoBERTa</strong>
    (Robustly Optimized BERT Approach) fine-tuned on the ISOT Fake News Dataset for deep contextual
    analysis of news articles. Submit any headline and article body below to
    determine its authenticity with AI-driven precision.
</p>
""", unsafe_allow_html=True)


# =============================================
#  DATA SOURCE STATUS
# =============================================
status_parts = []
if dataset_reals and dataset_fakes:
    status_parts.append(f"✅ Dataset loaded — {len(dataset_reals)} real, {len(dataset_fakes)} fake articles")
else:
    status_parts.append("⚠️ Dataset unavailable — install `datasets` (`pip install datasets`)")
if FEEDPARSER_AVAILABLE:
    status_parts.append("✅ RSS feeds available")
    if NEWSPAPER_AVAILABLE:
        status_parts.append("✅ Full article extraction available")
    else:
        status_parts.append("⚠️ Install `newspaper4k` for full article text")
else:
    status_parts.append("⚠️ Install `feedparser` for live news")

with st.expander("📡 Data Source Status"):
    for part in status_parts:
        st.markdown(part)

st.markdown("---")


# =============================================
#  MAIN LAYOUT — INPUT + RESULTS
# =============================================
col_input, col_result = st.columns(2, gap="large")

# --- INPUT PANEL ---
with col_input:
    st.markdown('<div class="column-header"><span class="column-label">SUBMIT ARTICLE FOR REVIEW</span></div>', unsafe_allow_html=True)

    # Fetch buttons row
    btn_cols = st.columns(3 if FEEDPARSER_AVAILABLE else 2)
    with btn_cols[0]:
        fetch_real = st.button("📥 Load Real News", use_container_width=True)
    with btn_cols[1]:
        fetch_fake = st.button("🧪 Load Fake News", use_container_width=True)
    if FEEDPARSER_AVAILABLE:
        with btn_cols[2]:
            fetch_live = st.button("🌐 Live RSS News", use_container_width=True)
    else:
        fetch_live = False

    # Handle fetch buttons — write directly to widget keys
    if fetch_real:
        article = get_real_news(dataset_reals)
        st.session_state.headline_input = article["t"]
        st.session_state.body_input = article["c"]
        st.session_state.source = "real"
        st.rerun()

    if fetch_fake:
        article = get_fake_news(dataset_fakes)
        st.session_state.headline_input = article["t"]
        st.session_state.body_input = article["c"]
        st.session_state.source = "fake"
        st.rerun()

    if fetch_live:
        with st.spinner("Fetching from RSS feeds..."):
            result = fetch_live_real_news()
            if result and result[0]:
                st.session_state.headline_input = result[0]
                st.session_state.body_input = result[1]
                st.session_state.source = "live"
                st.rerun()
            else:
                st.warning("Could not fetch live news. Check your internet connection.")

    # Input fields — key= controls the value via session state directly
    headline = st.text_input(
        "Headline",
        placeholder="Enter or paste the news headline…",
        key="headline_input"
    )
    body = st.text_area(
        "Article Body",
        placeholder="Enter or paste the full article content…",
        height=200,
        key="body_input"
    )

    # Analyze button
    analyze = st.button("🔍 ANALYZE WITH ROBERTA", use_container_width=True, type="primary")

    if analyze:
        if not headline.strip() or not body.strip():
            st.warning("⚠️ Please provide both a headline and article body.")
        else:
            with st.spinner("RoBERTa is scanning for semantic inconsistencies…"):
                res = predict_bert(headline.strip(), body.strip())
                real_score = res['Real'] * 100
                fake_score = res['Fake'] * 100
                verdict = "REAL" if real_score > fake_score else "FAKE"

                st.session_state.result = {
                    "verdict": verdict,
                    "real_score": round(real_score, 2),
                    "fake_score": round(fake_score, 2),
                    "headline": headline.strip()
                }

                # Emoji rain
                st.session_state.show_rain = "real" if verdict == "REAL" else "fake"

                # Add to history
                source_display = {"real": "📥 Dataset", "fake": "🧪 Dataset", "live": "🌐 RSS", "unknown": "✍️ Manual"}
                st.session_state.history.insert(0, {
                    "title": headline.strip(),
                    "verdict": verdict,
                    "real_score": round(real_score, 2),
                    "fake_score": round(fake_score, 2),
                    "source": source_display.get(st.session_state.source, "✍️ Manual"),
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                st.session_state.history = st.session_state.history[:8]

            st.rerun()


# --- RESULT PANEL ---
with col_result:
    st.markdown('<div class="column-header"><span class="column-label">ANALYSIS REPORT</span></div>', unsafe_allow_html=True)

    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div class="result-placeholder">
            <div class="placeholder-icon">📰</div>
            <p>Submit an article to receive your AI-powered authenticity report.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        is_real = result["verdict"] == "REAL"
        real_score = result["real_score"]
        fake_score = result["fake_score"]
        higher = max(real_score, fake_score)
        confidence = get_confidence_label(higher)

        # Source label
        source_display = {"real": "📥 Dataset (Real)", "fake": "🧪 Dataset (Fake)", "live": "🌐 Live RSS", "unknown": "✍️ Manual Input"}
        st.markdown(f"**Source:** {source_display.get(st.session_state.source, '✍️ Manual Input')}")

        # Verdict badge
        if is_real:
            st.markdown("""
            <div class="verdict-badge verdict-real">
                <span class="verdict-icon">✅</span>
                <span class="verdict-text">Authentic News</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verdict-badge verdict-fake">
                <span class="verdict-icon">❌</span>
                <span class="verdict-text">Fake News Detected</span>
            </div>
            """, unsafe_allow_html=True)

        # Score bars (from script.js displayResult)
        st.markdown(f"""
        <div class="score-row">
            <span class="score-label">✅ Authenticity</span>
            <div class="score-track">
                <div class="score-fill score-fill-real" style="width: {real_score}%;"></div>
            </div>
            <span class="score-value">{real_score:.1f}%</span>
        </div>
        <div class="score-row">
            <span class="score-label">❌ Fabrication</span>
            <div class="score-track">
                <div class="score-fill score-fill-fake" style="width: {fake_score}%;"></div>
            </div>
            <span class="score-value">{fake_score:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Confidence note
        st.markdown(f"""
        <div class="confidence-note">
            RoBERTa Analysis: {confidence} ({higher:.1f}%)
        </div>
        """, unsafe_allow_html=True)


# =============================================
#  PREVIOUS ANALYSES — HISTORY (from script.js)
# =============================================
st.markdown("---")
st.markdown('<div class="column-header"><span class="column-label">PREVIOUS ANALYSES</span></div>', unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown(
        '<p style="color: var(--ink-muted); font-style: italic;">No articles analyzed yet.</p>',
        unsafe_allow_html=True
    )
else:
    # Build history cards using Streamlit columns
    # We display up to 4 cards per row
    items_per_row = 4
    for i in range(0, len(st.session_state.history), items_per_row):
        row_items = st.session_state.history[i:i + items_per_row]
        cols = st.columns(items_per_row)
        
        for idx, item in enumerate(row_items):
            with cols[idx]:
                is_real = item["verdict"] == "REAL"
                card_class = "card-real" if is_real else "card-fake"
                verdict_label = "✅ AUTHENTIC" if is_real else "❌ FAKE"
                title_display = item["title"][:80] + ("…" if len(item["title"]) > 80 else "")

                st.markdown(f"""
                <div class="history-card {card_class}" style="height: 100%;">
                    <div class="history-card-verdict">{verdict_label}</div>
                    <div class="history-card-title">{title_display}</div>
                    <div class="history-card-scores">
                        Real: {item['real_score']:.1f}% · Fake: {item['fake_score']:.1f}%<br>
                        <span style="opacity: 0.8; font-size: 0.7rem;">{item.get('source', '')} · {item['time']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# =============================================
#  FOOTER (from index.html)
# =============================================
st.markdown("""
<div class="tf-footer">
    <div class="tf-footer-rule"></div>
    <p>TruthForge © 2026 — Powered by RoBERTa & Transformers — Thomas More University</p>
</div>
""", unsafe_allow_html=True)
