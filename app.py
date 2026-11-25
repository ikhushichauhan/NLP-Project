import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from preprocess import preprocess

st.set_page_config(page_title="Haryana News", page_icon="📰", layout="wide")

# ---------------------------------------------------------------- local images
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
IMG_DIR = BASE_DIR / "images"


@st.cache_data
def local_image(name):
    """Return a base64 data URI for images/<name>, or None if the file is missing."""
    path = IMG_DIR / name
    if not path.exists():
        return None
    ext = path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/webp" if ext == "webp" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


BG_IMAGE = local_image("news.png")
HERO_IMAGE = local_image("vardaat.png")

# ---------------------------------------------------------------- styles
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800;900&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { display: none; }

/* whole page content sits on one big glass sheet */
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
  max-width: 98vw !important;
  width: 98vw !important;
  margin: 14px auto 22px auto;
  padding: 0.6rem 1rem 1rem 1rem !important;
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(255,255,255,0.45);
  border-radius: 22px;
  box-shadow: 0 20px 60px rgba(15,23,42,.25);
}

/* shared glass look */
.glass, .panel, .card, .side, .feat, .ticker, .brand-box {
  background: rgba(255,255,255,0.55);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.65);
  box-shadow: 0 8px 30px rgba(15,23,42,.12);
}

/* top utility bar */
.topbar { background:rgba(15,23,42,.72); backdrop-filter: blur(12px);
  color:#e2e8f0; font-size:12.5px; padding:9px 20px; border-radius:12px;
  display:flex; justify-content:space-between; }
.topbar b { color:#fff; }

/* brand header */
.brand-box { display:flex; align-items:center; justify-content:space-between;
  padding:18px 22px; border-radius:16px; margin:14px 0 14px 0; }
.logo { font-family:'Playfair Display', serif; font-weight:900; font-size:46px;
  letter-spacing:-1px; color:#0f172a; line-height:1; }
.logo span { color:#dc2626; }
.tagline { color:#334155; font-size:13px; margin-top:4px; }
.pill { background:#dc2626; color:#fff; font-weight:700; font-size:12px;
  padding:6px 12px; border-radius:999px; letter-spacing:.5px; }

/* tabs as navbar */
div[data-baseweb="tab-list"] { gap:4px; padding:6px 10px; margin-bottom:18px;
  background: rgba(255,255,255,0.60);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border-radius:12px; border:1px solid rgba(255,255,255,0.7);
  box-shadow: 0 8px 30px rgba(15,23,42,.10); }
button[data-baseweb="tab"] { height:42px; padding:0 18px; border-radius:8px;
  background:transparent; }
button[data-baseweb="tab"] p,
button[data-baseweb="tab"] div,
button[data-baseweb="tab"] span { color:#000000 !important; font-weight:600; font-size:15px; }
button[data-baseweb="tab"][aria-selected="true"] { background:rgba(254,202,202,0.85); }
button[data-baseweb="tab"][aria-selected="true"] p { color:#000000 !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display:none; }

/* ticker */
.ticker { border-radius:12px; display:flex; align-items:center; overflow:hidden; margin-bottom:18px; }
.ticker .tag { background:#dc2626; color:#fff; font-weight:800; font-size:12px;
  padding:12px 16px; letter-spacing:1px; white-space:nowrap; }
.ticker .txt { padding:0 16px; color:#0f172a; font-size:14px; font-weight:600;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* hero */
.hero { position:relative; height:430px; border-radius:18px; overflow:hidden;
  background-size:cover; background-position:center;
  border:1px solid rgba(255,255,255,0.6);
  box-shadow:0 14px 40px rgba(15,23,42,.30); }
.hero:after { content:""; position:absolute; inset:0;
  background:linear-gradient(to top, rgba(2,6,23,.55) 0%, rgba(2,6,23,0) 65%); }
.hero .body { position:absolute; left:18px; right:18px; bottom:18px; z-index:2; color:#fff;
  padding:18px 22px; border-radius:14px;
  background: rgba(15,23,42,0.38);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border:1px solid rgba(255,255,255,0.28); }
.hero h1 { font-family:'Playfair Display', serif; font-size:30px; line-height:1.2;
  margin:10px 0 8px; color:#fff; }
.hero p { color:#e2e8f0; font-size:14.5px; margin:0; max-width:680px; }

/* badges */
.badge { display:inline-block; font-size:11px; font-weight:800; letter-spacing:.8px;
  padding:4px 10px; border-radius:6px; text-transform:uppercase; color:#fff; }
.b-World { background:#2563eb; } .b-Sports { background:#16a34a; }
.b-Business { background:#d97706; } .b-SciTech { background:#7c3aed; }
.ai { font-size:10.5px; color:#94a3b8; margin-left:8px; font-weight:600; }
.hero .ai { color:#cbd5e1; }

/* cards */
.card { border-radius:16px; overflow:hidden; margin-bottom:18px; transition:all .2s; height:100%; }
.card:hover { transform:translateY(-4px); box-shadow:0 16px 36px rgba(15,23,42,.22); }
.card .img { height:160px; background-size:cover; background-position:center; }
.card .c { padding:14px 16px 16px; }
.card h4 { font-size:16.5px; line-height:1.35; margin:10px 0 6px; color:#0f172a; font-weight:700; }
.card p { font-size:13.5px; color:#334155; margin:0; line-height:1.5; }
.meta { font-size:12px; color:#64748b; margin-top:10px; }

/* side list */
.side { border-radius:16px; padding:6px 16px; }
.side .it { padding:14px 0; border-bottom:1px solid rgba(148,163,184,.35); }
.side .it:last-child { border-bottom:none; }
.side .n { color:#dc2626; font-weight:800; font-size:13px; }
.side .t { color:#0f172a; font-weight:600; font-size:14.5px; line-height:1.4; margin-top:3px; }

/* section titles */
.sec { display:inline-flex; align-items:center; gap:10px; margin:26px 0 14px;
  padding:8px 18px 8px 12px; border-radius:12px;
  background: rgba(255,255,255,0.55);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border:1px solid rgba(255,255,255,0.65); }
.sec h2 { font-family:'Playfair Display', serif; font-size:24px; margin:0; color:#0f172a; }
.sec .bar { width:5px; height:24px; background:#dc2626; border-radius:3px; }

/* panels */
.panel { border-radius:16px; padding:22px; }
.panel h3 { margin:0 0 4px; color:#0f172a; }
.muted { color:#334155; font-size:14px; }
.result { background:rgba(15,23,42,0.70); backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px); color:#fff;
  border:1px solid rgba(255,255,255,0.25);
  border-radius:16px; padding:22px; margin:12px 0; }
.result .lbl { font-size:12px; letter-spacing:1px; color:#cbd5e1; font-weight:700; }
.result .val { font-family:'Playfair Display', serif; font-size:36px; margin-top:4px; }

/* about */
.feat { border-radius:16px; padding:18px; height:100%; }
.feat .ic { font-size:26px; }
.feat h4 { margin:8px 0 4px; color:#0f172a; }
.feat p { color:#334155; font-size:13.5px; margin:0; }
.stackchip { display:inline-block; background:rgba(255,255,255,0.65); color:#0f172a;
  border:1px solid rgba(255,255,255,0.8); font-weight:600;
  font-size:12.5px; padding:5px 12px; border-radius:999px; margin:3px 4px 3px 0; }

/* footer */
.foot { text-align:center; color:#334155; font-size:12.5px; padding:30px 0 12px; }

/* streamlit widgets */
.stButton > button { background:#dc2626; color:#fff; border:none; border-radius:8px;
  font-weight:700; padding:.55rem 1.4rem; }
.stButton > button:hover { background:#b91c1c; color:#fff; border:none; }
.stTextArea textarea { border-radius:10px; background:rgba(255,255,255,0.75); color:#0f172a; }
.stTextArea label p, .stFileUploader label p { color:#0f172a !important; font-weight:600; }
[data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,0.55);
  border:1px dashed rgba(15,23,42,.35); border-radius:12px; }
[data-testid="stMetric"] { background:rgba(255,255,255,0.55); border:1px solid rgba(255,255,255,0.65);
  border-radius:14px; padding:12px 16px; backdrop-filter: blur(14px); }
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] { color:#0f172a !important; }
[data-testid="stCaptionContainer"] p { color:#334155; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# background image + hero image (embedded so they work without any file server)
GRAD = "linear-gradient(135deg,#1e293b,#475569)"
if BG_IMAGE:
    bg_layer = f"linear-gradient(rgba(15,23,42,.25), rgba(15,23,42,.25)), url('{BG_IMAGE}')"
else:
    bg_layer = "linear-gradient(135deg,#cbd5e1,#e2e8f0)"

HERO_FALLBACK = "https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?auto=format&fit=crop&w=1200&q=70"
hero_layer = f"url('{HERO_IMAGE}')" if HERO_IMAGE else f"url('{HERO_FALLBACK}')"

st.markdown(
    f"""
<style>
.stApp {{
  background-image: {bg_layer};
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  background-repeat: no-repeat;
}}
.hero {{ background-image: {hero_layer}, {GRAD}; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    return joblib.load("models/news_model.joblib")


model = load_model()


def badge_class(topic):
    return "b-" + topic.replace("/", "")


def badge(topic, ai=False):
    extra = '<span class="ai">AI tagged</span>' if ai else ""
    return f'<span class="badge {badge_class(topic)}">{topic}</span>{extra}'


@st.cache_data
def tag(headline):
    clean = preprocess(headline)
    if len(clean.split()) < 3:
        return "World"
    return model.predict([clean])[0]


# ---------------------------------------------------------------- sample content
IMG = "https://images.unsplash.com/{}?auto=format&fit=crop&w=900&q=70"

STORIES = [
    {"h": "Leaders from neighbouring countries agree on a new regional trade and peace framework",
     "d": "Delegations wrapped up two days of talks with a joint statement on border cooperation and shared trade rules.",
     "img": "photo-1529107386315-e1a2ed48a620", "t": "2 hours ago"},
    {"h": "Local team wins the national football championship in a thrilling final",
     "d": "The striker scored twice as the home side lifted the trophy in front of a packed stadium.",
     "img": "photo-1431324155629-1a6deb1dec8d", "t": "3 hours ago"},
    {"h": "Stock market closes higher as investors react to strong company earnings",
     "d": "Banking and manufacturing shares led gains while analysts expect steady growth next quarter.",
     "img": "photo-1611974789855-9c2a0a7236a3", "t": "4 hours ago"},
    {"h": "Scientists unveil a new smartphone chip that doubles battery life",
     "d": "The processor uses a new design that cuts power use while improving speed for artificial intelligence tasks.",
     "img": "photo-1518770660439-4636190af475", "t": "5 hours ago"},
    {"h": "Space agency announces its next mission to study the moon's south pole",
     "d": "The spacecraft will carry instruments to map water ice and test new landing technology.",
     "img": "photo-1446776811953-b23d57bd21aa", "t": "6 hours ago"},
    {"h": "Cricket captain hits a century as the team seals the series with a big win",
     "d": "A record partnership in the middle overs set up a comfortable victory in the deciding match.",
     "img": "photo-1540747913346-19e32dc3e97e", "t": "7 hours ago"},
    {"h": "Central bank holds interest rates steady while inflation cools",
     "d": "Officials said the economy is on a stable path and promised to watch prices closely.",
     "img": "photo-1486406146926-c627a92ad1ab", "t": "8 hours ago"},
    {"h": "United Nations calls for urgent aid as flood waters displace thousands",
     "d": "Relief teams are delivering food and medicine to communities cut off by rising rivers.",
     "img": "photo-1547683905-f686c993aae5", "t": "9 hours ago"},
]
for s in STORIES:
    s["topic"] = tag(s["h"])


def card(s):
    url = IMG.format(s["img"])
    return (
        f'<div class="card"><div class="img" style="background-image:url({url}),{GRAD}"></div>'
        f'<div class="c">{badge(s["topic"], True)}<h4>{s["h"]}</h4>'
        f'<p>{s["d"]}</p><div class="meta">Haryana News Desk &bull; {s["t"]}</div></div></div>'
    )


# ---------------------------------------------------------------- header
st.markdown(
    '<div class="topbar"><div><b>Haryana News</b> &nbsp;|&nbsp; AI-powered news topic classification</div>'
    '<div>Wednesday, 25 November 2026</div></div>'
    '<div class="brand-box"><div><div class="logo">Haryana<span>News</span></div>'
    '<div class="tagline">Every headline, automatically sorted into the right topic.</div></div>'
    '<div class="pill">LIVE</div></div>',
    unsafe_allow_html=True,
)

tab_home, tab_single, tab_csv, tab_about = st.tabs(
    ["🏠 Home", "📝 Classify Headline", "📂 CSV Upload", "ℹ️ About"]
)

# ---------------------------------------------------------------- HOME
with tab_home:
    st.markdown(
        f'<div class="ticker"><div class="tag">BREAKING</div>'
        f'<div class="txt">{STORIES[0]["h"]} &nbsp;&bull;&nbsp; {STORIES[2]["h"]}</div></div>',
        unsafe_allow_html=True,
    )

    hero = STORIES[0]
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown(
            f'<div class="hero"><div class="body">{badge(hero["topic"], True)}'
            f'<h1>{hero["h"]}</h1><p>{hero["d"]}</p></div></div>',
            unsafe_allow_html=True,
        )
    with right:
        items = "".join(
            f'<div class="it"><div class="n">0{i + 1} &bull; {s["topic"].upper()}</div>'
            f'<div class="t">{s["h"]}</div></div>'
            for i, s in enumerate(STORIES[1:5])
        )
        st.markdown(
            f'<div class="sec" style="margin-top:0"><div class="bar"></div><h2>Top Stories</h2></div>'
            f'<div class="side">{items}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec"><div class="bar"></div><h2>Latest News</h2></div>',
                unsafe_allow_html=True)
    rows = STORIES[1:]
    for start in range(0, len(rows), 3):
        cols = st.columns(3, gap="medium")
        for col, s in zip(cols, rows[start:start + 3]):
            with col:
                st.markdown(card(s), unsafe_allow_html=True)

    st.caption("Sample headlines for demonstration. Topic badges are predicted live by the trained model.")

# ---------------------------------------------------------------- SINGLE
with tab_single:
    st.markdown(
        '<div class="panel"><h3>Classify a news headline</h3>'
        '<div class="muted">Paste any English headline or article. The model will predict '
        'whether it is World, Sports, Business or Sci/Tech.</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    text = st.text_area("News headline or article", height=200,
                        placeholder="e.g. The striker scored two goals as his team won the final")
    if st.button("Classify"):
        clean = preprocess(text)
        if len(clean.split()) < 3:
            st.warning("Please enter a longer English news text.")
        else:
            probs = model.predict_proba([clean])[0]
            result = pd.Series(probs, index=model.classes_)
            topic = result.idxmax()
            st.success(f"Topic: **{topic}**")
            st.markdown(
                f'<div class="result"><div class="lbl">PREDICTED TOPIC</div>'
                f'<div class="val">{topic}</div></div>',
                unsafe_allow_html=True,
            )
            st.metric("Model confidence", f"{result.max():.1%}")
            st.bar_chart(result)

# ---------------------------------------------------------------- CSV
with tab_csv:
    st.markdown(
        '<div class="panel"><h3>Classify many articles at once</h3>'
        '<div class="muted">Upload a CSV file that has a column named <b>text</b>. '
        'You can download the results with a predicted topic for every row.</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    file = st.file_uploader("Upload CSV file", type="csv")
    if file:
        try:
            data = pd.read_csv(file)
        except Exception:
            st.error("Could not read the CSV file.")
            st.stop()
        if "text" not in data.columns or data.empty:
            st.error("The CSV must have a 'text' column and at least one row.")
        else:
            cleaned = data["text"].fillna("").apply(preprocess)
            valid = cleaned.str.split().str.len() >= 3
            data["predicted_topic"] = "Text too short"
            if valid.any():
                data.loc[valid, "predicted_topic"] = model.predict(cleaned[valid])
            st.success(f"Classified {int(valid.sum())} of {len(data)} rows.")
            c1, c2 = st.columns([2, 1], gap="large")
            with c1:
                st.dataframe(data, use_container_width=True)
            with c2:
                st.bar_chart(data["predicted_topic"].value_counts())
            st.download_button("Download Results", data.to_csv(index=False),
                               "predictions.csv", "text/csv")

# ---------------------------------------------------------------- ABOUT
with tab_about:
    st.markdown(
        '<div class="sec" style="margin-top:6px"><div class="bar"></div><h2>About Haryana News</h2></div>'
        '<div class="panel"><div class="muted" style="font-size:15.5px;line-height:1.7">'
        'Haryana News is a <b>News Topic Classification</b> project. Newsrooms receive thousands '
        'of articles every day and sorting them by hand is slow. This application reads a headline '
        'or article and automatically predicts its topic: <b>World</b>, <b>Sports</b>, '
        '<b>Business</b> or <b>Sci/Tech</b>. It is a supervised, multi-class text '
        'classification problem trained on the AG News dataset (120,000 training articles).'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    a, b, c, d = st.columns(4, gap="medium")
    feats = [
        (a, "🧹", "Clean data", "Missing values, duplicates and conflicting labels are removed."),
        (b, "🔤", "NLP pipeline", "Tokenization, stopword removal, lemmatization and TF-IDF features."),
        (c, "🤖", "ML models", "Naive Bayes, Logistic Regression, Linear SVM and Word2Vec compared."),
        (d, "📊", "Evaluation", "Accuracy, precision, recall and F1-score on held-out data."),
    ]
    for col, ic, t, desc in feats:
        with col:
            st.markdown(f'<div class="feat"><div class="ic">{ic}</div><h4>{t}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="sec"><div class="bar"></div><h2>Tech Stack</h2></div>',
                unsafe_allow_html=True)
    chips = ["Python", "Pandas", "NLTK", "Scikit-learn", "Gensim", "Matplotlib", "Seaborn", "Streamlit"]
    st.markdown("".join(f'<span class="stackchip">{c}</span>' for c in chips), unsafe_allow_html=True)

    st.markdown('<div class="sec"><div class="bar"></div><h2>Limitations</h2></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><div class="muted" style="font-size:15px;line-height:1.8">'
        '&bull; Works for <b>English</b> news only.<br>'
        '&bull; Supports only <b>4 topics</b>. Politics or entertainment news will fall into one of them.<br>'
        '&bull; The dataset is old, so very new terms may be misclassified.<br><br>'
        '<b>Future scope:</b> DistilBERT model, Hindi news support, online deployment.'
        '</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">&copy; 2026 Haryana News &bull; Built with Streamlit &amp; Scikit-learn</div>',
            unsafe_allow_html=True)