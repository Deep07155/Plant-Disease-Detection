"""Plant Disease Detection | premium Streamlit frontend.

Run: streamlit run app.py
The model is trained already. This file only provides the interface and inference.
"""
import base64
import hashlib
import html
import io
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from predict import format_class, load_model, predict_image

st.set_page_config(
    page_title="LeafLens • Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets" / "hero.jpg"
CLASS_FILE = ROOT / "model" / "class_names.json"
ACCURACY_CHART = ROOT / "image" / "accuracy_comparison.png"
LOSS_CHART = ROOT / "image" / "loss_comparison.png"

STYLE_OPTIONS = ["Premium Dark", "Natural & Clean", "AgriTech Dashboard"]
NAV_OPTIONS = ["Home", "Analyze", "Insights", "About"]

if "visual_style" not in st.session_state:
    st.session_state.visual_style = "Premium Dark"
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "prediction" not in st.session_state:
    st.session_state.prediction = None


@st.cache_data
# Read this local image only once; base64 makes the background work offline.
def hero_image_data():
    return base64.b64encode(ASSET.read_bytes()).decode("utf-8")


@st.cache_resource(show_spinner="Loading your trained MobileNetV2 model...")
def cached_model():
    return load_model()


def safe_html(value):
    return html.escape(str(value), quote=True)


def css_for_style(style):
    """One layout, three preview-inspired palettes (dark is the default)."""
    theme = {
        "Premium Dark": {
            "bg": "#07120f", "card": "#10221a", "card2": "#173327",
            "text": "#edf7e9", "muted": "#a9c2b1", "line": "#2b4b39",
            "accent": "#b2fb3d", "accenttext": "#10230b",
            "hero": "linear-gradient(90deg, rgba(4,19,10,.98) 0%, rgba(7,24,12,.93) 48%, rgba(7,24,12,.18) 100%)",
        },
        "Natural & Clean": {
            "bg": "#f1f6e9", "card": "#fffef8", "card2": "#e5f2d8",
            "text": "#193e27", "muted": "#637c65", "line": "#cedbc1",
            "accent": "#226b37", "accenttext": "#ffffff",
            "hero": "linear-gradient(90deg, rgba(247,251,236,.99) 0%, rgba(241,250,230,.92) 51%, rgba(245,251,231,.02) 100%)",
        },
        "AgriTech Dashboard": {
            "bg": "#eef6fa", "card": "#ffffff", "card2": "#e2f3f7",
            "text": "#153449", "muted": "#5c7989", "line": "#d2e5ec",
            "accent": "#008c91", "accenttext": "#ffffff",
            "hero": "linear-gradient(90deg, rgba(238,250,251,.99) 0%, rgba(227,245,247,.96) 51%, rgba(9,89,74,.12) 100%)",
        },
    }[style]

    return """
<style>
:root { --bg: %s; --card: %s; --card2: %s; --fg: %s; --muted: %s; --line: %s; --accent: %s; --accenttext: %s; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:var(--bg)!important; color:var(--fg)!important;}
[data-testid="stAppViewContainer"] > .main {background:var(--bg)!important;}
.main .block-container {max-width:1440px;padding-top:1.0rem;padding-bottom:3.5rem;padding-left:2.0rem;padding-right:2.0rem;}
#MainMenu, footer {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stMarkdownContainer"], label, h1,h2,h3,h4, p,small {color:var(--fg);}
.stCaption, [data-testid="stCaptionContainer"] {color:var(--muted)!important;}
section[data-testid="stSidebar"] {background:var(--card)!important;}
[data-testid="stVerticalBlockBorderWrapper"] {background:var(--card);border:1px solid var(--line)!important;border-radius:19px!important;box-shadow:0 12px 35px rgba(0,0,0,.04);}
[data-testid="stVerticalBlockBorderWrapper"] > div {padding:4px 5px;}
[data-testid="stFileUploader"] {background:var(--card2);border:1px dashed var(--accent);border-radius:13px;padding:12px;}
[data-testid="stFileUploader"] section {background:transparent!important;border:0!important;color:var(--fg)!important;}
[data-testid="stFileUploader"] button {border-color:var(--line)!important;}
[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {color:var(--muted)!important;}
.stButton button[kind="primary"], button[data-testid="stBaseButton-primary"] {background:var(--accent)!important;color:var(--accenttext)!important;font-weight:800;border:0!important;border-radius:12px;min-height:3.0rem;}
.stButton button[kind="primary"]:hover {filter:brightness(.94);}
.stButton button:not([kind="primary"]) {border-color:var(--line)!important;background:var(--card2)!important;color:var(--fg)!important;border-radius:12px;}
.stDownloadButton button {border-radius:12px;background:var(--card2)!important;border:1px solid var(--line)!important;color:var(--fg)!important;width:100%%;}
[data-testid="stWidgetLabel"] label {color:var(--muted)!important;font-size:.83rem;font-weight:650;}
[data-testid="stSegmentedControl"] {background:var(--card)!important;padding:5px!important;border-radius:12px;border:1px solid var(--line);}
[data-testid="stSegmentedControl"] button {border-radius:9px!important;}
[data-testid="stSelectbox"] > div > div {border-color:var(--line)!important;background:var(--card)!important;color:var(--fg)!important;}
[data-baseweb="select"] > div {background:var(--card)!important;color:var(--fg)!important;border:1px solid var(--line)!important;border-radius:10px;}
div[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:var(--line)!important;}
.brand {display:flex;align-items:center;gap:12px;padding:5px 0 0;}
.brand .leaf-mark {display:flex;width:39px;height:39px;align-items:center;justify-content:center;border-radius:12px;background:var(--card2);font-size:23px;}
.brand .name {font-size:1.2rem;font-weight:850;letter-spacing:-.04em;line-height:1.12;}
.brand .sub {font-size:.74rem;color:var(--muted);margin-top:2px;}
.hero {position:relative;height:322px;border-radius:22px;background-image:%s,url('data:image/jpeg;base64,%s');background-size:cover;background-position:center;overflow:hidden;border:1px solid var(--line);display:flex;align-items:center;padding:30px 29px;margin:0 0 8px;}
.hero-content {max-width:420px;position:relative;z-index:2;}
.eyebrow {letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:850;font-size:.69rem;margin-bottom:13px;}
.hero h1 {font-size:clamp(2rem,3.3vw,3.7rem);line-height:1.04;letter-spacing:-.065em;margin:8px 0 14px;color:var(--fg);font-weight:870;}
.hero h1 em {color:var(--accent);font-style:normal;}
.hero p {font-size:.94rem;line-height:1.6;color:var(--muted);max-width:360px;margin:0;}
.hero-stats {display:flex;gap:10px;flex-wrap:wrap;margin-top:24px;}
.hero-stats span {font-size:.75rem;font-weight:760;background:var(--card);border:1px solid var(--line);color:var(--fg);border-radius:999px;padding:7px 11px;}
.section-header {font-size:1.15rem;font-weight:780;letter-spacing:-.03em;color:var(--fg);margin:0 0 6px;}
.section-description {color:var(--muted);font-size:.85rem;line-height:1.5;margin:0 0 16px;}
.metric-card {background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px;min-height:116px;position:relative;overflow:hidden;}
.metric-card::after {content:'✳';position:absolute;right:16px;top:7px;font-size:24px;color:var(--accent);opacity:.44;}
.metric-icon {font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;font-weight:750;}
.metric-value {font-size:2.15rem;letter-spacing:-.06em;font-weight:870;color:var(--accent);line-height:1.2;margin-top:9px;}
.metric-desc {font-size:.78rem;color:var(--muted);}
.info-chip {display:inline-block;padding:7px 11px;background:var(--card2);color:var(--fg);border:1px solid var(--line);border-radius:8px;font-size:.76rem;font-weight:800;margin:5px 0 10px;}
.pre-result {min-height:172px;border:1px dashed var(--line);border-radius:13px;background:var(--card2);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:18px;text-align:center;}
.pre-result .symbol {font-size:45px;margin-bottom:10px;}
.pre-result strong {font-size:1.05rem;color:var(--fg);}
.pre-result small {display:block;color:var(--muted);margin-top:7px;line-height:1.5;max-width:240px;}
.result-title {font-size:1.27rem;font-weight:850;letter-spacing:-.025em;margin:4px 0;color:var(--fg);}
.result-caption {font-size:.75rem;color:var(--muted);}
.confidence {font-size:2.05rem;color:var(--accent);font-weight:860;letter-spacing:-.06em;}
.conf-track {height:9px;border-radius:30px;background:var(--card2);overflow:hidden;margin:8px 0 14px;}
.conf-fill {height:100%%;background:var(--accent);border-radius:30px;}
.rank-row {display:flex;justify-content:space-between;gap:18px;font-size:.79rem;color:var(--muted);padding-top:7px;}
.rank-row b {color:var(--fg);}
.divider-label {color:var(--accent);font-weight:800;font-size:.75rem;letter-spacing:.09em;text-transform:uppercase;margin:6px 0 13px;}
.note-box {background:var(--card2);border:1px solid var(--line);padding:13px 16px;border-radius:12px;line-height:1.54;font-size:.85rem;color:var(--muted);}
.footnote {font-size:.73rem;color:var(--muted);line-height:1.65;}
.footer-line {text-align:center;color:var(--muted);font-size:.76rem;padding:24px 0 5px;border-top:1px solid var(--line);margin-top:24px;}
[data-testid="stImage"] img {border-radius:12px;}
@media (max-width:900px) {.main .block-container {padding-left:.9rem;padding-right:.9rem;} .hero {height:285px;padding:22px;} .hero h1{font-size:2.5rem;} .metric-card {min-height:105px;padding:13px;} }
</style>
""" % (
        theme["bg"], theme["card"], theme["card2"], theme["text"],
        theme["muted"], theme["line"], theme["accent"], theme["accenttext"],
        theme["hero"], hero_image_data(),
    )


def metric(title, value, description):
    return f"""<div class="metric-card">
        <div class="metric-icon">{safe_html(title)}</div>
        <div class="metric-value">{safe_html(value)}</div>
        <div class="metric-desc">{safe_html(description)}</div>
    </div>"""


def summary_metrics():
    columns = st.columns(4, gap="small")
    values = [
        ("Custom CNN", "95.56%", "Validation accuracy"),
        ("MobileNetV2", "96.53%", "Validation accuracy · deployed"),
        ("Classes", "38", "Healthy and disease categories"),
        ("Plant groups", "14", "Crops covered by training data"),
    ]
    for column, (title, value, description) in zip(columns, values):
        with column:
            st.markdown(metric(title, value, description), unsafe_allow_html=True)


def read_upload(upload):
    """Check actual file data, not just the filename extension."""
    data = upload.getvalue()
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("File is too large. Please upload a JPG or PNG below 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as check:
            if check.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPEG and PNG images are supported.")
            if check.width * check.height > 20_000_000:
                raise ValueError("This image is too large. Please use an image below 20 megapixels.")
            check.verify()
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.load()
            return image
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The image could not be read. Try another JPG or PNG.") from error


def health_guidance(class_name):
    """General visual-inspection suggestions, not species-specific treatment advice."""
    plant, condition = format_class(class_name)
    lower = condition.lower()
    if lower == "healthy":
        return ("The model classified this leaf as healthy. Continue monitoring for changes, "
                "and remember that a single leaf photo cannot confirm the plant's overall health.")
    if any(word in lower for word in ("virus", "mosaic", "curl")):
        return ("Compare symptoms on several leaves, check nearby plants, and consult a local plant "
                "health specialist for confirmation. Avoid moving possibly affected plant material.")
    if any(word in lower for word in ("mite", "spider")):
        return ("Examine the undersides of leaves for pests and webbing. Monitor nearby plants "
                "and ask a local agricultural adviser about appropriate management.")
    return ("Examine other leaves for similar symptoms, improve airflow where appropriate, "
            "avoid keeping foliage wet unnecessarily, and seek local advice before applying treatments.")


def prediction_card(prediction):
    if prediction is None:
        st.markdown('<div class="divider-label">◎ Analysis result</div>', unsafe_allow_html=True)
        st.markdown("""<div class="pre-result">
            <div class="symbol">🌱</div>
            <strong>Your prediction appears here</strong>
            <small>Upload a leaf image and click <b>Analyze Leaf</b> to get a real MobileNetV2 result.</small>
        </div>""", unsafe_allow_html=True)
        st.markdown("<p class='footnote'>No example accuracy is shown as a real prediction. Results appear only after you run the model.</p>", unsafe_allow_html=True)
        return

    best_class, best_score = prediction["top3"][0]
    plant, condition = format_class(best_class)
    is_healthy = condition.lower() == "healthy"
    st.markdown('<div class="divider-label">✓ Analysis complete</div>', unsafe_allow_html=True)
    st.image(prediction["image"], width="stretch")
    st.markdown(f"""<div class="info-chip">{'✓ HEALTHY CATEGORY' if is_healthy else '◉ DISEASE CATEGORY'}</div>
        <div class="result-title">{safe_html(plant)} — {safe_html(condition)}</div>
        <div class="result-caption">MobileNetV2 softmax confidence</div>
        <div class="confidence">{best_score * 100:.1f}%</div>
        <div class="conf-track"><div class="conf-fill" style="width:{min(best_score * 100, 100):.2f}%"></div></div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider-label'>Top 3 predictions</div>", unsafe_allow_html=True)
    for cls, score in prediction["top3"]:
        label_plant, label_condition = format_class(cls)
        st.markdown(
            f'<div class="rank-row"><span>{safe_html(label_plant)} · {safe_html(label_condition)}</span>'
            f'<b>{score*100:.1f}%</b></div>', unsafe_allow_html=True,
        )
    st.markdown("<br><div class='divider-label'>What to do next</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="note-box">{safe_html(health_guidance(best_class))}</div>', unsafe_allow_html=True)
    st.caption("Model confidence is not certainty. Results are informational, not a confirmed plant diagnosis.")

    text_report = (
        "PLANT DISEASE DETECTION — RESULT\n"
        "Model: saved MobileNetV2 (PyTorch)\n\n"
        + "\n".join(f"{format_class(c)[0]} — {format_class(c)[1]}: {score * 100:.2f}%"
                    for c, score in prediction["top3"])
        + "\n\nNote: predictions are not a confirmed diagnosis.\n"
    )
    st.download_button("↓ Download text report", data=text_report,
                       file_name="leaf_prediction.txt", mime="text/plain", use_container_width=True)


def upload_card():
    st.markdown('<div class="section-header">◉ Analyze a leaf</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Upload a JPG or PNG to classify your leaf using the actual saved model.</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"],
                              key="leaf_upload", label_visibility="collapsed", help="Maximum 10 MB")
    st.markdown('<div class="info-chip">⚡ MobileNetV2 · 96.53% validation accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="footnote">Image size: 160 × 160 · Preprocessing matches your Kaggle notebook · 38 output classes</div>', unsafe_allow_html=True)

    if upload is not None:
        signature = hashlib.sha256(upload.getvalue()).hexdigest()
        if st.session_state.prediction is not None and st.session_state.prediction["signature"] != signature:
            st.session_state.prediction = None
        st.caption(f"Ready: {upload.name} ({len(upload.getvalue()) / 1024:.0f} KB)")
    else:
        st.session_state.prediction = None

    if st.button("✦ Analyze Leaf", type="primary", use_container_width=True, disabled=upload is None):
        try:
            image = read_upload(upload)
            with st.spinner("Classifying with your MobileNetV2 checkpoint..."):
                model, class_names = cached_model()
                top3 = predict_image(image, model, class_names)
            st.session_state.prediction = {"signature": signature, "top3": top3, "image": image}
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            st.error(f"Prediction failed: {error}")
    st.markdown('<p class="footnote">🔒 The image is processed during this session and is not uploaded to a separate prediction API.</p>', unsafe_allow_html=True)


def chart_preview():
    st.markdown('<div class="section-header">Training insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Actual curves exported from your final Kaggle notebook.</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Accuracy over training**")
        st.image(str(ACCURACY_CHART), width="stretch")
    with right:
        st.markdown("**Loss over training**")
        st.image(str(LOSS_CHART), width="stretch")


def hero_section():
    st.markdown("""<section class="hero">
      <div class="hero-content">
        <div class="eyebrow">✦ AI × AGRICULTURE</div>
        <h1>Plant Disease<br><em>Detection</em></h1>
        <p>Understand leaf health with a PyTorch model trained to recognize 38 healthy and disease categories.</p>
        <div class="hero-stats"><span>⚡ MobileNetV2</span><span>38 classes</span><span>14 plant groups</span></div>
      </div>
    </section>""", unsafe_allow_html=True)


def home_page():
    left, middle, right = st.columns([1.02, 1.04, 1.04], gap="medium", vertical_alignment="top")
    with left:
        hero_section()
    with middle:
        with st.container(border=True):
            upload_card()
    with right:
        with st.container(border=True):
            prediction_card(st.session_state.prediction)
    st.write("")
    summary_metrics()
    st.write("")
    chart_preview()


def analyze_page():
    st.markdown("## Analyze a plant leaf")
    st.caption("Real inference from your saved mobilenet_best.pth checkpoint. No FastAPI required.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            upload_card()
    with right:
        with st.container(border=True):
            prediction_card(st.session_state.prediction)


def insights_page():
    st.markdown("## Model comparison & training insights")
    st.caption("From the final notebook and its saved validation-loss checkpoints. These are not test-set or field accuracies.")
    summary_metrics()
    st.write("")
    chart_preview()
    st.markdown("### What the numbers mean")
    st.markdown("""- **Custom CNN:** 95.56% validation accuracy at its saved best-validation-loss checkpoint (epoch 14).
- **MobileNetV2:** 96.53% validation accuracy at its saved best-validation-loss checkpoint (epoch 8).
- **Deployment choice:** MobileNetV2 because it had the higher reported validation accuracy and is small enough for a simple Streamlit app.

The dataset contains augmented leaf images and uses its provided validation split. These scores **do not guarantee the same accuracy on photos from new farms or phones**.""")


def about_page():
    st.markdown("## About LeafLens")
    st.caption("A beginner-friendly PyTorch computer vision portfolio project")
    st.markdown("""This website uses your **actual trained MobileNetV2 weights**, not a mocked prediction or external AI service. The site loads the model in CPU mode, resizes the photo to **160 × 160**, applies the same **ImageNet normalization** from training, and predicts one of **38** dataset classes.

**Pipeline:** Upload image → preprocess → MobileNetV2 → softmax scores → results.

**Training setup:** Kaggle GPU · PyTorch · frozen MobileNetV2 feature extractor · custom classifier · early stopping · no additional augmentation.""")
    st.markdown("### Supported plant groups")
    classes = json.loads(CLASS_FILE.read_text(encoding="utf-8"))
    plants = sorted({format_class(label)[0] for label in classes})
    st.write(" · ".join(plants))
    st.markdown("### Browse the 38 model classes")
    choice = st.selectbox("Select class", classes,
                          format_func=lambda label: " — ".join(format_class(label)))
    plant, condition = format_class(choice)
    st.markdown(f'<div class="note-box"><b>{safe_html(plant)} — {safe_html(condition)}</b><br>Class stored as: {safe_html(choice)}</div>',
                unsafe_allow_html=True)
    st.markdown("### Project limitations")
    st.info("The prediction is based only on a single photo. It cannot verify disease cause, plant identity, or treatment suitability. Consult local agricultural expertise for important decisions.")


# ---- Header / theme / navigation ----
# The dashboard preview uses a sidebar; the other two match the top-bar preview.
if st.session_state.visual_style == "AgriTech Dashboard":
    with st.sidebar:
        st.markdown("### 🌿 LeafLens")
        st.caption("Plant Disease Detection")
        st.radio("Navigation", NAV_OPTIONS, key="page")
        st.selectbox("Website design", STYLE_OPTIONS, key="visual_style")
    st.markdown(css_for_style(st.session_state.visual_style), unsafe_allow_html=True)
    st.markdown('<div class="brand"><span class="leaf-mark">🌿</span><div><div class="name">Plant Disease Detection</div><div class="sub">AgriTech Dashboard · PyTorch + MobileNetV2</div></div></div>', unsafe_allow_html=True)
    st.write("")
else:
    st.markdown(css_for_style(st.session_state.visual_style), unsafe_allow_html=True)
    logo, navigation, picker = st.columns([1.24, 1.8, 1.0], gap="medium", vertical_alignment="center")
    with logo:
        st.markdown('<div class="brand"><span class="leaf-mark">🌿</span><div><div class="name">LeafLens</div><div class="sub">PLANT DISEASE DETECTION</div></div></div>', unsafe_allow_html=True)
    with navigation:
        st.segmented_control("Navigation", NAV_OPTIONS, key="page", label_visibility="collapsed")
    with picker:
        st.selectbox("Design", STYLE_OPTIONS, key="visual_style", label_visibility="collapsed")
    st.write("")

if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Analyze":
    analyze_page()
elif st.session_state.page == "Insights":
    insights_page()
else:
    about_page()

st.markdown("<div class='footer-line'>🌿 LeafLens · Built with Streamlit and PyTorch · Your saved MobileNetV2 checkpoint · Predictions are informational</div>", unsafe_allow_html=True)
