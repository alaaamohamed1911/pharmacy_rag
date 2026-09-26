import os
import streamlit as st
from rag_core import RagIndex

st.set_page_config(
    page_title="مساعد الصيدلية الذكي",
    page_icon="💊",
    layout="centered",
)

# ---------------------------------------------------------------- styling --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Cairo', sans-serif; }

.stApp {
    background: linear-gradient(180deg, #f4faf8 0%, #eef6f3 100%);
}

.ph-header {
    background: linear-gradient(120deg, #0f766e 0%, #14b8a6 100%);
    padding: 28px 24px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(15,118,110,0.25);
}
.ph-header h1 { margin: 0; font-size: 26px; font-weight: 700; }
.ph-header p  { margin: 6px 0 0; opacity: 0.9; font-size: 14px; }

.ph-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 12px;
    margin-inline-end: 6px;
}

.stChatMessage { border-radius: 14px !important; }

div[data-testid="stChatMessage"]:has(> div[data-testid="stChatMessageAvatarUser"]) {
    background: #e6f4f1;
}
div[data-testid="stChatMessage"]:has(> div[data-testid="stChatMessageAvatarAssistant"]) {
    background: #ffffff;
    border: 1px solid #e2e8e6;
}

.ph-disclaimer {
    font-size: 12px;
    color: #6b7280;
    border-inline-start: 3px solid #14b8a6;
    padding-inline-start: 10px;
    margin-top: 10px;
}

.ph-source-chip {
    display: inline-block;
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 2px 10px;
    border-radius: 8px;
    font-size: 12px;
    margin: 3px 4px 0 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- header ---
st.markdown("""
<div class="ph-header">
  <h1>💊 مساعد الصيدلية الذكي</h1>
  <p>
    <span class="ph-badge">Egyptian Drug Authority</span>
    <span class="ph-badge">RAG</span>
    <span class="ph-badge">مصادر موثّقة فقط</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.markdown("### 🏥 عن المساعد")
    st.write(
        "بيجاوب من قاعدة معرفة تحتوي على معلومات أدوية شائعة في مصر "
        "ووثائق مختارة من هيئة الدواء المصرية (OTC، التداخلات الدوائية، "
        "الوقاية من عدوى المواضع الجراحية، اليقظة الدوائية)."
    )
    st.markdown("### 💡 جرّب تسأل")
    for q in [
        "هل يوجد تفاعل بين الميتفورمين والأوميبرازول؟",
        "إيه موانع استعمال الإيبوبروفين؟",
        "إيه شروط تصنيف الدواء كـ OTC في مصر؟",
    ]:
        st.markdown(f"- {q}")
    st.markdown("---")
    st.markdown(
        '<p class="ph-disclaimer">⚠️ الأداة دي للمعلومات المرجعية فقط، '
        "ومش بديل عن استشارة صيدلي أو طبيب مرخّص.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- index ---
@st.cache_resource(show_spinner="بيحمّل قاعدة المعرفة...")
def load_index():
    try:
        api_key = st.secrets["COHERE_API_KEY"]
    except Exception:
        api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        st.error("لازم تضيف COHERE_API_KEY في Streamlit Secrets.")
        st.stop()
    return RagIndex("kb_index.json", api_key)

index = load_index()

# ---------------------------------------------------------------- chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander("📎 المصادر المسترجَعة"):
                for h in msg["chunks"]:
                    meta = h["metadata"]
                    label = meta.get("source_doc") or meta.get("drug_name", "")
                    section = meta.get("section", "")
                    st.markdown(
                        f'<span class="ph-source-chip">{label} — {section} '
                        f'(score={h.get("rerank_score", 0):.2f})</span>',
                        unsafe_allow_html=True,
                    )

question = st.chat_input("اكتب سؤالك عن دواء أو معلومة من هيئة الدواء...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[-6:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("بيدوّر في المصادر..."):
            result = index.rag_query(question, chat_history=history)
        st.markdown(result["answer"])
        if result.get("debug"):
            st.caption(f"🔧 debug: {result['debug']}")
        if result.get("chunks"):
            with st.expander("📎 المصادر المسترجَعة (حتى لو اتلغت)"):
                for h in result["chunks"]:
                    meta = h["metadata"]
                    label = meta.get("source_doc") or meta.get("drug_name", "")
                    section = meta.get("section", "")
                    st.markdown(
                        f'<span class="ph-source-chip">{label} — {section} '
                        f'(sim={h.get("similarity", 0):.2f}, rerank={h.get("rerank_score", 0):.2f})</span>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "chunks": result.get("chunks", []),
    })
