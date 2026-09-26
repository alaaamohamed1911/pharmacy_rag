# Pharmacy RAG — Streamlit App

## الملفات
- `app.py` — واجهة Streamlit (ثيم صيدلية: أخضر/تركواز، هيدر متدرّج، chat bubbles، مصادر قابلة للطي).
- `rag_core.py` — منطق الـ RAG بالكامل عن طريق Cohere API فقط (embed + rerank + chat)، بيقرأ `kb_index.json`.
- `kb_index.json` — **لازم تحطه أنت هنا** (تصدير من النوت بوك، خلية "Optional — Export chunk embeddings for an external app").
- `requirements.txt`, `.streamlit/config.toml`.

## خطوات الـ deploy

1. **صدّر قاعدة المعرفة من النوت بوك**
   شغّل خلية "Optional — Export chunk embeddings for an external app (Cohere)" لحد ما تاخد `kb_index.json`، وحطّه في نفس فولدر `app.py`.

2. **اعمل ريبو على GitHub**
   ```bash
   git init
   git add app.py rag_core.py requirements.txt kb_index.json .streamlit/config.toml README.md
   git commit -m "pharmacy rag streamlit app"
   git remote add origin <رابط الريبو بتاعك>
   git push -u origin main
   ```
   لو `kb_index.json` كبير جدًا (أكتر من ~100MB) استخدم Git LFS، أو ارفعه على تخزين خارجي (S3/Drive) وعدّل `load_index()` في `app.py` عشان يحمّله أول مرة (`requests.get` ثم يحفظه محليًا) بدل ما يكون داخل الريبو.

3. **روح على [streamlit.io/cloud](https://streamlit.io/cloud)**
   - سجّل دخول بحساب GitHub.
   - "New app" → اختار الريبو والـ branch → main file: `app.py`.

4. **ضيف الـ Secret**
   في إعدادات التطبيق (Settings → Secrets) حط:
   ```toml
   COHERE_API_KEY = "your-cohere-key-here"
   ```

5. **Deploy** — أول تشغيل هياخد شوية ثواني بس (مفيش موديلات محلية تتحمّل، كل حاجة API).

## تجربة محلية قبل الرفع
```bash
pip install -r requirements.txt
export COHERE_API_KEY=your-key
streamlit run app.py
```

## ملاحظات
- الـ abstention threshold (`RERANK_ABSTAIN_THRESHOLD` في `rag_core.py`) مبني على مقياس Cohere Rerank (0 إلى 1)، مختلف عن الـ CrossEncoder المحلي في النوت بوك. جرّبه على كام سؤال معروف الإجابة وظبطه.
- لو عايز تقلل تكلفة الـ API calls، احذف استدعاء `is_rag_question` أو `rewrite_query` لو مش محتاجهم لأول نسخة.
