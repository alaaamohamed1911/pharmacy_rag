"""
Core RAG logic for the pharmacy assistant — Streamlit-friendly version.

Difference from the notebook: instead of a local sentence-transformers
embedding model + local CrossEncoder + Chroma, everything here goes through
the Cohere API (embed + rerank + chat), reading a pre-built kb_index.json
(chunk text + metadata + Cohere embedding per chunk) produced by the
notebook's export cell. This avoids downloading ~470MB+ of local models on
every Streamlit Cloud cold start.
"""
import json
import os
from typing import List, Dict, Optional

import numpy as np
import cohere

EMBED_MODEL = "embed-multilingual-v3.0"   # must match the notebook's export cell
RERANK_MODEL = "rerank-multilingual-v3.0"
CHAT_MODEL = "command-r-plus-08-2024"

TOP_K = 12
FINAL_TOP_N = 4
MIN_SIMILARITY = 0.20
# Cohere's rerank returns relevance_score in [0, 1] (not the negative
# cross-encoder scale from the notebook) — 0.20 is a reasonable starting
# abstention line; tune it against a few known-answerable/unanswerable
# questions the same way the notebook does.
RERANK_ABSTAIN_THRESHOLD = 0.02

SYSTEM_PROMPT = """You are a careful pharmacy knowledge assistant.

The knowledge base contains:
1. Drug-specific pharmacy information for medications commonly sold/used in Egypt.
2. Selected official Egyptian Drug Authority (EDA) documents (antimicrobial
   prophylaxis, pharmacovigilance, non-prescription/OTC medicines).

Rules:
1. Answer ONLY from the CONTEXT. Never use outside knowledge, even if you are
   confident it's correct.
2. If the requested information is not explicitly in the CONTEXT, say:
   "This information is not available in the knowledge base." Do not guess.
3. Every factual claim must carry the exact [SOURCE: ...] tag from the CONTEXT
   that supports it. Never invent a source tag. Do not cite a source you did
   not actually use.
4. Do not combine separate facts into a new conclusion unless the source
   explicitly connects them. For interactions between two drugs, answer only
   if the CONTEXT explicitly discusses that specific interaction.
5. The knowledge base contains SELECTED EDA documents only — never imply it
   represents the complete Egyptian Drug Authority database.
6. Reply in the SAME language as the user's question. In Arabic, keep precise
   clinical terms precise (e.g. نقص سكر الدم لا "مشكلة في السكر").
7. Be concise: plain sentences or short bullets, no markdown headers, bold
   only for a drug name or a single critical warning.
8. End every answer with a short reminder that this is not a substitute for a
   licensed pharmacist or physician and that patient-specific factors should
   be checked by a professional.
"""


class RagIndex:
    """Loads kb_index.json once and holds chunk texts / metadata / vectors."""

    def __init__(self, path: str, api_key: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.texts = [c["text"] for c in data]
        self.metadatas = [c["metadata"] for c in data]
        self.vectors = np.array([c["embedding"] for c in data], dtype="float32")
        # normalize once so retrieval is a plain dot product (cosine sim)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms

        self.co = cohere.ClientV2(api_key=api_key)

    # ---- retrieval -----------------------------------------------------
    def embed_query(self, text: str) -> np.ndarray:
        resp = self.co.embed(
            texts=[text],
            model=EMBED_MODEL,
            input_type="search_query",
            embedding_types=["float"],
        )
        vec = np.array(resp.embeddings.float[0], dtype="float32")
        vec = vec / (np.linalg.norm(vec) or 1.0)
        return vec

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        qvec = self.embed_query(question)
        sims = self.vectors @ qvec
        order = np.argsort(-sims)[:top_k]

        hits = []
        for i in order:
            sim = float(sims[i])
            if sim >= MIN_SIMILARITY:
                hits.append({
                    "text": self.texts[i],
                    "metadata": self.metadatas[i],
                    "similarity": sim,
                })
        return hits

    # ---- rerank ----------------------------------------------------------
    def rerank(self, question: str, hits: List[Dict], top_n: int = FINAL_TOP_N) -> List[Dict]:
        if not hits:
            return []

        resp = self.co.rerank(
            model=RERANK_MODEL,
            query=question,
            documents=[h["text"] for h in hits],
            top_n=min(top_n, len(hits)),
        )

        reranked = []
        for r in resp.results:
            h = hits[r.index]
            h["rerank_score"] = float(r.relevance_score)
            reranked.append(h)

        reranked.sort(key=lambda h: h["rerank_score"], reverse=True)
        return reranked

    # ---- context building -------------------------------------------------
    def build_context(self, hits: List[Dict]) -> str:
        seen = set()
        blocks = []
        for h in hits:
            meta = h["metadata"]
            source_type = meta.get("source_type", "unknown")
            source_doc = meta.get("source_doc", "")
            drug_name = meta.get("drug_name", "")
            section = meta.get("section", "")

            # FIX: some EDA chunks were built with several section headings
            # merged into one metadata string (comma-separated), e.g.
            # "3.1 ... , 3.3 ... , 3.4 ... , 4. ...". That produces a long,
            # ugly [SOURCE: ...] tag that gets repeated verbatim by the LLM
            # for every cited fact. We only ever want ONE section label per
            # chunk, so take just the first one here — this is applied
            # before the tag is built, so the tag the LLM sees (and cites)
            # is already clean and consistent.
            if "," in section:
                section = section.split(",")[0].strip()

            key = (source_type, source_doc, drug_name, section)
            if key in seen:
                continue
            seen.add(key)

            if source_type == "eda":
                tag = f"[SOURCE: EDA | {source_doc} | Section: {section}]"
            elif source_type == "drug_knowledge":
                tag = f"[SOURCE: Drug Knowledge Base | {drug_name} — {section}]"
            else:
                tag = f"[SOURCE: {source_doc} | {section}]"

            blocks.append(f"{tag}\n{h['text']}")
        return "\n\n---\n\n".join(blocks)

    # ---- LLM calls ---------------------------------------------------------
    def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        response = self.co.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return "".join(
            block.text for block in response.message.content
            if getattr(block, "type", None) == "text"
        )

    def is_rag_question(self, question: str) -> bool:
        system_prompt = (
            "Classify the user question. Return ONLY one word.\n"
            "YES = related to pharmacy, medications, drug safety/interactions, "
            "dosage, OTC medicines, adverse drug reactions, pharmacovigilance, "
            "antimicrobial prophylaxis, or Egyptian Drug Authority (EDA) documents.\n"
            "NO = casual conversation, asking about the assistant itself, "
            "greetings, general chat, unrelated knowledge.\n"
            "Return ONLY: YES or NO."
        )
        result = self.call_llm(system_prompt, question, temperature=0.0)
        return result.strip().upper().startswith("YES")

    def rewrite_query(self, question: str, chat_history: Optional[List[str]]) -> str:
        history_block = ""
        if chat_history:
            history_block = "Previous conversation:\n" + "\n".join(chat_history) + "\n\n"

        system_prompt = (
            "You rewrite a user's pharmacy question into ONE short, clear, "
            "standalone search query, in the SAME language. Do not answer the "
            "question. Resolve pronouns using the conversation history if given. "
            "Output ONLY the rewritten query."
        )
        user_prompt = f"{history_block}Question: {question}\nRewritten standalone query:"
        return self.call_llm(system_prompt, user_prompt, temperature=0.0).strip().strip('"')

    # ---- full pipeline ------------------------------------------------------
    def rag_query(self, question: str, chat_history: Optional[List[str]] = None) -> Dict:
        """Returns {'answer': str, 'chunks': List[Dict], 'abstained': bool}."""

        if not self.is_rag_question(question):
            return {
                "answer": "السؤال ده خارج نطاق قاعدة المعرفة (الصيدلة / هيئة الدواء المصرية).",
                "chunks": [],
                "abstained": True,
            }

        search_query = question
        if chat_history or len(question.split()) <= 3:
            search_query = self.rewrite_query(question, chat_history)

        hits = self.retrieve(search_query)
        if not hits:
            return {
                "answer": "This information is not available in the knowledge base.\n\n"
                          "This information is not a substitute for a licensed pharmacist or physician.",
                "chunks": [],
                "abstained": True,
                "debug": "No chunks passed MIN_SIMILARITY — retrieval found nothing close enough.",
            }

        top_hits = self.rerank(search_query, hits)
        if not top_hits:
            return {
                "answer": "This information is not available in the knowledge base.\n\n"
                          "This information is not a substitute for a licensed pharmacist or physician.",
                "chunks": hits,
                "abstained": True,
                "debug": "Rerank returned nothing (unexpected — check Cohere rerank call).",
            }

        best_score = max(h["rerank_score"] for h in top_hits)
        if best_score < RERANK_ABSTAIN_THRESHOLD:
            return {
                "answer": "This information is not available in the knowledge base.\n\n"
                          "This information is not a substitute for a licensed pharmacist or physician.",
                "chunks": top_hits,
                "abstained": True,
                "debug": f"best_rerank_score={best_score:.3f} < threshold={RERANK_ABSTAIN_THRESHOLD}",
            }

        context = self.build_context(top_hits)

        # Detect the ORIGINAL question's language (not the possibly-rewritten
        # search_query) and inject an explicit, hard instruction. Relying on
        # the system prompt's general "reply in the same language" rule alone
        # was not reliable enough — the model sometimes answered in Arabic
        # even when the user asked in English.
        has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in question)
        language_instruction = (
            "The user's question is written in ARABIC. You MUST write your "
            "entire answer in Arabic (including the safety reminder at the "
            "end). Do not switch to English."
            if has_arabic else
            "The user's question is written in ENGLISH. You MUST write your "
            "entire answer in English (including the safety reminder at the "
            "end). Do not switch to Arabic."
        )

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"{language_instruction}\n\n"
            "Reminder: cite the exact [SOURCE: ...] tag for every factual claim.\n\n"
            "ANSWER:"
        )
        answer = self.call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)

        return {"answer": answer, "chunks": top_hits, "abstained": False}
