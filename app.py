import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import numpy as np
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from thefuzz import fuzz # pip install thefuzz

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="God Mode ATS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CORE ENGINE ====================

@dataclass
class MatchEvidence:
    requirement: str
    match_type: str  # "Semantic" or "Keyword"
    evidence: str    # The snippet from CV that triggered it
    confidence: float

@dataclass
class MatchResult:
    final_score: float
    satisfied_reqs: List[MatchEvidence]
    missing_reqs: List[str]
    feedback: str

class TextProcessor:
    @staticmethod
    def extract_text(file) -> str:
        try:
            name = file.name.lower()
            text = ""
            if name.endswith('.pdf'):
                reader = PyPDF2.PdfReader(file)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
            elif name.endswith('.docx'):
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif name.endswith('.txt'):
                text = file.getvalue().decode('utf-8')
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except: return ""

    @staticmethod
    def split_into_requirements(jd_text: str) -> List[str]:
        # Intelligent split by bullet points and newlines
        raw_chunks = re.split(r'[\n•●\-\;]', jd_text)
        cleaned_reqs = []
        for chunk in raw_chunks:
            c = chunk.strip()
            # Filter noise
            if len(c) > 10 and len(c.split()) > 2: 
                if "equal opportunity" not in c.lower():
                    cleaned_reqs.append(c)
        return cleaned_reqs

    @staticmethod
    def extract_keywords(text: str) -> set:
        # Simple extraction of significant words (len > 3)
        words = re.sub(r'[^a-zA-Z0-9\+\#]', ' ', text.lower()).split()
        return set([w for w in words if len(w) > 3])

class HybridCoverageEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def analyze(self, cv_text: str, jd_text: str) -> MatchResult:
        requirements = TextProcessor.split_into_requirements(jd_text)
        
        if not requirements:
            return MatchResult(0, [], [], "JD Empty")

        # Split CV into sentences for granular matching
        cv_sentences = re.split(r'(?<=[.!?]) +', cv_text)
        
        # Pre-compute CV embeddings
        cv_embeddings = self.model.encode(cv_sentences, convert_to_tensor=True)
        
        satisfied = []
        missing = []
        
        # === THE HYBRID LOOP ===
        for req in requirements:
            is_match = False
            best_evidence = ""
            match_type = ""
            confidence = 0.0

            # 1. SEMANTIC CHECK (The "Meaning" Check)
            req_embedding = self.model.encode(req, convert_to_tensor=True)
            cos_scores = util.cos_sim(req_embedding, cv_embeddings)[0]
            best_idx = int(torch_max_idx(cos_scores))
            semantic_score = float(cos_scores[best_idx])
            
            # Dynamic Threshold: Shorter requirements need higher exactness
            threshold = 0.35 if len(req.split()) > 5 else 0.45
            
            if semantic_score > threshold:
                is_match = True
                match_type = "Semantic (Meaning)"
                best_evidence = cv_sentences[best_idx]
                confidence = semantic_score
            
            # 2. KEYWORD FALLBACK (The "Exact Wording" Check)
            # If semantic failed, check if we have >50% keyword overlap
            if not is_match:
                req_keywords = TextProcessor.extract_keywords(req)
                if req_keywords:
                    # Search entire CV for these keywords
                    cv_keywords = TextProcessor.extract_keywords(cv_text)
                    overlap = req_keywords.intersection(cv_keywords)
                    coverage = len(overlap) / len(req_keywords)
                    
                    if coverage >= 0.6: # If you have 60% of the important words
                        is_match = True
                        match_type = "Keyword (Exact)"
                        best_evidence = f"Found keywords: {', '.join(list(overlap)[:5])}"
                        confidence = coverage

            if is_match:
                satisfied.append(MatchEvidence(req, match_type, best_evidence, confidence))
            else:
                missing.append(req)

        # SCORING CURVE
        raw_pct = len(satisfied) / len(requirements)
        
        # The "God Mode" Curve:
        # If you satisfy 40% of bullets -> You get 75% Score
        # If you satisfy 60% of bullets -> You get 90% Score
        final_score = 1 / (1 + math.exp(-8 * (raw_pct - 0.25)))

        if final_score > 0.85: f = "Top 1% Candidate"
        elif final_score > 0.70: f = "Interview Ready"
        elif final_score > 0.50: f = "Strong Potential"
        else: f = "Needs Optimization"

        return MatchResult(final_score, satisfied, missing, f)

def torch_max_idx(tensor):
    try: return tensor.argmax()
    except: return np.argmax(tensor.numpy())

# ==================== UI LAYER ====================

def main():
    st.markdown("""
        <style>
        .match-card { background-color: #e8f5e9; border-left: 5px solid #2e7d32; padding: 15px; margin-bottom: 10px; border-radius: 5px; }
        .miss-card { background-color: #ffebee; border-left: 5px solid #c62828; padding: 15px; margin-bottom: 10px; border-radius: 5px; }
        .evidence-text { color: #555; font-style: italic; font-size: 0.9em; margin-top: 5px; }
        .badge { background-color: #2196F3; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
        </style>
    """, unsafe_allow_html=True)

    st.title("⚡ God Mode ATS")
    st.caption("Hybrid Engine: Semantic Vectors + Fuzzy Keyword Fallback")
    
    col1, col2 = st.columns(2)
    cv_file = col1.file_uploader("Upload CV", type=["pdf", "docx", "txt"])
    jd_text = col2.text_area("Job Description", height=200, placeholder="Paste bullets...")

    if st.button("Run Hybrid Analysis", type="primary"):
        if cv_file and jd_text:
            with st.spinner("Running Double-Pass verification..."):
                cv_text = TextProcessor.extract_text(cv_file)
                if len(cv_text) < 50:
                    st.error("CV empty")
                    return
                
                @st.cache_resource
                def load_engine(): return HybridCoverageEngine()
                
                engine = load_engine()
                res = engine.analyze(cv_text, jd_text)
                
                score = res.final_score * 100
                color = "#00c853" if score > 75 else "#ffab00" if score > 50 else "#ff1744"
                
                st.markdown(f"""
                <div style="text-align:center; padding: 20px; background: {color}15; border: 2px solid {color}; border-radius: 15px;">
                    <h1 style="color:{color}; font-size: 4em; margin:0;">{score:.0f}%</h1>
                    <h2 style="color:#444; margin:0;">{res.feedback}</h2>
                    <p>Requirements Met: <b>{len(res.satisfied_reqs)}</b> / {len(res.satisfied_reqs)+len(res.missing_reqs)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader(f"✅ Matched Requirements ({len(res.satisfied_reqs)})")
                    for item in res.satisfied_reqs:
                        st.markdown(f"""
                        <div class="match-card">
                            <b>{item.requirement}</b><br>
                            <span class="badge">{item.match_type} Match</span>
                            <div class="evidence-text">"Found evidence: {item.evidence[:100]}..."</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                with c2:
                    st.subheader(f"❌ Unmet Requirements ({len(res.missing_reqs)})")
                    for req in res.missing_reqs:
                        st.markdown(f"""
                        <div class="miss-card">
                            <b>{req}</b><br>
                            <div class="evidence-text">Could not find semantic meaning OR keywords for this.</div>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
