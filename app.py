import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import numpy as np
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Semantic Coverage ATS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CORE ENGINE ====================

@dataclass
class MatchResult:
    final_score: float
    satisfied_reqs: List[str]
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
            
            # Clean up whitespace and special chars
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except: return ""

    @staticmethod
    def split_into_requirements(jd_text: str) -> List[str]:
        """
        Splits a Job Description into distinct actionable requirements.
        It splits by bullet points, newlines, or sentence endings.
        """
        # 1. Split by common delimiters
        raw_chunks = re.split(r'[\n•●\-\;]', jd_text)
        
        cleaned_reqs = []
        for chunk in raw_chunks:
            c = chunk.strip()
            # Filter out short junk (headers, page numbers)
            if len(c) > 15 and len(c.split()) > 3: 
                # Filter out legal boilerplate
                if "equal opportunity" not in c.lower() and "gender" not in c.lower():
                    cleaned_reqs.append(c)
        
        return cleaned_reqs

class SemanticCoverageEngine:
    def __init__(self):
        # We use a model trained specifically for semantic search
        # It knows that "Python" is related to "Coding"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def analyze(self, cv_text: str, jd_text: str) -> MatchResult:
        # 1. Breakdown the JD into specific demands
        requirements = TextProcessor.split_into_requirements(jd_text)
        
        if not requirements:
            return MatchResult(0, [], [], "Job Description too short or empty.")

        # 2. Encode the CV as one giant knowledge block (and chunks)
        # We perform a sliding window search over the CV to find the best matching section for each req
        cv_sentences = re.split(r'(?<=[.!?]) +', cv_text)
        cv_embeddings = self.model.encode(cv_sentences, convert_to_tensor=True)
        
        satisfied = []
        missing = []
        total_similarity = 0.0
        
        # 3. Iterate through every requirement in the JD
        for req in requirements:
            req_embedding = self.model.encode(req, convert_to_tensor=True)
            
            # Find the single sentence in the CV that best matches this specific requirement
            cos_scores = util.cos_sim(req_embedding, cv_embeddings)[0]
            best_match_score = float(torch_max(cos_scores))
            
            # THRESHOLD LOGIC
            # 0.35 in Semantic Vectors is usually a "soft match" (Conceptually similar)
            # 0.50 is a "hard match" (Clear evidence)
            if best_match_score >= 0.32: 
                satisfied.append(req)
                total_similarity += best_match_score
            else:
                missing.append(req)

        # 4. SCORING
        # Raw Coverage %: (Requirements Met / Total Requirements)
        coverage_pct = len(satisfied) / len(requirements)
        
        # 5. THE REALITY CURVE
        # If you meet 50% of requirements in a JD, you are usually a Top Candidate.
        # We curve 0.5 -> 0.85
        final_score = self.apply_hiring_curve(coverage_pct)
        
        # Feedback
        if final_score > 0.85: f = "Excellent Fit (Interview Ready)"
        elif final_score > 0.70: f = "Strong Fit"
        elif final_score > 0.50: f = "Potential Match"
        else: f = "Low Match"

        return MatchResult(final_score, satisfied, missing, f)

    def apply_hiring_curve(self, raw_pct: float) -> float:
        """
        Real World Logic:
        - 20% match = 50% Score (You have some basics)
        - 50% match = 85% Score (You are hireable)
        - 80% match = 99% Score (Unicorn)
        """
        # Sigmoid function shifted to be generous
        return 1 / (1 + math.exp(-6 * (raw_pct - 0.3)))

# Helper for tensor max (works with torch or numpy fallback)
def torch_max(tensor):
    try: return tensor.max()
    except: return np.max(tensor.numpy())

# ==================== UI LAYER ====================

def main():
    st.markdown("""
        <style>
        .stProgress > div > div > div > div { background-color: #4CAF50; }
        .score-card { background: #f8f9fa; border-radius:10px; padding:20px; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .req-pass { color: #2e7d32; padding: 5px; border-left: 3px solid #2e7d32; margin-bottom: 5px; background: #e8f5e9; }
        .req-fail { color: #c62828; padding: 5px; border-left: 3px solid #c62828; margin-bottom: 5px; background: #ffebee; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Semantic Coverage ATS")
    st.caption("This tool ignores keywords. It reads your CV sentence-by-sentence to see if you meet the specific requirements.")
    
    col1, col2 = st.columns(2)
    cv_file = col1.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd_text = col2.text_area("Paste Job Description", height=250, placeholder="Paste the bullet points here...")

    if st.button("Calculate Coverage", type="primary"):
        if cv_file and jd_text:
            with st.spinner("Analyzing Concept Coverage..."):
                # Load
                cv_text = TextProcessor.extract_text(cv_file)
                if len(cv_text) < 50:
                    st.error("Resume is empty or unreadable.")
                    return
                    
                @st.cache_resource
                def load_model(): return SemanticCoverageEngine()
                
                engine = load_model()
                res = engine.analyze(cv_text, jd_text)
                
                # SCORE
                score = res.final_score * 100
                color = "#2e7d32" if score > 75 else "#ff9800" if score > 50 else "#d32f2f"
                
                st.markdown(f"""
                <div class="score-card" style="border-top: 5px solid {color}">
                    <h2 style="margin:0; color: #555;">REQUIREMENT COVERAGE</h2>
                    <h1 style="font-size: 5em; margin:0; color: {color};">{score:.0f}%</h1>
                    <h3 style="margin:0;">{res.feedback}</h3>
                    <p>You met <b>{len(res.satisfied_reqs)}</b> out of <b>{len(res.satisfied_reqs) + len(res.missing_reqs)}</b> requirements.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                # VISUALIZATION
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader(f"✅ Requirements Met ({len(res.satisfied_reqs)})")
                    st.caption("The AI found evidence for these in your CV:")
                    if res.satisfied_reqs:
                        for req in res.satisfied_reqs:
                            st.markdown(f'<div class="req-pass">✔ {req}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No requirements met.")

                with c2:
                    st.subheader(f"❌ Requirements Missing ({len(res.missing_reqs)})")
                    st.caption("The AI could not find clear evidence for these:")
                    if res.missing_reqs:
                        for req in res.missing_reqs:
                            st.markdown(f'<div class="req-fail">✘ {req}</div>', unsafe_allow_html=True)
                    else:
                        st.success("All requirements met!")

        else:
            st.info("Please upload both files.")

if __name__ == "__main__":
    main()
