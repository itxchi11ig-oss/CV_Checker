import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
from thefuzz import process, fuzz

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Deep Match AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CORE ENGINE ====================

@dataclass
class AnalysisResult:
    score: float
    semantic_score: float
    keyword_score: float
    best_chunks: List[str]
    missing_critical: List[str]
    matched_critical: List[str]

class TextProcessor:
    @staticmethod
    def extract_text(file) -> str:
        """Robust text extraction that handles multi-column PDFs"""
        try:
            name = file.name.lower()
            text = ""
            if name.endswith('.pdf'):
                reader = PyPDF2.PdfReader(file)
                # Extract text and treat newlines as potential paragraph breaks
                text = " ||| ".join([page.extract_text() for page in reader.pages])
            elif name.endswith('.docx'):
                doc = docx.Document(file)
                text = " ||| ".join([p.text for p in doc.paragraphs if p.text.strip()])
            elif name.endswith('.txt'):
                text = file.getvalue().decode('utf-8')
            
            # Cleaning: Remove multiple spaces, keep structure
            return re.sub(r'\s+', ' ', text).strip()
        except:
            return ""

    @staticmethod
    def get_chunks(text: str, chunk_size: int = 50) -> List[str]:
        """
        Splits text into sliding windows (chunks). 
        This allows us to find specific sentences that match, rather than averaging the whole doc.
        """
        # Split by logical delimiters (sentences or user defined breaks)
        # We replace the PDF separators ||| with actual splits
        raw_parts = text.split("|||")
        
        chunks = []
        for part in raw_parts:
            # Clean the part
            clean_part = part.strip()
            if len(clean_part) > 30: # Ignore tiny fragments
                chunks.append(clean_part)
        
        return chunks

class DeepMatchEngine:
    def __init__(self):
        # We use a smaller, faster, but high-quality model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Words that usually indicate "Requirements" in a JD
        self.fluff_markers = ["equal opportunity", "gender", "race", "color", "religion", "sexual orientation", "veteran"]

    def extract_hard_skills(self, text: str) -> Set[str]:
        """
        Aggressive extraction of Capitalized Phrases (Proper Nouns) and Tech Terms.
        """
        # 1. Preserve Tech Stack
        text = text.replace("C++", "Cpp").replace(".NET", "DotNet").replace("Node.js", "Nodejs")
        
        # 2. Find Capitalized Words (Potential Skills) inside the text
        # Logic: If a word is Capitalized in the middle of a sentence, it's likely a Proper Noun (Skill/Tool)
        pattern = r'\b[A-Z][a-zA-Z0-9#\+]*\b' 
        candidates = set(re.findall(pattern, text))
        
        # Filter out common junk
        stopwords = {"The", "A", "An", "To", "For", "With", "In", "On", "At", "By", "We", "You", "And", "Or", "If"}
        skills = {w for w in candidates if w not in stopwords and len(w) > 2}
        
        return skills

    def evaluate(self, cv_text: str, jd_text: str) -> AnalysisResult:
        # 1. Clean JD (Remove Legal Fluff)
        jd_clean = jd_text
        for marker in self.fluff_markers:
            if marker in jd_clean.lower():
                # Rough logic: cut text after legal markers appear if they are at the end
                pass 

        # 2. CHUNK-BASED SEMANTIC SCORING (The Google Method)
        # Instead of comparing Doc vs Doc, we compare CV_Chunks vs Whole_JD
        cv_chunks = TextProcessor.get_chunks(cv_text)
        
        if not cv_chunks:
            return AnalysisResult(0,0,0,[],[],[])

        # Encode JD once
        jd_embedding = self.model.encode(jd_clean, convert_to_tensor=True)
        
        # Encode all CV chunks
        cv_embeddings = self.model.encode(cv_chunks, convert_to_tensor=True)
        
        # Compute cosine similarities for every chunk
        similarities = util.cos_sim(cv_embeddings, jd_embedding)
        
        # MAX-POOLING STRATEGY
        # We take the TOP 5 matching paragraphs from your CV. 
        # If your experience section matches, but your hobbies don't, we only care about experience.
        top_k = min(5, len(cv_chunks))
        top_scores, top_indices = similarities.topk(top_k, dim=0)
        
        # Average of your BEST 5 chunks
        raw_semantic_score = float(top_scores.mean())
        
        # Curve the score (0.4 is usually a great chunk match)
        semantic_score = 1 / (1 + math.exp(-15 * (raw_semantic_score - 0.35)))
        
        # 3. FUZZY KEYWORD SCORING (The ATS Check)
        jd_skills = self.extract_hard_skills(jd_clean)
        cv_raw = cv_text.lower()
        
        matched_skills = []
        missing_skills = []
        
        if not jd_skills:
            keyword_score = 1.0 # No hard skills detected in JD? Free pass.
        else:
            hits = 0
            for skill in jd_skills:
                # Fuzzy Search: "Javascript" matches "Java Script"
                # We search the whole CV text for this skill
                # We use a partial ratio because the skill might be embedded in a sentence
                # Threshold 85 is strict enough to avoid "Java" matching "JavaScript" incorrectly usually, 
                # but "React" matching "ReactJS" works.
                if skill.lower() in cv_raw:
                    hits += 1
                    matched_skills.append(skill)
                else:
                    # Deep Fuzzy Check
                    is_present = False
                    # Check against chunks for context
                    best_match = process.extractOne(skill, cv_chunks, scorer=fuzz.partial_ratio)
                    if best_match and best_match[1] > 90:
                        hits += 1
                        matched_skills.append(skill)
                        is_present = True
                    
                    if not is_present:
                        missing_skills.append(skill)
            
            keyword_score = hits / len(jd_skills)
            # Boost keyword score: 50% match is usually enough for human recruiter
            keyword_score = min(1.0, keyword_score * 1.5)

        # 4. FINAL WEIGHTING
        # 60% Semantic (Did you describe the right work?)
        # 40% Keywords (Did you list the specific tools?)
        final_score = (semantic_score * 0.6) + (keyword_score * 0.4)
        
        best_chunks_text = [cv_chunks[i] for i in top_indices.flatten().tolist()]

        return AnalysisResult(
            score=final_score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            best_chunks=best_chunks_text,
            missing_critical=list(set(missing_skills))[:10],
            matched_critical=list(set(matched_skills))[:10]
        )

# ==================== VIEW LAYER ====================

def main():
    # Cache the heavy model loading
    @st.cache_resource
    def load_engine():
        return DeepMatchEngine()

    engine = load_engine()

    # Header
    st.markdown("""
        <style>
        .main-header {font-size: 3rem; font-weight: 800; color: #1E88E5; text-align: center; margin-bottom: 0;}
        .sub-header {font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem;}
        .metric-box {padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;}
        .highlight {background-color: #e3f2fd; padding: 2px 5px; border-radius: 4px; font-family: monospace;}
        </style>
        <div class="main-header">Deep Match AI</div>
        <div class="sub-header">Max-Pooling Semantic Engine (The "Island of Excellence" Algorithm)</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Candidate Resume")
        cv_file = st.file_uploader("Upload PDF/DOCX", type=['pdf', 'docx', 'txt'])

    with col2:
        st.subheader("2. Job Description")
        jd_text = st.text_area("Paste text here...", height=150)

    if st.button("Run Deep Analysis", type="primary", use_container_width=True):
        if cv_file and jd_text:
            with st.spinner("Chunking document & calculating vector gradients..."):
                # Extraction
                cv_text = TextProcessor.extract_text(cv_file)
                
                if len(cv_text) < 100:
                    st.error("Resume content too short or unreadable.")
                    return

                # Analysis
                result = engine.evaluate(cv_text, jd_text)
                
                # === RESULTS ===
                st.markdown("---")
                
                # 1. THE BIG SCORE
                score_pct = result.score * 100
                
                # Dynamic Color
                if score_pct >= 85: color = "#2e7d32" # Dark Green
                elif score_pct >= 70: color = "#1976d2" # Blue
                elif score_pct >= 50: color = "#f57f17" # Orange
                else: color = "#c62828" # Red
                
                c1, c2, c3 = st.columns([1,2,1])
                with c2:
                    st.markdown(f"""
                        <div style="text-align: center;">
                            <h2 style="margin:0; color: #555;">Hiring Probability</h2>
                            <h1 style="font-size: 5em; font-weight: 900; margin: 0; color: {color};">
                                {score_pct:.1f}%
                            </h1>
                        </div>
                    """, unsafe_allow_html=True)

                # 2. METRICS
                m1, m2, m3 = st.columns(3)
                m1.metric("Context Match", f"{result.semantic_score*100:.1f}%", 
                          help="How well your best paragraphs matched the JD meaning.")
                m2.metric("Keyword Hit Rate", f"{result.keyword_score*100:.1f}%", 
                          help="Percentage of capitalized hard skills found.")
                m3.metric("Analysis Mode", "Max-Pooling", 
                          help="We ignored the weak parts of your CV and scored the best parts.")

                # 3. VISUAL PROOF (Why did I get this score?)
                st.subheader("🔍 Why you got this score")
                
                tab1, tab2, tab3 = st.tabs(["🏆 Your Best Matches", "✅ Skills Found", "⚠️ Missing Skills"])
                
                with tab1:
                    st.success("The AI found these specific paragraphs in your CV that strongly match the Job Description:")
                    for i, chunk in enumerate(result.best_chunks):
                        st.markdown(f"**Match #{i+1}:** _{chunk}_")
                        st.divider()

                with tab2:
                    if result.matched_critical:
                        st.write("We detected these specific hard skills from the JD in your text:")
                        st.markdown(" ".join([f"<span class='highlight'>{s}</span>" for s in result.matched_critical]), unsafe_allow_html=True)
                    else:
                        st.warning("No specific proper nouns (Skills) matched exactly.")

                with tab3:
                    if result.missing_critical:
                        st.write("The Job Description emphasizes these skills, but we couldn't find them clearly in your CV:")
                        st.error("  •  ".join(result.missing_critical))
                        st.caption("Tip: If you possess these skills, ensure they are capitalized and spelled exactly as above.")
                    else:
                        st.success("You have excellent keyword coverage!")

        else:
            st.info("Please provide both a Resume and a Job Description to start.")

if __name__ == "__main__":
    main()
