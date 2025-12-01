import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
from enum import Enum
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
from thefuzz import process, fuzz  # pip install thefuzz
import openai

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Ensemble CV Matcher",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Language(Enum):
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"

def get_safe_lang_code(lang) -> str:
    if isinstance(lang, Language): return lang.value
    if isinstance(lang, str): return lang
    try: return lang.value
    except AttributeError: return "en"

# ==================== ADVANCED LOGIC LAYER ====================

@dataclass
class MatchResult:
    final_score: float
    semantic_score: float
    keyword_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    feedback: str
    ai_analysis: Optional[str] = None

class TextCleaner:
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
            return text
        except:
            return ""

    @staticmethod
    def clean_jd_fluff(text: str) -> List[str]:
        """
        Splits JD into chunks to isolate 'Requirements' from 'Benefits/About Us'.
        Returns a list of chunks (paragraphs).
        """
        # Split by double newlines or long spaces
        chunks = re.split(r'\n\s*\n', text)
        return [c.strip() for c in chunks if len(c.split()) > 10]

    @staticmethod
    def extract_entities(text: str) -> Set[str]:
        """
        Extracts Proper Nouns (Capitalized Words) and Technical Terms.
        """
        # Normalize fancy quotes
        text = text.replace("’", "'").replace("“", '"')
        
        # 1. Regex for Capitalized Words (e.g. "Project Management", "Python")
        # We also allow C++, C#, .NET etc.
        pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\s[A-Z][a-zA-Z0-9]*)*\b|\b[a-zA-Z0-9\.]+\+\+\b|\bC#\b|\.NET\b'
        matches = re.findall(pattern, text)
        
        # Filter stopwords
        stopwords = {"The", "A", "An", "To", "For", "With", "In", "On", "At", "By", "We", "You", "And", "Or", "If", "Be", "Is", "Are", "This", "That", "It", "Of", "About", "Us", "Job", "Role", "Work", "Team", "Experience", "Skills", "Requirements", "Duties"}
        
        cleaned = set()
        for m in matches:
            clean_m = m.strip()
            if len(clean_m) > 2 and clean_m not in stopwords:
                cleaned.add(clean_m)
        
        return cleaned

class EnsembleMatcher:
    def __init__(self):
        # High quality multilingual model
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    def calculate_ensemble_semantic_score(self, cv_text: str, jd_text: str) -> float:
        """
        Runs TWO semantic checks and picks the winner.
        1. Whole CV vs Whole JD (Global Context)
        2. Whole CV vs JD Chunks (Did we match the 'Requirements' paragraph?)
        """
        cv_emb = self.model.encode(cv_text, convert_to_tensor=True)
        
        # Test 1: Global Match
        jd_emb_global = self.model.encode(jd_text, convert_to_tensor=True)
        score_global = float(util.cos_sim(cv_emb, jd_emb_global)[0][0])
        
        # Test 2: Local Chunk Match (The "Needle in Haystack" search)
        jd_chunks = TextCleaner.clean_jd_fluff(jd_text)
        if not jd_chunks:
            score_local = 0.0
        else:
            jd_chunk_embs = self.model.encode(jd_chunks, convert_to_tensor=True)
            cos_scores = util.cos_sim(cv_emb, jd_chunk_embs)[0]
            score_local = float(cos_scores.max()) # Take the BEST matching chunk
        
        # WINNER TAKES ALL strategy
        # If your resume matches the "Requirements" chunk perfectly (0.6), but misses the "About Us" (0.1),
        # The global average might be 0.35. We want the 0.6.
        raw_score = max(score_global, score_local)
        
        # RECRUITER CURVE (Sigmoid)
        # 0.25 -> 50%
        # 0.45 -> 85%
        # 0.65 -> 98%
        final_semantic = 1 / (1 + math.exp(-12 * (raw_score - 0.25)))
        
        return final_semantic

    def calculate_keyword_score(self, cv_text: str, jd_text: str) -> Tuple[float, List[str], List[str]]:
        """
        Fuzzy matches extracted entities (Hard Skills).
        """
        jd_skills = TextCleaner.extract_entities(jd_text)
        cv_text_lower = cv_text.lower()
        
        matched = []
        missing = []
        
        if not jd_skills:
            return 1.0, [], [] # No hard skills found in JD? Benefit of doubt.
        
        hits = 0
        for skill in jd_skills:
            # 1. Direct Search (Fast)
            if skill.lower() in cv_text_lower:
                hits += 1
                matched.append(skill)
                continue
                
            # 2. Fuzzy Search (Slow but catches "React.js" vs "React")
            # We assume CV is one giant string. We check if the skill exists roughly.
            # partial_ratio allows "React" to match inside "I used ReactJS in my project"
            if fuzz.partial_ratio(skill.lower(), cv_text_lower) >= 85:
                hits += 1
                matched.append(skill)
            else:
                missing.append(skill)
                
        score = hits / len(jd_skills)
        # Curve: 60% keyword match is usually interview-worthy
        score = min(1.0, score * 1.6)
        
        return score, sorted(list(set(matched))), sorted(list(set(missing)))

    def evaluate(self, cv_text: str, jd_text: str, lang: str) -> MatchResult:
        # 1. Semantic (Ensemble)
        sem_score = self.calculate_ensemble_semantic_score(cv_text, jd_text)
        
        # 2. Keywords (Hard Skills)
        kw_score, matched, missing = self.calculate_keyword_score(cv_text, jd_text)
        
        # 3. Final Weighting
        # If Semantic is high (you described the right experience), we care less about exact keywords.
        if sem_score > 0.85:
            final = sem_score
        else:
            final = (sem_score * 0.65) + (kw_score * 0.35)
            
        # Feedback Text
        if final > 0.85: feedback = "Excellent Match"
        elif final > 0.70: feedback = "Strong Match"
        elif final > 0.50: feedback = "Potential Match"
        else: feedback = "Weak Match"
        
        return MatchResult(final, sem_score, kw_score, matched, missing, feedback)

# ==================== GPT LAYER ====================

class GPTAnalyst:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        
    def analyze(self, cv, jd, score, missing):
        prompt = f"""
        Role: Expert Recruiter.
        Candidate Score: {score*100:.1f}% (Algorithm).
        Missing Keywords Detected: {', '.join(missing[:5])}.
        
        JD Snippet: {jd[:800]}
        CV Snippet: {cv[:800]}
        
        Task:
        1. Ignore the score. Does this candidate actually fit? (Yes/No).
        2. List 2 missing hard skills that truly matter.
        3. Suggest a 1-sentence summary to add to the CV to fix the gap.
        """
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
        except: return "GPT Error"

# ==================== UI CONTROLLER ====================

def main():
    st.sidebar.title("⚙️ Controls")
    
    lang_map = {"en": "English", "de": "Deutsch", "fr": "Français", "es": "Español", "it": "Italiano"}
    sel_lang = st.sidebar.selectbox("Language", list(lang_map.keys()), format_func=lambda x: lang_map[x])
    
    use_gpt = st.sidebar.checkbox("Use GPT-4 Audit")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password") if use_gpt else None
    
    st.title("⚖️ Ensemble CV Matcher")
    st.markdown("Runs **Global Context**, **Requirement-Specific**, and **Hard Skill** models simultaneously.")
    
    c1, c2 = st.columns(2)
    cv_file = c1.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd_text = c2.text_area("Job Description", height=250)
    
    if st.button("Run Ensemble Analysis", type="primary"):
        if cv_file and jd_text:
            with st.spinner("Running Ensemble Models (Semantic + Lexical + Fuzzy)..."):
                
                # 1. Extraction
                cv_text = TextCleaner.extract_text(cv_file)
                if len(cv_text) < 50:
                    st.error("Resume unreadable.")
                    return
                
                # 2. Engine
                @st.cache_resource
                def load_engine(): return EnsembleMatcher()
                
                engine = load_engine()
                res = engine.evaluate(cv_text, jd_text, sel_lang)
                
                # 3. GPT
                if use_gpt and api_key:
                    gpt = GPTAnalyst(api_key)
                    res.ai_analysis = gpt.analyze(cv_text, jd_text, res.final_score, res.missing_skills)

                # === DISPLAY ===
                st.divider()
                
                score = res.final_score * 100
                color = "#28a745" if score >= 70 else "#ffc107" if score >= 50 else "#dc3545"
                
                c_main, c_det = st.columns([1, 2])
                
                with c_main:
                    st.markdown(f"""
                    <div style="background:{color}20; padding:20px; border-radius:15px; text-align:center; border:2px solid {color}">
                        <h4 style="margin:0; color:{color}">MATCH SCORE</h4>
                        <h1 style="font-size:4em; margin:0; color:{color}">{score:.0f}%</h1>
                        <h3 style="margin:0; color:#555">{res.feedback}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c_det:
                    st.subheader("📊 Signal Breakdown")
                    m1, m2 = st.columns(2)
                    m1.metric("Semantic Fit", f"{res.semantic_score*100:.0f}%", 
                             help="How well you matched the Requirements paragraph (ignoring JD fluff).")
                    m2.metric("Skill Match", f"{res.keyword_score*100:.0f}%", 
                             help="How many Capitalized Skills (Java, Sales) you matched.")
                    
                    st.progress(res.final_score, text="Aggregate Confidence")

                st.divider()
                st.subheader("🔍 Skill Gap Analysis")
                
                t1, t2 = st.tabs(["✅ Matched Skills", "⚠️ Missing / Mismatched"])
                with t1:
                    if res.matched_skills:
                        st.markdown(" ".join([f"`{s}`" for s in res.matched_skills]))
                    else: st.info("No exact hard skills found (Relied purely on context).")
                with t2:
                    if res.missing_skills:
                        st.write("The JD specifically mentions these capitalized terms, but we didn't find them in your CV:")
                        st.markdown(" ".join([f"`{s}`" for s in res.missing_skills]))
                    else: st.success("Clean sweep! No obvious skills missing.")

                if res.ai_analysis:
                    st.divider()
                    st.subheader("🤖 GPT-4 Audit")
                    st.success(res.ai_analysis)

        else:
            st.warning("Upload files to start.")

if __name__ == "__main__":
    main()
