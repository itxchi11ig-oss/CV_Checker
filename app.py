import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
from thefuzz import fuzz # pip install thefuzz
import openai

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Final Boss ATS",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== ADVANCED ENGINE ====================

@dataclass
class MatchReport:
    final_score: float
    semantic_score: float
    keyword_coverage: float
    critical_matches: List[str]
    missing_critical: List[str]
    feedback: str
    ai_audit: Optional[str] = None

class DocumentProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        # Aggressive PDF cleanup (fixing glued words like "TeamworkLeadership")
        text = text.replace('\n', ' ')
        # Insert space between lowerUpper case changes (e.g., camelCase -> camel Case)
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        return text.strip()

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
            return DocumentProcessor.clean_text(text)
        except: return ""

    @staticmethod
    def extract_critical_terms(text: str) -> List[str]:
        """
        Extracts the 'DNA' of the Job Description.
        It looks for High-Frequency Nouns and Capitalized Phrases.
        """
        # 1. Protect Tech Stack
        text = text.replace("C++", "Cpp").replace(".NET", "DotNet").replace("Node.js", "Nodejs")
        
        # 2. Extract Capitalized Phrases (Proper Nouns)
        # Matches "Project Management", "Python", "AWS"
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s[A-Z][a-zA-Z0-9]*)*\b', text)
        
        # 3. Filter Garbage
        stopwords = {
            "The", "A", "An", "To", "For", "With", "In", "On", "At", "By", "We", "You", "And", "Or", 
            "Job", "Role", "Work", "Team", "Experience", "Skills", "Requirements", "Duties", 
            "Summary", "Description", "Company", "About", "Us", "Opportunity", "Equal", "Employer",
            "Year", "Years", "Degree", "Bachelor", "Master", "University", "Candidate", "Strong", "Good"
        }
        
        cleaned_nouns = [w for w in proper_nouns if len(w) > 2 and w not in stopwords]
        
        # 4. Count Frequency (If "Java" appears 5 times, it's critical. If "Friday" appears once, ignore it.)
        counts = Counter(cleaned_nouns)
        
        # Return top 25 most frequent proper nouns
        return [word for word, count in counts.most_common(25)]

class RecruiterSimulation:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    def calculate_coverage(self, cv_text: str, jd_terms: List[str]) -> Tuple[float, List[str], List[str]]:
        """
        Calculates 'Requirement Coverage' instead of 'Text Similarity'.
        """
        cv_lower = cv_text.lower()
        matched = []
        missing = []
        
        # Restore tech stack names for display
        display_terms = [t.replace("Cpp", "C++").replace("DotNet", ".NET").replace("Nodejs", "Node.js") for t in jd_terms]
        
        hits = 0
        for term in display_terms:
            # Check 1: Exact Match
            if term.lower() in cv_lower:
                hits += 1
                matched.append(term)
                continue
            
            # Check 2: Fuzzy Match (e.g. "ReactJS" vs "React.js")
            # Partial ratio allows matching "AWS" inside "AWS Certified"
            if fuzz.partial_ratio(term.lower(), cv_lower) >= 85:
                hits += 1
                matched.append(term)
            else:
                missing.append(term)
        
        # Coverage Score
        if not jd_terms: return 1.0, [], []
        coverage = hits / len(jd_terms)
        
        return coverage, matched, missing

    def analyze(self, cv_text: str, jd_text: str) -> MatchReport:
        # 1. Critical Term Extraction (The "Human Scan")
        jd_terms = DocumentProcessor.extract_critical_terms(jd_text)
        
        # 2. Coverage Calculation (The "Hard Skills" Check)
        coverage_raw, matched, missing = self.calculate_coverage(cv_text, jd_terms)
        
        # 3. Semantic Context (The "Vibe" Check)
        # We assume the CV is concise, so we extract the densest part of the JD
        cv_emb = self.model.encode(cv_text, convert_to_tensor=True)
        jd_emb = self.model.encode(jd_text, convert_to_tensor=True)
        sem_score_raw = float(util.cos_sim(cv_emb, jd_emb)[0][0])
        
        # 4. THE HIRED CURVE (Normalization)
        # A raw coverage of 40% (0.4) is usually enough to pass ATS.
        # We map 0.4 -> 0.85 (High Score)
        coverage_norm = min(1.0, coverage_raw * 2.2) 
        
        # Semantic scores usually hover at 0.35 for good matches.
        sem_score_norm = 1 / (1 + math.exp(-12 * (sem_score_raw - 0.25)))
        
        # 5. Final Weighting
        # Coverage is king (60%), Context is queen (40%)
        final_score = (coverage_norm * 0.60) + (sem_score_norm * 0.40)
        
        # Feedback
        if final_score >= 0.85: feedback = "Top Candidate (Interview Ready)"
        elif final_score >= 0.70: feedback = "Strong Match"
        elif final_score >= 0.50: feedback = "Potential Match"
        else: feedback = "Low Match"
        
        return MatchReport(final_score, sem_score_norm, coverage_norm, matched, missing, feedback)

# ==================== GPT AUDIT ====================

def run_gpt_audit(api_key, cv_text, jd_text, score, missing):
    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = f"""
        Role: Senior Recruiter.
        Algorithm Score: {score*100:.1f}%.
        Missing Keywords: {', '.join(missing[:8])}.
        
        JD Snippet: {jd_text[:800]}
        CV Snippet: {cv_text[:800]}
        
        Task:
        1. Validated Score: Give me your human estimated score (0-100%).
        2. Reality Check: Did the algorithm miss something implied?
        3. One quick win to fix the CV.
        """
        res = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except: return "GPT Unavailable"

# ==================== UI ====================

def main():
    st.markdown("""
        <style>
        .big-font {font-size:30px !important; font-weight: bold;}
        .score-box {padding: 20px; border-radius: 10px; text-align: center; color: white;}
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("⚙️ Engine Settings")
    use_gpt = st.sidebar.checkbox("Enable GPT-4 Audit")
    api_key = st.sidebar.text_input("OpenAI Key", type="password") if use_gpt else None

    st.title("🔥 Final Boss ATS")
    st.markdown("Uses **Critical Term Extraction** & **Recruiter Normalization Curves**. If you match the core hard skills, you get the score.")
    
    col1, col2 = st.columns(2)
    cv_file = col1.file_uploader("Candidate Resume", type=["pdf", "docx", "txt"])
    jd_text = col2.text_area("Job Description", height=250)

    if st.button("Run Simulation", type="primary"):
        if cv_file and jd_text:
            with st.spinner("Extracting Critical Terms & Calculating Coverage..."):
                # Extract
                cv_text = DocumentProcessor.extract_text(cv_file)
                if len(cv_text) < 50:
                    st.error("Resume unreadable.")
                    return
                
                # Analyze
                @st.cache_resource
                def load_engine(): return RecruiterSimulation()
                engine = load_engine()
                
                res = engine.analyze(cv_text, jd_text)
                
                # GPT
                if use_gpt and api_key:
                    res.ai_audit = run_gpt_audit(api_key, cv_text, jd_text, res.final_score, res.missing_critical)

                # === DISPLAY ===
                st.divider()
                
                score = res.final_score * 100
                color = "#2196F3" if score < 50 else "#ff9800" if score < 80 else "#4CAF50"
                
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown(f"""
                    <div class="score-box" style="background-color: {color};">
                        <h2 style="margin:0; color:white;">MATCH SCORE</h2>
                        <h1 style="font-size: 5em; margin:0; color:white;">{score:.0f}%</h1>
                        <h3 style="margin:0; color:white;">{res.feedback}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.subheader("📊 Why this score?")
                    m1, m2 = st.columns(2)
                    m1.metric("Requirement Coverage", f"{res.keyword_coverage*100:.0f}%", 
                             help="How many of the Top 25 Critical Keywords (Proper Nouns) you hit.")
                    m2.metric("Semantic Context", f"{res.semantic_score*100:.0f}%", 
                             help="Vector similarity of the overall text.")
                    
                    st.progress(res.final_score, text="Overall Probability")
                    
                    if score < 50:
                        st.warning("Score is low? Check 'Missing Skills' tab. You might be missing simple keywords.")

                st.divider()
                st.subheader("🔑 Critical Keyword Analysis")
                t1, t2 = st.tabs(["✅ Skills Found", "⚠️ Skills Missing"])
                
                with t1:
                    if res.critical_matches:
                        st.markdown(" ".join([f"`{s}`" for s in res.critical_matches]))
                    else: st.error("No critical hard skills found.")
                
                with t2:
                    if res.missing_critical:
                        st.write("The JD prioritizes these terms (by frequency/capitalization) but they are missing:")
                        st.error("  •  ".join(res.missing_critical))
                    else: st.success("Perfect coverage!")

                if res.ai_audit:
                    st.divider()
                    st.subheader("🤖 GPT-4 Auditor")
                    st.info(res.ai_audit)

        else:
            st.warning("Please provide files.")

if __name__ == "__main__":
    main()
