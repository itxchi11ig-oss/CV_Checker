import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import openai
from enum import Enum
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Counter
from thefuzz import process, fuzz  # pip install thefuzz

# ==================== CONFIGURATION & CONSTANTS ====================

class Language(Enum):
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"

STOPWORDS = set([
    "the", "and", "for", "that", "with", "from", "have", "will", "work", "team", 
    "skills", "experience", "years", "responsible", "duties", "required", 
    "preferred", "summary", "objective", "education", "qualifications",
    "opportunity", "employer", "equal", "status", "gender", "race", "color",
    "religion", "sexual", "orientation", "identity", "expression", "veteran",
    "disability", "accommodation", "apply", "click", "link", "website",
    "und", "der", "die", "das", "mit", "für", "von", "erfahrung", "kenntnisse",
    "is", "a", "an", "or", "to", "in", "at", "be", "as", "on", "by", "it", "of",
    "about", "us", "we", "are", "looking", "seeking", "candidate", "role"
])

# ==================== ADVANCED PARSING LAYER ====================

class TextProcessor:
    @staticmethod
    def extract_text(file) -> str:
        """Extracts text with layout preservation attempts"""
        try:
            file_type = file.name.split('.')[-1].lower()
            text = ""
            if file_type == 'pdf':
                reader = PyPDF2.PdfReader(file)
                # Join with newlines to preserve paragraph structure
                text = " \n ".join([page.extract_text() or "" for page in reader.pages])
            elif file_type in ['docx', 'doc']:
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif file_type == 'txt':
                text = file.getvalue().decode('utf-8')
            return text
        except Exception:
            return ""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Enterprise Cleaning:
        1. Lowers case.
        2. Preserves tech stack special chars (C++, C#, .NET).
        3. Normalizes whitespace.
        """
        text = text.lower()
        # Protect C++, C#, .NET, Node.js before regex cleaning
        text = text.replace("c++", "cpp_placeholder")
        text = text.replace("c#", "csharp_placeholder")
        text = text.replace(".net", "dotnet_placeholder")
        text = text.replace("node.js", "nodejs_placeholder")
        
        # Remove non-alphanumeric except underscores (placeholders)
        text = re.sub(r'[^a-z0-9_\s]', ' ', text)
        
        # Restore placeholders
        text = text.replace("cpp_placeholder", "c++")
        text = text.replace("csharp_placeholder", "c#")
        text = text.replace("dotnet_placeholder", ".net")
        text = text.replace("nodejs_placeholder", "node.js")
        
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def extract_key_terms(text: str, top_n: int = 25) -> List[Tuple[str, int]]:
        """
        Extracts the 'DNA' of the text: The most frequent non-stopword terms.
        This ignores 'Company Culture' fluff and focuses on repeated hard skills.
        """
        clean = TextProcessor.clean_text(text)
        words = [w for w in clean.split() if w not in STOPWORDS and len(w) > 2]
        # Count frequencies
        counter = Counter(words)
        return counter.most_common(top_n)

# ==================== HYBRID SCORING ENGINE ====================

@dataclass
class MatchResult:
    final_score: float
    semantic_score: float
    lexical_score: float
    matched_terms: List[str]
    missing_terms: List[str]
    feedback: str
    suggestions: str = ""

class HybridMatcher:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def calculate_semantic_score(self, cv_text: str, jd_text: str) -> float:
        """
        Vector Space Matching (The 'Vibe' Check).
        Good for: Concepts (Managing vs Leadership).
        """
        # We only encode the first 4000 chars to stay within model limits and focus on core content
        emb1 = self.model.encode(cv_text[:4000], convert_to_tensor=True)
        emb2 = self.model.encode(jd_text[:4000], convert_to_tensor=True)
        raw_score = float(util.cos_sim(emb1, emb2)[0][0])
        
        # Enterprise Normalization Curve (Sigmoid)
        # Centers the AI's raw 0.30 score to a human 60%
        return 1 / (1 + math.exp(-10 * (raw_score - 0.25)))

    def calculate_lexical_score(self, cv_text: str, jd_text: str) -> Tuple[float, List[str], List[str]]:
        """
        Keyword Frequency Matching (The 'ATS' Check).
        Good for: Exact Hard Skills (Python, SQL).
        """
        # Get the "DNA" of the JD (Top 30 most important words)
        jd_terms = [t[0] for t in TextProcessor.extract_key_terms(jd_text, 30)]
        cv_clean = TextProcessor.clean_text(cv_text)
        cv_tokens = set(cv_clean.split())
        
        matched = []
        missing = []
        hits = 0
        
        for term in jd_terms:
            # Exact Match
            if term in cv_tokens:
                hits += 1.0
                matched.append(term)
            else:
                # Fuzzy Fallback (e.g. "Github" vs "Git")
                # We use a high threshold (90) to avoid false positives
                match = process.extractOne(term, cv_tokens, scorer=fuzz.ratio)
                if match and match[1] >= 90:
                    hits += 1.0
                    matched.append(term)
                else:
                    missing.append(term)

        # Score is purely based on how many of the JD's TOP words appear in the CV
        if not jd_terms: return 0.0, [], []
        
        raw_score = hits / len(jd_terms)
        
        # Boost: If you match 60% of the Top 30 words, you are a 100% match lexically
        normalized_score = min(1.0, raw_score * 1.6)
        
        return normalized_score, matched, missing

    def evaluate(self, cv_text: str, jd_text: str) -> MatchResult:
        # 1. Semantic (AI Meaning)
        sem_score = self.calculate_semantic_score(cv_text, jd_text)
        
        # 2. Lexical (Exact Keywords)
        lex_score, matched, missing = self.calculate_lexical_score(cv_text, jd_text)
        
        # 3. Hybrid Weighting
        # We trust Semantic more (65%) because it captures context, but Lexical (35%) ensures hard skills
        final_score = (sem_score * 0.65) + (lex_score * 0.35)
        
        # Feedback Logic
        feedback = "Potential Match"
        if final_score > 0.85: feedback = "Top Tier Candidate"
        elif final_score > 0.70: feedback = "Strong Match"
        elif final_score > 0.50: feedback = "Moderate Match"
        else: feedback = "Low Match"

        return MatchResult(
            final_score=final_score,
            semantic_score=sem_score,
            lexical_score=lex_score,
            matched_terms=matched,
            missing_terms=missing,
            feedback=feedback
        )

# ==================== GPT LAYER ====================

class GPTAdvisor:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def analyze(self, cv_text: str, jd_text: str, match_data: MatchResult) -> str:
        prompt = f"""
        Act as a Senior Recruiter at Google. Analyze this match.
        
        MATCH SCORE: {match_data.final_score*100:.1f}%
        MISSING KEYWORDS: {', '.join(match_data.missing_terms[:8])}
        
        JOB (Summary): {jd_text[:1000]}
        RESUME (Summary): {cv_text[:1000]}
        
        Provide:
        1. Honest verdict (Hired/Rejected?).
        2. Three specific missing skills to add.
        3. One bullet point to rewrite for impact.
        """
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            return res.choices[0].message.content
        except:
            return "GPT API Error."

# ==================== UI CONTROLLER ====================

def main():
    st.set_page_config(page_title="Enterprise CV Matcher", page_icon="🏢", layout="wide")
    
    # Custom CSS for "Google-like" clean look
    st.markdown("""
        <style>
        .metric-card {background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 10px; text-align: center;}
        .score-high {color: #28a745; font-size: 3em; font-weight: bold;}
        .score-med {color: #ffc107; font-size: 3em; font-weight: bold;}
        .score-low {color: #dc3545; font-size: 3em; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("⚙️ System Config")
    use_gpt = st.sidebar.checkbox("Enable GPT-4 Audit")
    api_key = st.sidebar.text_input("OpenAI Key", type="password") if use_gpt else None

    st.title("🏢 Enterprise CV Matcher")
    st.markdown("Uses **Hybrid Search (Vector + Lexical)** and **Term Frequency Analysis** to ignore JD fluff.")

    col1, col2 = st.columns(2)
    cv_file = col1.file_uploader("Candidate Resume", type=["pdf", "docx", "txt"])
    jd_text = col2.text_area("Job Description", height=300, help="Paste the full job description here.")

    if st.button("Run Analysis", type="primary"):
        if cv_file and jd_text:
            with st.spinner("Initializing Hybrid Engine..."):
                # 1. Extraction
                cv_text = TextProcessor.extract_text(cv_file)
                if len(cv_text) < 50:
                    st.error("Resume file appears empty or unreadable.")
                    return

                # 2. Evaluation
                engine = HybridMatcher()
                result = engine.evaluate(cv_text, jd_text)
                
                # 3. AI Audit
                if use_gpt and api_key:
                    gpt = GPTAdvisor(api_key)
                    result.suggestions = gpt.analyze(cv_text, jd_text, result)

                # ==================== DASHBOARD ====================
                st.divider()
                
                # Top Level Score
                score = result.final_score * 100
                score_class = "score-high" if score > 70 else "score-med" if score > 50 else "score-low"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin:0">Match Confidence</h2>
                    <div class="{score_class}">{score:.1f}%</div>
                    <h3 style="margin:0; color: #6c757d;">{result.feedback}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Explanation
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"**Semantic Score: {result.semantic_score*100:.1f}%**\n\nHow well the resume's 'vibe' and context matches the job (using Vector Embeddings).")
                with c2:
                    st.info(f"**Lexical Score: {result.lexical_score*100:.1f}%**\n\nHow many of the JD's top 30 frequency words appear in the resume.")

                # Gap Analysis
                st.subheader("🔍 Deep Dive")
                tab1, tab2 = st.tabs(["✅ Matched Core Terms", "⚠️ Missing Core Terms"])
                
                with tab1:
                    if result.matched_terms:
                        st.write("The candidate explicitly mentions these high-value JD terms:")
                        st.markdown(" ".join([f"`{t}`" for t in result.matched_terms]), unsafe_allow_html=True)
                    else:
                        st.warning("No high-value keywords found.")
                        
                with tab2:
                    if result.missing_terms:
                        st.write("The JD uses these terms frequently, but they are absent in the Resume:")
                        st.markdown(" ".join([f"`{t}`" for t in result.missing_terms]), unsafe_allow_html=True)
                    else:
                        st.success("Candidate covers all core terminology.")

                # GPT Output
                if result.suggestions:
                    st.subheader("🤖 Recruiter Audit (GPT-4)")
                    st.success(result.suggestions)

if __name__ == "__main__":
    main()
