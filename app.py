import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import openai
from enum import Enum
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from thefuzz import process, fuzz  # pip install thefuzz

# ==================== CONFIGURATION ====================

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

# ==================== LOGIC LAYER ====================

@dataclass
class EvaluationResult:
    overall_score: float
    semantic_score: float
    keyword_score: float
    detailed_feedback: Dict[str, str]
    matched_keywords: List[str]
    missing_keywords: List[str]
    ai_suggestions: Optional[str] = None

class CVParser:
    @staticmethod
    def extract_text(file) -> str:
        try:
            file_type = file.name.split('.')[-1].lower()
            text = ""
            if file_type == 'pdf':
                reader = PyPDF2.PdfReader(file)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
            elif file_type in ['docx', 'doc']:
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif file_type == 'txt':
                text = file.getvalue().decode('utf-8')
            return text
        except Exception:
            return ""

class CVEvaluator:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        
        # Aggressive Stopword List to remove JD Fluff
        self.stopwords = set([
            "the", "and", "for", "that", "with", "from", "have", "will", "work", "team", 
            "skills", "experience", "years", "responsible", "duties", "required", 
            "preferred", "summary", "objective", "education", "qualifications",
            "opportunity", "employer", "equal", "status", "gender", "race", "color",
            "religion", "sexual", "orientation", "identity", "expression", "veteran",
            "disability", "accommodation", "apply", "click", "link", "website",
            "und", "der", "die", "das", "mit", "für", "von", "erfahrung", "kenntnisse",
            "is", "a", "an", "or", "to", "in", "at", "be", "as", "on", "by", "it", "of"
        ])

    def clean_tokens(self, text: str) -> Set[str]:
        """Turns text into a set of significant words (lowercase, no punctuation)"""
        # Keep C++, C#, .NET intact, remove other symbols
        text = re.sub(r'[^a-zA-Z0-9\+\#\.]', ' ', text.lower())
        words = text.split()
        return set([w for w in words if len(w) > 2 and w not in self.stopwords])

    def calculate_segmented_similarity(self, cv_text: str, jd_text: str) -> float:
        """
        Smart Logic: Splits JD into chunks (Requirements, About Us, etc.)
        Compares CV against each chunk and takes the HIGHEST score.
        This prevents 'About Us' fluff from tanking the score.
        """
        # 1. Encode CV
        cv_emb = self.model.encode(cv_text, convert_to_tensor=True)
        
        # 2. Split JD into chunks of ~150 words (likely paragraphs)
        jd_words = jd_text.split()
        chunk_size = 150
        chunks = [" ".join(jd_words[i:i + chunk_size]) for i in range(0, len(jd_words), 100)] # 100 overlap
        
        if not chunks: 
            return 0.0
            
        # 3. Encode all JD chunks
        jd_embs = self.model.encode(chunks, convert_to_tensor=True)
        
        # 4. Compute similarity of CV vs ALL chunks
        similarities = util.cos_sim(cv_emb, jd_embs)[0]
        
        # 5. Take the Max Score (The CV matched the "Requirements" chunk well, ignore the rest)
        max_score = float(similarities.max())
        
        return max_score

    def jaccard_similarity(self, cv_tokens: Set[str], jd_tokens: Set[str]) -> float:
        """Classic 'Bag of Words' overlap. Acts as a safety net."""
        if not jd_tokens: return 0.0
        intersection = cv_tokens.intersection(jd_tokens)
        return len(intersection) / len(jd_tokens)

    def normalize_score(self, raw_score: float) -> float:
        """
        Generous Curve for Resume Matching.
        Raw 0.25 (typical for resume/JD) -> 65%
        Raw 0.40 -> 85%
        Raw 0.60 -> 100%
        """
        if raw_score <= 0.1: return raw_score * 3 # Penalty box
        
        # Log-like curve boost
        # Maps 0.15 -> 0.50
        # Maps 0.35 -> 0.80
        boosted = 1 / (1 + math.exp(-12 * (raw_score - 0.28)))
        return boosted

    def evaluate(self, cv_text: str, job_desc: str, lang_code: str) -> EvaluationResult:
        # 1. Segmented Semantic Score (AI)
        raw_ai_score = self.calculate_segmented_similarity(cv_text, job_desc)
        sem_score = self.normalize_score(raw_ai_score)
        
        # 2. Keyword/Jaccard Score (Exact Match)
        cv_tokens = self.clean_tokens(cv_text)
        jd_tokens = self.clean_tokens(job_desc)
        
        # Find explicit keywords using Fuzzy Matching on significant tokens
        # We assume Capitalized words in JD are important (e.g. Java, Sales)
        jd_important_keywords = [w for w in job_desc.split() if w[0].isupper() and len(w) > 3]
        jd_important_set = set([re.sub(r'[^a-zA-Z0-9]', '', w).lower() for w in jd_important_keywords])
        
        matched = []
        missing = []
        
        if jd_important_set:
            # Check overlap on important words only
            for req in jd_important_set:
                if req in cv_tokens:
                    matched.append(req)
                else:
                    # Try fuzzy fallback
                    best = process.extractOne(req, cv_tokens, scorer=fuzz.ratio)
                    if best and best[1] > 85:
                        matched.append(req)
                    else:
                        missing.append(req)
            
            # Keyword score based on important words
            kw_score = len(matched) / len(jd_important_set) if jd_important_set else 0
            # Curve it: 50% keyword match is usually great
            kw_score = min(1.0, kw_score * 1.8)
        else:
            # Fallback to general Jaccard if no proper nouns found
            kw_score = min(1.0, self.jaccard_similarity(cv_tokens, jd_tokens) * 3.0)
            matched = list(cv_tokens.intersection(jd_tokens))[:10]

        # 3. Final Score: 60% Semantic, 40% Keyword
        # However, if Semantic is very high (>85%), we ignore low keyword score (implies synonyms used)
        if sem_score > 0.85:
            final_score = sem_score
        else:
            final_score = (sem_score * 0.6) + (kw_score * 0.4)

        return EvaluationResult(
            overall_score=final_score,
            semantic_score=sem_score,
            keyword_score=kw_score,
            detailed_feedback=self.generate_feedback(final_score, lang_code),
            matched_keywords=list(set(matched))[:15],
            missing_keywords=list(set(missing))[:15],
            ai_suggestions=None
        )

    def generate_feedback(self, score: float, lang: str) -> Dict[str, str]:
        texts = {
            "en": ["Mismatch", "Potential", "Strong Candidate", "Perfect Match"],
            "de": ["Passt nicht", "Potenzial", "Starker Kandidat", "Perfekt"],
            "fr": ["Mauvais", "Potentiel", "Fort", "Parfait"],
            "es": ["No apto", "Potencial", "Fuerte", "Perfecto"],
            "it": ["No", "Potenziale", "Forte", "Perfetto"]
        }
        labels = texts.get(lang, texts["en"])
        
        if score < 0.5: idx = 0
        elif score < 0.7: idx = 1
        elif score < 0.85: idx = 2
        else: idx = 3
        
        return {"overall": labels[idx]}

class GPTEvaluator:
    def __init__(self, api_key: str, language_code: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.language_code = language_code
    
    def get_detailed_suggestions(self, cv_text: str, job_desc: str, result: EvaluationResult) -> str:
        prompt = f"""
        You are a hiring manager. Analyze this Resume vs JD.
        SCORE: {result.overall_score*100:.1f}%
        
        MISSING KEYWORDS (detected): {', '.join(result.missing_keywords[:8])}
        
        RESUME: {cv_text[:1000]}...
        JD: {job_desc[:1000]}...
        
        The candidate scored {result.overall_score*100:.0f}%.
        1. Is this actually a good match? (Be honest, ignore the score if the content looks good).
        2. List 3 critical missing Hard Skills.
        3. Suggest a Summary to add to the top of the CV.
        Output Language: {self.language_code}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"GPT Error: {str(e)}"

# ==================== CONTROLLER & UI ====================

class Controller:
    def __init__(self):
        self.evaluator = self.load_model()
        self.gpt = None

    @st.cache_resource
    def load_model(_self):
        return CVEvaluator()

    def process(self, cv, jd, lang, api_key, use_gpt):
        lang_code = get_safe_lang_code(lang)
        cv_text = CVParser.extract_text(cv)
        if len(cv_text) < 50: return None
        result = self.evaluator.evaluate(cv_text, jd, lang_code)
        
        if use_gpt and api_key:
            if not self.gpt or self.gpt.language_code != lang_code:
                self.gpt = GPTEvaluator(api_key, lang_code)
            result.ai_suggestions = self.gpt.get_detailed_suggestions(cv_text, jd, result)
        return result

def main():
    st.set_page_config(page_title="Resume Matcher Ultra", page_icon="🚀", layout="wide")
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    
    st.sidebar.title("⚙️ Settings")
    lang_map = {"en": "English", "de": "Deutsch", "fr": "Français", "es": "Español", "it": "Italiano"}
    sel_lang = st.sidebar.selectbox("Language", options=list(lang_map.keys()), format_func=lambda x: lang_map[x])
    st.session_state.lang = sel_lang
    
    use_gpt = st.sidebar.checkbox("Enable GPT-4 Analysis")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password") if use_gpt else None
    
    st.title(f"🚀 Resume Matcher Ultra")
    st.markdown("Optimized for **Resumes** (Short/Dense) vs **Job Descriptions** (Long/Verbose). Uses Segmented Analysis.")
    
    c1, c2 = st.columns(2)
    cv = c1.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd = c2.text_area("Paste Job Description", height=250)
    
    if st.button("Analyze Match", type="primary"):
        if cv and jd:
            with st.spinner("Analyzing..."):
                ctrl = Controller()
                res = ctrl.process(cv, jd, st.session_state.lang, api_key, use_gpt)
                
                if res:
                    score = res.overall_score * 100
                    # Color Logic
                    color = "#d9534f" if score < 50 else "#f0ad4e" if score < 75 else "#5cb85c"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: #f1f3f6; border-radius: 15px; margin-bottom: 20px;">
                        <h2 style="color: #666; margin:0;">Hiring Probability</h2>
                        <h1 style="font-size: 4em; color: {color}; margin: 0;">{score:.0f}%</h1>
                        <h3 style="margin:0;">{res.detailed_feedback['overall']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Breakdown
                    c1, c2 = st.columns(2)
                    c1.metric("Context Match (AI)", f"{res.semantic_score*100:.0f}%", help="Does the resume 'sound' like the job?")
                    c2.metric("Skill Overlap", f"{res.keyword_score*100:.0f}%", help="Do you have the specific capitalized skills?")
                    
                    st.subheader("🔑 Key Skills Analysis")
                    t1, t2 = st.tabs(["✅ Matched", "⚠️ Missing"])
                    with t1:
                        if res.matched_keywords:
                            st.write(", ".join([f"**{k}**" for k in res.matched_keywords]))
                        else: st.warning("No exact keyword matches found.")
                    with t2:
                        if res.missing_keywords:
                            st.write("JD mentions these (Capitalized), but Resume missing them:")
                            st.error(", ".join(res.missing_keywords))
                        else: st.success("No obvious missing skills!")
                        
                    if res.ai_suggestions:
                        st.divider()
                        st.subheader("🤖 Recruiter Notes")
                        st.info(res.ai_suggestions)
                else: st.error("Error reading CV text.")
        else: st.warning("Upload files first.")

if __name__ == "__main__":
    main()
