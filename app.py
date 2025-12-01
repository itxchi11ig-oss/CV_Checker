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
from thefuzz import process, fuzz  # REQUIRED: pip install thefuzz

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
    raw_ai_score: float
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
            
            # Cleaning artifacts
            return text.replace('\n', ' ').strip()
        except Exception:
            return ""

class CVEvaluator:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        
        # Stopwords (Generic + Recruitment specific)
        self.stopwords = set([
            "the", "and", "for", "that", "with", "from", "have", "will", "work", "team", 
            "skills", "experience", "years", "responsible", "duties", "required", 
            "preferred", "summary", "objective", "education", "qualifications",
            "und", "der", "die", "das", "mit", "für", "von", "erfahrung", "kenntnisse",
            "is", "a", "an", "or", "to", "in", "at", "be", "as", "on", "by", "it", "of",
            "knowledge", "ability", "demonstrated", "strong", "excellent"
        ])

    def extract_entities(self, text: str) -> Set[str]:
        """
        Smart Extraction: Focuses on Capitalized Words (Skills) and Technical terms
        """
        # 1. Clean basic punctuation but keep C++, C#, .NET, Node.js chars
        clean_text = re.sub(r'[^a-zA-Z0-9\+\#\.\s\-]', ' ', text)
        
        words = clean_text.split()
        entities = set()
        
        for i, word in enumerate(words):
            # Logic: If word is Capitalized (Java) or has tech chars (C++), it's likely a skill
            # We ignore common stopwords
            lower_word = word.lower()
            
            if lower_word in self.stopwords or len(word) < 2:
                continue
                
            # If it looks like a skill (Capitalized or contains digit/symbol)
            if word[0].isupper() or any(c in word for c in "+#."):
                entities.add(lower_word)
            
            # Grab Bigrams for things like "Project Management"
            if i < len(words) - 1:
                next_word = words[i+1]
                if next_word[0].isupper():
                    entities.add(f"{lower_word} {next_word.lower()}")

        return entities

    def fuzzy_match_score(self, cv_entities: Set[str], jd_entities: Set[str]) -> Tuple[float, List[str], List[str]]:
        """
        Uses Levenshtein Distance to find matches even if spelling differs.
        e.g., "ReactJS" in CV matches "React.js" in JD
        """
        if not jd_entities:
            return 0.0, [], []

        matches = []
        missing = []
        
        # Check every required JD skill against CV skills using Fuzzy Logic
        hits = 0
        for req in jd_entities:
            # If we find a match > 85% similarity, we count it
            best_match = process.extractOne(req, cv_entities, scorer=fuzz.token_sort_ratio)
            
            if best_match and best_match[1] >= 80: # 80% similarity threshold
                hits += 1
                matches.append(req) # We list the JD requirement as matched
            else:
                missing.append(req)
        
        score = hits / len(jd_entities)
        
        # Curve: If you have 50% of the keywords, that is usually enough for an interview
        # We boost the score. 50% raw match -> 100% score.
        final_score = min(1.0, score * 2.0)
        
        return final_score, matches, missing

    def semantic_curve(self, raw_score: float) -> float:
        """
        THE 'HIRED' CALIBRATION:
        Real CVs usually score 0.35 - 0.55 in raw cosine similarity.
        We map 0.25 -> 60% (Passable) and 0.50 -> 95% (Excellent).
        """
        if raw_score < 0.1: return 0.0
        
        # Linear interpolation between 0.15 (30%) and 0.55 (100%)
        min_benchmark = 0.15
        max_benchmark = 0.55
        
        normalized = (raw_score - min_benchmark) / (max_benchmark - min_benchmark)
        return max(0.0, min(1.0, normalized))

    def evaluate(self, cv_text: str, job_desc: str, lang_code: str) -> EvaluationResult:
        # 1. Semantic Score (The 'Vibe' Check)
        emb1 = self.model.encode(cv_text, convert_to_tensor=True)
        emb2 = self.model.encode(job_desc, convert_to_tensor=True)
        raw_ai_score = float(util.cos_sim(emb1, emb2)[0][0])
        
        human_semantic_score = self.semantic_curve(raw_ai_score)
        
        # 2. Fuzzy Keyword Match (The 'Hard Skills' Check)
        cv_entities = self.extract_entities(cv_text)
        jd_entities = self.extract_entities(job_desc)
        
        keyword_score, matched, missing = self.fuzzy_match_score(cv_entities, jd_entities)
        
        # 3. Final Weighted Score
        # If Semantic is high, we trust the CV more even if keywords are missing
        final_score = (human_semantic_score * 0.65) + (keyword_score * 0.35)
        
        # Sort output for display
        matched.sort()
        missing = [m for m in missing if len(m) > 3] # Filter noise
        missing.sort()

        return EvaluationResult(
            overall_score=final_score,
            raw_ai_score=raw_ai_score,
            keyword_score=keyword_score,
            detailed_feedback=self.generate_feedback(final_score, lang_code),
            matched_keywords=matched[:15],
            missing_keywords=missing[:15],
            ai_suggestions=None
        )

    def generate_feedback(self, score: float, lang: str) -> Dict[str, str]:
        texts = {
            "en": ["Not a Match", "Potential Match", "Strong Match", "Top Candidate"],
            "de": ["Kein Treffer", "Potenzieller Kandidat", "Starker Kandidat", "Top Kandidat"],
            "fr": ["Pas de correspondance", "Correspondance potentielle", "Forte correspondance", "Candidat idéal"],
            "es": ["No coincide", "Coincidencia potencial", "Coincidencia fuerte", "Candidato ideal"],
            "it": ["Nessuna corrispondenza", "Corrispondenza potenziale", "Forte corrispondenza", "Candidato ideale"]
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
        lang_map = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian"}
        lang_name = lang_map.get(self.language_code, "English")
        
        prompt = f"""
        You are a hiring manager. I have a candidate who scored {result.overall_score*100:.0f}% match.
        
        MISSING SKILLS (Fuzzy Match Failed): {', '.join(result.missing_keywords[:10])}
        
        JOB REQ: {job_desc[:1000]}
        CV TEXT: {cv_text[:1000]}
        
        Give me 3 bullet points. 
        1. If the score is good, say why.
        2. If specific hard skills are missing in the CV but present in the JD, list them clearly.
        3. Suggest one phrasing change.
        Language: {lang_name}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {str(e)}"

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
    st.set_page_config(page_title="Pro ATS Matcher", page_icon="👔", layout="wide")
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    
    st.sidebar.title("⚙️ Setup")
    lang_map = {"en": "English", "de": "Deutsch", "fr": "Français", "es": "Español", "it": "Italiano"}
    sel_lang = st.sidebar.selectbox("Language", options=list(lang_map.keys()), format_func=lambda x: lang_map[x])
    st.session_state.lang = sel_lang
    
    use_gpt = st.sidebar.checkbox("Enable GPT-4 Suggestions")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password") if use_gpt else None
    
    st.title(f"👔 Pro ATS Matcher ({lang_map[st.session_state.lang]})")
    st.markdown("Uses **Fuzzy Logic** & **Calibrated Vectors**. A score > **70%** indicates an Interview-Ready candidate.")
    
    c1, c2 = st.columns(2)
    cv = c1.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd = c2.text_area("Paste Job Description", height=200)
    
    if st.button("Analyze Match", type="primary"):
        if cv and jd:
            with st.spinner("Analyzing..."):
                ctrl = Controller()
                res = ctrl.process(cv, jd, st.session_state.lang, api_key, use_gpt)
                
                if res:
                    score = res.overall_score * 100
                    color = "#d9534f" if score < 60 else "#f0ad4e" if score < 80 else "#5cb85c"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 25px; background-color: #f9f9f9; border-radius: 12px; margin-bottom: 25px;">
                        <h3 style="margin:0; color: #555;">ATS Probability</h3>
                        <h1 style="margin:10px 0; font-size: 4em; color: {color};">{score:.0f}%</h1>
                        <p style="margin:0; font-weight:bold; font-size: 1.5em; color: #333;">{res.detailed_feedback['overall']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("🔑 Skill Gap Analysis (Fuzzy Match)")
                    t1, t2 = st.tabs(["✅ Skills Found", "⚠️ Skills Missing"])
                    
                    with t1:
                        if res.matched_keywords:
                            st.write("Found in CV:")
                            st.markdown(" ".join([f"`{k}`" for k in res.matched_keywords]), unsafe_allow_html=True)
                        else: st.warning("No specific hard skills matched.")
                            
                    with t2:
                        if res.missing_keywords:
                            st.write("JD requires these, but not found in CV:")
                            st.markdown(" ".join([f"`{k}`" for k in res.missing_keywords]), unsafe_allow_html=True)
                        else: st.success("No missing skills!")
                    
                    if res.ai_suggestions:
                        st.divider()
                        st.subheader("🤖 AI Recruiter Feedback")
                        st.info(res.ai_suggestions)
                        
                    with st.expander("Debug Info (Math)"):
                        st.write(f"Raw Vector Similarity: {res.raw_ai_score:.3f} (Note: >0.35 is good)")
                        st.write(f"Keyword Hit Rate: {res.keyword_score*100:.1f}%")
                else: st.error("Could not read text.")
        else: st.warning("Upload both files.")

if __name__ == "__main__":
    main()
