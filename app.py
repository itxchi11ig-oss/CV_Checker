import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import openai
from enum import Enum
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==================== CONFIGURATION ====================

class Language(Enum):
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"

def get_safe_lang_code(lang) -> str:
    """Helper to ensure we always get a simple string 'en', 'de', etc."""
    if isinstance(lang, Language):
        return lang.value
    if isinstance(lang, str):
        return lang
    try:
        return lang.value
    except AttributeError:
        return "en"

# ==================== LOGIC LAYER ====================

@dataclass
class EvaluationResult:
    overall_score: float
    raw_ai_score: float  # Added for transparency
    keyword_score: float
    detailed_feedback: Dict[str, str]
    matched_skills: List[str]
    missing_skills: List[str]
    ai_suggestions: Optional[str] = None

class CVParser:
    @staticmethod
    def extract_text(file) -> str:
        try:
            file_type = file.name.split('.')[-1].lower()
            if file_type == 'pdf':
                reader = PyPDF2.PdfReader(file)
                return " ".join([page.extract_text() or "" for page in reader.pages])
            elif file_type in ['docx', 'doc']:
                doc = docx.Document(file)
                return "\n".join([p.text for p in doc.paragraphs])
            elif file_type == 'txt':
                return file.getvalue().decode('utf-8')
            return ""
        except Exception as e:
            return ""

class GPTEvaluator:
    def __init__(self, api_key: str, language_code: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.language_code = language_code
    
    def get_detailed_suggestions(self, cv_text: str, job_desc: str, result: EvaluationResult) -> str:
        lang_map = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian"}
        lang_name = lang_map.get(self.language_code, "English")
        
        prompt = f"""
        Act as a senior technical recruiter. Review this CV against the Job Description.
        
        MATCH SCORE: {result.overall_score*100:.1f}%
        
        JOB DESCRIPTION:
        {job_desc[:1500]}
        
        CV SUMMARY:
        {cv_text[:1500]}
        
        Provide 3 specific, brutal, and actionable changes to improve the match score. 
        Focus on hard skills missing from the CV that are present in the JD.
        Output language: {lang_name}.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {str(e)}"

class CVEvaluator:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        # Extensive stopword lists to clean noise
        self.stopwords = set("""
        the and for that with from have will work team skills experience years 
        responsible duties required preferred summary objective
        und der die das mit für von dass wir erfahrung kenntnisse jahre aufgaben
        pour avec dans les des une est sur expérience
        para con las los una que por experiencia
        per con del della che una sono esperienza
        is a an or to in at be as on by it
        """.split())

    def clean_text(self, text: str) -> List[str]:
        """Cleans text to extracting meaningful unique keywords"""
        # Remove special chars, lower case
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        # Filter stopwords and short words
        return [w for w in words if len(w) > 3 and w not in self.stopwords and not w.isdigit()]

    def calculate_similarity(self, cv_text: str, job_desc: str) -> float:
        """Calculates raw cosine similarity (Math Layer)"""
        emb1 = self.model.encode(cv_text, convert_to_tensor=True)
        emb2 = self.model.encode(job_desc, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0])

    def sigmoid_normalize(self, raw_score: float) -> float:
        """
        THE MAGIC SAUCE: Converts Robot Math to Human Grades.
        Raw scores usually float between 0.2 (bad) and 0.6 (perfect).
        We map this to a 0% - 100% scale using a Logistic Curve.
        """
        # Center point (x0): 0.35 (This counts as a 50% match)
        # Steepness (k): 10 (How fast it rises)
        x0 = 0.30
        k = 12
        
        # Logistic Function
        human_score = 1 / (1 + math.exp(-k * (raw_score - x0)))
        return human_score

    def evaluate(self, cv_text: str, job_desc: str, lang_code: str) -> EvaluationResult:
        # 1. Semantic Match (The Meaning)
        raw_ai_score = self.calculate_similarity(cv_text, job_desc)
        human_semantic_score = self.sigmoid_normalize(raw_ai_score)
        
        # 2. Keyword Match (The Buzzwords)
        cv_words = set(self.clean_text(cv_text))
        jd_words = set(self.clean_text(job_desc))
        
        if not jd_words:
            keyword_score = 0.0
            matched = []
            missing = []
        else:
            matched = list(cv_words.intersection(jd_words))
            missing = list(jd_words - cv_words)
            # Keyword score: heavily boosted because JDs have junk words too
            keyword_score = min(1.0, (len(matched) / len(jd_words)) * 2.5)

        # 3. Final Weighted Score (70% Semantic, 30% Keywords)
        final_score = (human_semantic_score * 0.7) + (keyword_score * 0.3)
        
        return EvaluationResult(
            overall_score=final_score,
            raw_ai_score=raw_ai_score,
            keyword_score=keyword_score,
            detailed_feedback=self.generate_feedback(final_score, lang_code),
            matched_skills=sorted(matched)[:10],
            missing_skills=sorted(missing)[:10]
        )

    def generate_feedback(self, score: float, lang: str) -> Dict[str, str]:
        # Simple feedback logic based on the Human Score
        texts = {
            "en": ["Low Match", "Moderate Match", "Good Match", "Excellent Match"],
            "de": ["Geringe Übereinstimmung", "Mäßige Übereinstimmung", "Gute Übereinstimmung", "Exzellent"],
            "fr": ["Faible", "Modérée", "Bonne", "Excellente"],
            "es": ["Baja", "Moderada", "Buena", "Excelente"],
            "it": ["Bassa", "Moderata", "Buona", "Eccellente"]
        }
        labels = texts.get(lang, texts["en"])
        
        if score < 0.4: idx = 0
        elif score < 0.6: idx = 1
        elif score < 0.8: idx = 2
        else: idx = 3
        
        return {"overall": labels[idx]}

# ==================== VIEW LAYER ====================

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
        
        if len(cv_text) < 50:
            return None
            
        result = self.evaluator.evaluate(cv_text, jd, lang_code)
        
        if use_gpt and api_key:
            if not self.gpt or self.gpt.language_code != lang_code:
                self.gpt = GPTEvaluator(api_key, lang_code)
            result.ai_suggestions = self.gpt.get_detailed_suggestions(cv_text, jd, result)
            
        return result

def main():
    st.set_page_config(page_title="AI CV Matcher Pro", page_icon="🚀", layout="wide")
    
    # Session State Init
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    
    # Sidebar
    st.sidebar.title("⚙️ Config")
    lang_map = {"en": "English", "de": "Deutsch", "fr": "Français", "es": "Español", "it": "Italiano"}
    sel_lang = st.sidebar.selectbox("Language", options=list(lang_map.keys()), format_func=lambda x: lang_map[x])
    st.session_state.lang = sel_lang
    
    use_gpt = st.sidebar.checkbox("Enable GPT-4 Suggestions")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password") if use_gpt else None
    
    # Main UI
    st.title(f"🚀 AI CV Matcher ({lang_map[st.session_state.lang]})")
    st.markdown("This tool uses **Sigmoid Normalization** to convert raw AI vectors into realistic human match scores.")
    
    c1, c2 = st.columns(2)
    cv = c1.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd = c2.text_area("Paste Job Description", height=200)
    
    if st.button("Analyze Match", type="primary"):
        if cv and jd:
            with st.spinner("Analyzing semantic vectors..."):
                ctrl = Controller()
                res = ctrl.process(cv, jd, st.session_state.lang, api_key, use_gpt)
                
                if res:
                    # Score Card
                    score = res.overall_score * 100
                    color = "red" if score < 50 else "orange" if score < 75 else "green"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
                        <h2 style="margin:0; color: #31333F;">Overall Match</h2>
                        <h1 style="margin:0; font-size: 3em; color: {color};">{score:.1f}%</h1>
                        <p style="margin:0;"><b>{res.detailed_feedback['overall']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Semantic Match", f"{res.overall_score*100:.0f}%", help="Based on meaning/context")
                    m2.metric("Keyword Overlap", f"{res.keyword_score*100:.0f}%", help="Based on exact word matches")
                    m3.metric("Raw AI Score", f"{res.raw_ai_score:.3f}", help="The raw Cosine Similarity (0-1)")

                    # Keywords
                    st.subheader("🔑 Keyword Gap Analysis")
                    k1, k2 = st.columns(2)
                    k1.success(f"✅ Matched ({len(res.matched_skills)})")
                    k1.write(", ".join(res.matched_skills) if res.matched_skills else "No exact matches found")
                    
                    k2.error(f"❌ Missing / Potential Gaps")
                    k2.write(", ".join(res.missing_skills) if res.missing_skills else "No major gaps found")
                    
                    # AI Suggestions
                    if res.ai_suggestions:
                        st.divider()
                        st.subheader("🤖 GPT-4 Recruiter Feedback")
                        st.write(res.ai_suggestions)
                else:
                    st.error("Could not read text from CV. Please try a different file format.")
        else:
            st.warning("Please provide both a CV and a Job Description.")

if __name__ == "__main__":
    main()
