import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import openai
from enum import Enum
import math
import re

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

# ==================== MODEL LAYER ====================

@dataclass
class EvaluationResult:
    overall_score: float
    relevance_score: float
    keyword_match_score: float
    detailed_feedback: Dict[str, str]
    matched_skills: List[str]
    missing_skills: List[str]
    ai_suggestions: Optional[str] = None

class CVParser:
    """Handles CV document parsing with better error safety"""
    
    @staticmethod
    def extract_text(file) -> str:
        file_type = file.name.split('.')[-1].lower()
        try:
            if file_type == 'pdf':
                pdf_reader = PyPDF2.PdfReader(file)
                text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
                return text
            elif file_type in ['docx', 'doc']:
                doc = docx.Document(file)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            elif file_type == 'txt':
                return file.getvalue().decode('utf-8')
            else:
                return ""
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return ""

class GPTEvaluator:
    """GPT-4 based evaluation for detailed suggestions"""
    
    def __init__(self, api_key: str, language_code: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.language_code = language_code
    
    def get_detailed_suggestions(self, cv_text: str, job_desc: str, 
                                eval_result: EvaluationResult) -> str:
        
        lang_names = {
            "en": "English", "de": "German", "fr": "French", 
            "es": "Spanish", "it": "Italian"
        }
        lang_name = lang_names.get(self.language_code, "English")
        
        prompt = f"""You are an expert technical recruiter. Analyze this CV against the Job Description.

Job Description:
{job_desc[:2000]}

CV Snippet:
{cv_text[:2000]}

Data:
- Match Score: {eval_result.overall_score*100:.1f}%
- Missing Keywords: {', '.join(eval_result.missing_skills[:10])}

Task:
Provide 3 highly specific, critical changes to improve this CV's match rate. Focus on hard skills and measurable results.
Respond in {lang_name}."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a helpful recruiter speaking {lang_name}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Suggestion Error: {str(e)}"

class CVEvaluator:
    """Core evaluation logic with Expert Calibration"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        
        # EXPERT TUNING: Stopwords to ignore in keyword calculation
        # These are filler words that inflate/deflate scores artificially.
        self.stopwords = {
            # English
            "the", "and", "for", "that", "this", "with", "from", "have", "will", 
            "work", "team", "skills", "experience", "years", "responsible", "duties",
            "required", "preferred", "qualification", "summary", "objective",
            # German
            "und", "der", "die", "das", "mit", "für", "von", "dass", "wir", 
            "erfahrung", "kenntnisse", "jahre", "teamfähig", "aufgaben",
            # French
            "pour", "avec", "dans", "les", "des", "une", "est", "sur", "expérience",
            # Spanish
            "para", "con", "las", "los", "una", "que", "por", "experiencia",
            # Italian
            "per", "con", "del", "della", "che", "una", "sono", "esperienza"
        }
    
    def calculate_semantic_similarity(self, cv_text: str, job_desc: str) -> float:
        # 1. Compute Embeddings
        cv_embedding = self.model.encode(cv_text, convert_to_tensor=True)
        job_embedding = self.model.encode(job_desc, convert_to_tensor=True)
        
        # 2. Compute Raw Cosine Similarity (Usually between 0.0 and 0.5 for text)
        similarity = util.cos_sim(cv_embedding, job_embedding)
        return float(similarity[0][0])
    
    def calibrate_score(self, raw_score: float) -> float:
        """
        EXPERT CALIBRATION:
        Raw Cosine Similarity is rarely 1.0 for documents.
        A score of 0.45 is usually a "Great Match".
        A score of 0.15 is usually "Irrelevant".
        We map [0.15, 0.75] -> [0.0, 1.0]
        """
        lower_bound = 0.15
        upper_bound = 0.75
        
        if raw_score <= lower_bound:
            return 0.0
        if raw_score >= upper_bound:
            return 1.0
            
        # Linear mapping between bounds
        return (raw_score - lower_bound) / (upper_bound - lower_bound)

    def extract_keywords(self, text: str) -> List[str]:
        # Simple but effective cleaning
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        # Filter out stopwords and short words
        keywords = [
            w for w in words 
            if len(w) > 3 
            and w not in self.stopwords
            and not w.isdigit()
        ]
        return list(set(keywords))
    
    def calculate_keyword_match(self, cv_text: str, job_desc: str) -> Tuple[float, List[str], List[str]]:
        cv_keywords = set(self.extract_keywords(cv_text))
        job_keywords = set(self.extract_keywords(job_desc))
        
        if not job_keywords:
            return 0.0, [], []
        
        matched = cv_keywords.intersection(job_keywords)
        missing = job_keywords - cv_keywords
        
        raw_match = len(matched) / len(job_keywords)
        
        # Curve the keyword score: If you have 40% of the exact keywords, that's excellent.
        # We multiply by 2.0, capping at 1.0
        adjusted_match = min(1.0, raw_match * 2.0)
        
        return adjusted_match, list(matched)[:20], list(missing)[:20]
    
    def evaluate(self, cv_text: str, job_desc: str, language_code: str) -> EvaluationResult:
        # 1. Semantics (The "Meaning" Match)
        raw_relevance = self.calculate_semantic_similarity(cv_text, job_desc)
        calibrated_relevance = self.calibrate_score(raw_relevance)
        
        # 2. Keywords (The "Buzzword" Match)
        keyword_score, matched, missing = self.calculate_keyword_match(cv_text, job_desc)
        
        # 3. Weighted Score (75% Meaning, 25% Keywords)
        overall_score = (calibrated_relevance * 0.75) + (keyword_score * 0.25)
        
        feedback = self._generate_feedback(overall_score, calibrated_relevance, keyword_score, language_code)
        
        return EvaluationResult(
            overall_score=overall_score,
            relevance_score=calibrated_relevance,
            keyword_match_score=keyword_score,
            detailed_feedback=feedback,
            matched_skills=matched,
            missing_skills=missing
        )
    
    def _generate_feedback(self, overall: float, relevance: float, 
                          keyword: float, language_code: str) -> Dict[str, str]:
        
        FEEDBACK_DB = {
            "en": {
                "excellent": "Excellent match! Your profile is highly competitive.",
                "good": "Good match. You have the core skills, but could optimize further.",
                "moderate": "Moderate match. Focus on including more specific keywords from the job desc.",
                "low": "Low match. The semantics of your CV do not align well with this role.",
            },
            "de": {
                "excellent": "Exzellentes Ergebnis! Ihr Profil ist sehr wettbewerbsfähig.",
                "good": "Gutes Ergebnis. Sie haben die Kernkompetenzen, könnten aber noch optimieren.",
                "moderate": "Mäßiges Ergebnis. Versuchen Sie, mehr spezifische Begriffe aus der Stelle zu nutzen.",
                "low": "Geringes Ergebnis. Die Semantik Ihres Lebenslaufs passt nicht gut zur Rolle.",
            }
            # (Simplified for brevity, English/German defaults cover most testing)
        }
        
        # Default to English if missing
        texts = FEEDBACK_DB.get(language_code, FEEDBACK_DB["en"])
        
        feedback = {}
        
        # Grading Scale
        if overall >= 0.80:
            feedback['overall'] = texts["excellent"]
        elif overall >= 0.60:
            feedback['overall'] = texts["good"]
        elif overall >= 0.40:
            feedback['overall'] = texts["moderate"]
        else:
            feedback['overall'] = texts["low"]
            
        return feedback

# ==================== CONTROLLER LAYER ====================

class CVEvaluationController:
    def __init__(self):
        self.parser = CVParser()
        self.gpt_evaluator = None
        self.evaluator = None
    
    @st.cache_resource
    def load_expert_model(_self):
        return CVEvaluator()
    
    def process_evaluation(self, cv_file, job_description: str, 
                          language: Language, api_key: Optional[str] = None,
                          use_gpt: bool = False) -> EvaluationResult:
        
        lang_code = get_safe_lang_code(language)
        
        if self.evaluator is None:
            self.evaluator = self.load_expert_model()
        
        cv_text = self.parser.extract_text(cv_file)
        if len(cv_text) < 50:
             # Basic error handling for empty/unreadable PDFs
            return EvaluationResult(0,0,0, {"overall": "Error reading CV text."}, [], [])

        result = self.evaluator.evaluate(cv_text, job_description, lang_code)
        
        if use_gpt and api_key:
            try:
                if self.gpt_evaluator is None or self.gpt_evaluator.language_code != lang_code:
                    self.gpt_evaluator = GPTEvaluator(api_key, lang_code)
                
                result.ai_suggestions = self.gpt_evaluator.get_detailed_suggestions(
                    cv_text, job_description, result
                )
            except Exception as e:
                result.ai_suggestions = f"GPT Error: {str(e)}"
        
        return result

# ==================== VIEW LAYER ====================

def render_sidebar(language: Language):
    st.sidebar.title("⚙️ Settings")
    
    lang_options = [
        ("English", Language.ENGLISH),
        ("Deutsch", Language.GERMAN),
        ("Français", Language.FRENCH),
        ("Español", Language.SPANISH),
        ("Italiano", Language.ITALIAN)
    ]
    
    current_code = get_safe_lang_code(language)
    current_index = 0
    for i, (_, lang) in enumerate(lang_options):
        if lang.value == current_code:
            current_index = i
            break
            
    selected_lang_tuple = st.sidebar.selectbox(
        "Select Language",
        options=lang_options,
        format_func=lambda x: x[0],
        index=current_index
    )
    
    st.sidebar.markdown("---")
    use_gpt = st.sidebar.checkbox("Use GPT-4 (Optional)", value=False)
    api_key = None
    if use_gpt:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        
    return selected_lang_tuple[1], use_gpt, api_key

def main():
    st.set_page_config(page_title="Expert CV Evaluator", page_icon="🚀", layout="wide")
    
    if 'language' not in st.session_state:
        st.session_state.language = Language.ENGLISH
        
    language, use_gpt, api_key = render_sidebar(st.session_state.language)
    st.session_state.language = language
    
    st.title("🚀 AI Smart Resume Matcher")
    st.markdown("This tool uses **Expert Calibrated** scoring. A score of **60%+** is considered a good match.")
    
    col1, col2 = st.columns(2)
    with col1:
        cv_file = st.file_uploader("Upload CV (PDF/DOCX)", type=['pdf', 'docx', 'txt'])
    with col2:
        job_desc = st.text_area("Job Description", height=150)
        
    if st.button("Analyze Match", type="primary"):
        if cv_file and job_desc:
            with st.spinner("Calculating semantic vectors..."):
                controller = CVEvaluationController()
                result = controller.process_evaluation(cv_file, job_desc, language, api_key, use_gpt)
                
                # Visualization
                st.divider()
                score = result.overall_score * 100
                
                # Dynamic Color
                if score >= 80: color = "green"
                elif score >= 60: color = "orange"
                else: color = "red"
                
                st.markdown(f"<h1 style='text-align: center; color: {color}'>{score:.1f}% Match</h1>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("Semantic Match (Meaning)", f"{result.relevance_score*100:.1f}%")
                c2.metric("Keyword Match (Buzzwords)", f"{result.keyword_match_score*100:.1f}%")
                
                st.info(f"**Feedback:** {result.detailed_feedback.get('overall', '')}")
                
                if result.ai_suggestions:
                    st.subheader("🤖 AI Suggestions")
                    st.write(result.ai_suggestions)
                    
                st.subheader("🔍 Keyword Analysis")
                c3, c4 = st.columns(2)
                c3.success(f"Matched: {', '.join(result.matched_skills)}")
                c4.error(f"Missing: {', '.join(result.missing_skills)}")
        else:
            st.warning("Please upload a CV and enter a Job Description.")

if __name__ == "__main__":
    main()
