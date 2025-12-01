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

# ==================== CONFIGURATION ====================

class Language(Enum):
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"

def get_safe_lang_code(lang) -> str:
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
                # Add space between pages to prevent word glueing
                text = " \n ".join([page.extract_text() or "" for page in reader.pages])
            elif file_type in ['docx', 'doc']:
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif file_type == 'txt':
                text = file.getvalue().decode('utf-8')
            
            # Basic cleanup of PDF artifacts
            text = text.replace('\n', ' ').replace('•', ' ').replace('●', ' ')
            return text
        except Exception as e:
            return ""

class CVEvaluator:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        
        # STOPWORDS: Common words to ignore
        self.stopwords = set("""
        the and for that with from have will work team skills experience years 
        responsible duties required preferred summary objective education
        und der die das mit für von dass wir erfahrung kenntnisse jahre aufgaben
        pour avec dans les des une est sur expérience
        para con las los una que por experiencia
        per con del della che una sono esperienza
        is a an or to in at be as on by it of
        """.split())

    def clean_text(self, text: str) -> str:
        """
        Smart cleaning that preserves C++, C#, .NET, Node.js
        """
        text = text.lower()
        # Replace mostly non-alphanumeric, BUT keep +, #, . for tech terms
        # This regex keeps letters, numbers, whitespace, +, #, and dots
        text = re.sub(r'[^a-z0-9\s\+\#\.]', ' ', text)
        return text

    def get_ngrams(self, text: str, n: int) -> Set[str]:
        """Generates n-grams (phrases) from text"""
        words = [w for w in text.split() if len(w) > 1 and w not in self.stopwords]
        return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))

    def extract_smart_keywords(self, text: str) -> Set[str]:
        """Extracts both single words AND 2-word phrases (Bigrams)"""
        clean = self.clean_text(text)
        
        # 1. Unigrams (Single words: "Python", "Java")
        unigrams = self.get_ngrams(clean, 1)
        
        # 2. Bigrams (Phrases: "Project Management", "Machine Learning")
        bigrams = self.get_ngrams(clean, 2)
        
        # Combine them
        return unigrams.union(bigrams)

    def calculate_similarity(self, cv_text: str, job_desc: str) -> float:
        """Calculates raw semantic similarity"""
        # Chunking strategy: Encodes text in smaller windows if too large (simplified here)
        emb1 = self.model.encode(cv_text[:4000], convert_to_tensor=True) # Limit to first ~4000 chars to avoid error
        emb2 = self.model.encode(job_desc[:4000], convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0])

    def sigmoid_normalize(self, raw_score: float) -> float:
        """Converts Robot Math to Human Grades"""
        # Adjusted curve for better "feeling"
        x0 = 0.25 # Center point (easier to pass)
        k = 10    # Steepness
        return 1 / (1 + math.exp(-k * (raw_score - x0)))

    def evaluate(self, cv_text: str, job_desc: str, lang_code: str) -> EvaluationResult:
        # 1. Semantic Match (The Meaning)
        raw_ai_score = self.calculate_similarity(cv_text, job_desc)
        human_semantic_score = self.sigmoid_normalize(raw_ai_score)
        
        # 2. Smart Keyword Match (Unigrams + Bigrams)
        cv_keywords = self.extract_smart_keywords(cv_text)
        jd_keywords = self.extract_smart_keywords(job_desc)
        
        if not jd_keywords:
            keyword_score = 0.0
            matched = []
            missing = []
        else:
            matched = list(cv_keywords.intersection(jd_keywords))
            missing = list(jd_keywords - cv_keywords)
            
            # Score Calculation: match / total_jd_keywords
            # We add a small boost (1.2x) because CVs rarely have EVERYTHING
            raw_k_score = len(matched) / len(jd_keywords)
            keyword_score = min(1.0, raw_k_score * 1.5)

        # 3. Final Weighted Score 
        # 60% Semantic (Context), 40% Keywords (Hard Skills)
        final_score = (human_semantic_score * 0.6) + (keyword_score * 0.4)
        
        # Limit missing skills display to just the most relevant single words or distinct bigrams
        # Filter out very short missing keywords to reduce noise
        filtered_missing = [m for m in missing if len(m) > 4]
        
        return EvaluationResult(
            overall_score=final_score,
            raw_ai_score=raw_ai_score,
            keyword_score=keyword_score,
            detailed_feedback=self.generate_feedback(final_score, lang_code),
            matched_keywords=sorted(matched, key=len, reverse=True)[:15], # Show longest phrases first
            missing_keywords=sorted(filtered_missing, key=len, reverse=True)[:15],
            ai_suggestions=None
        )

    def generate_feedback(self, score: float, lang: str) -> Dict[str, str]:
        texts = {
            "en": ["Needs Improvement", "Fair Match", "Good Match", "Excellent Match"],
            "de": ["Verbesserungswürdig", "Akzeptabel", "Gut", "Exzellent"],
            "fr": ["À améliorer", "Correct", "Bon", "Excellent"],
            "es": ["Necesita mejorar", "Aceptable", "Bueno", "Excelente"],
            "it": ["Da migliorare", "Accettabile", "Buono", "Eccellente"]
        }
        labels = texts.get(lang, texts["en"])
        
        if score < 0.4: idx = 0
        elif score < 0.6: idx = 1
        elif score < 0.75: idx = 2
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
        Act as a hiring manager. Compare the CV to the JD.
        
        SCORE: {result.overall_score*100:.1f}%
        MISSING KEYWORDS DETECTED: {', '.join(result.missing_keywords[:10])}
        
        JOB DESCRIPTION:
        {job_desc[:1000]}
        
        CV CONTENT:
        {cv_text[:1000]}
        
        Provide 3 clear bullet points on exactly what to change in the CV to get hired.
        Focus on Hard Skills. Output language: {lang_name}.
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
        
        if len(cv_text) < 50:
            return None
            
        result = self.evaluator.evaluate(cv_text, jd, lang_code)
        
        if use_gpt and api_key:
            if not self.gpt or self.gpt.language_code != lang_code:
                self.gpt = GPTEvaluator(api_key, lang_code)
            result.ai_suggestions = self.gpt.get_detailed_suggestions(cv_text, jd, result)
            
        return result

def main():
    st.set_page_config(page_title="Pro CV Matcher", page_icon="💼", layout="wide")
    
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    
    st.sidebar.title("⚙️ Setup")
    lang_map = {"en": "English", "de": "Deutsch", "fr": "Français", "es": "Español", "it": "Italiano"}
    sel_lang = st.sidebar.selectbox("Language", options=list(lang_map.keys()), format_func=lambda x: lang_map[x])
    st.session_state.lang = sel_lang
    
    use_gpt = st.sidebar.checkbox("Enable GPT-4 Suggestions")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password") if use_gpt else None
    
    st.title(f"💼 Pro CV Matcher ({lang_map[st.session_state.lang]})")
    st.markdown("Uses **N-Gram Analysis** (detects phrases like 'Project Management') and **Smart Cleaning** (keeps 'C++' and 'C#').")
    
    c1, c2 = st.columns(2)
    cv = c1.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd = c2.text_area("Paste Job Description", height=200)
    
    if st.button("Analyze Match", type="primary"):
        if cv and jd:
            with st.spinner("Analyzing phrases and semantics..."):
                ctrl = Controller()
                res = ctrl.process(cv, jd, st.session_state.lang, api_key, use_gpt)
                
                if res:
                    score = res.overall_score * 100
                    color = "#ff4b4b" if score < 50 else "#ffa421" if score < 75 else "#21c354"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 25px; background-color: #f8f9fa; border-radius: 12px; border: 1px solid #e9ecef; margin-bottom: 25px;">
                        <h3 style="margin:0; color: #6c757d;">Match Quality</h3>
                        <h1 style="margin:10px 0; font-size: 3.5em; color: {color};">{score:.0f}%</h1>
                        <p style="margin:0; font-weight:bold; font-size: 1.2em;">{res.detailed_feedback['overall']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    m1, m2 = st.columns(2)
                    m1.progress(res.overall_score, text="Match Confidence")
                    
                    st.subheader("🔑 Smart Keyword Analysis (Phrases)")
                    t1, t2 = st.tabs(["✅ Matched Skills", "❌ Missing Skills"])
                    
                    with t1:
                        if res.matched_keywords:
                            # Display as tags
                            st.write("The CV contains these key phrases from the Job Description:")
                            st.markdown(" ".join([f"`{k}`" for k in res.matched_keywords]), unsafe_allow_html=True)
                        else:
                            st.warning("No direct keyword phrases found.")
                            
                    with t2:
                        if res.missing_keywords:
                            st.write("Consider adding these exact phrases to your CV:")
                            st.markdown(" ".join([f"`{k}`" for k in res.missing_keywords]), unsafe_allow_html=True)
                        else:
                            st.success("No major missing keywords detected!")
                    
                    if res.ai_suggestions:
                        st.divider()
                        st.subheader("🤖 AI Career Coach")
                        st.info(res.ai_suggestions)
                        
                    with st.expander("View Debug Info"):
                        st.json({
                            "raw_semantic_score": res.raw_ai_score,
                            "keyword_score": res.keyword_score,
                            "final_calculation": f"({res.raw_ai_score:.2f} normalized * 0.6) + ({res.keyword_score:.2f} * 0.4)"
                        })
                else:
                    st.error("Could not read text from CV.")
        else:
            st.warning("Please upload a CV and enter a Job Description.")

if __name__ == "__main__":
    main()
