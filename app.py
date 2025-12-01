import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import openai
from enum import Enum

# ==================== CONFIGURATION ====================

class Language(Enum):
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"

# Helper to always get a string "en", "de", etc.
def get_safe_lang_code(lang) -> str:
    if isinstance(lang, Language):
        return lang.value
    if isinstance(lang, str):
        return lang
    try:
        return lang.value
    except AttributeError:
        return "en"

# KEYS ARE NOW STRICTLY STRINGS
TRANSLATIONS = {
    "en": {
        "title": "🎯 AI-Powered CV Evaluator",
        "description": "Upload your CV and paste the job description to get an AI-powered evaluation of how well your CV matches the position requirements.",
        "upload_cv": "📄 Upload Your CV",
        "job_desc": "📋 Job Description",
        "job_desc_placeholder": "Paste the full job description including responsibilities and requirements...",
        "evaluate_btn": "🚀 Evaluate CV",
        "results_title": "📊 Evaluation Results",
        "overall_score": "Overall Match Score",
        "semantic_relevance": "Semantic Relevance",
        "keyword_match": "Keyword Match",
        "detailed_feedback": "💡 Detailed Feedback",
        "ai_suggestions": "🤖 AI-Powered Suggestions",
        "matched_keywords": "✅ Matched Keywords",
        "missing_keywords": "❌ Missing Keywords",
        "error_no_cv": "Please upload a CV file",
        "error_no_job": "Please enter a job description",
        "error_no_api": "Please enter your OpenAI API key in the sidebar",
        "analyzing": "Analyzing your CV... This may take a moment.",
        "api_key_label": "OpenAI API Key",
        "api_key_help": "Enter your OpenAI API key for enhanced AI suggestions",
        "language_label": "Select Language",
        "model_selection": "Evaluation Model",
        "use_gpt": "Use GPT-4 for detailed suggestions",
    },
    "de": {
        "title": "🎯 KI-gestützte Lebenslauf-Bewertung",
        "description": "Laden Sie Ihren Lebenslauf hoch und fügen Sie die Stellenbeschreibung ein, um eine KI-gestützte Bewertung zu erhalten.",
        "upload_cv": "📄 Lebenslauf hochladen",
        "job_desc": "📋 Stellenbeschreibung",
        "job_desc_placeholder": "Fügen Sie die vollständige Stellenbeschreibung mit Aufgaben und Anforderungen ein...",
        "evaluate_btn": "🚀 Lebenslauf bewerten",
        "results_title": "📊 Bewertungsergebnisse",
        "overall_score": "Gesamtübereinstimmung",
        "semantic_relevance": "Semantische Relevanz",
        "keyword_match": "Schlüsselwort-Übereinstimmung",
        "detailed_feedback": "💡 Detailliertes Feedback",
        "ai_suggestions": "🤖 KI-gestützte Vorschläge",
        "matched_keywords": "✅ Übereinstimmende Schlüsselwörter",
        "missing_keywords": "❌ Fehlende Schlüsselwörter",
        "error_no_cv": "Bitte laden Sie eine Lebenslauf-Datei hoch",
        "error_no_job": "Bitte geben Sie eine Stellenbeschreibung ein",
        "error_no_api": "Bitte geben Sie Ihren OpenAI API-Schlüssel in der Seitenleiste ein",
        "analyzing": "Analysiere Ihren Lebenslauf... Dies kann einen Moment dauern.",
        "api_key_label": "OpenAI API-Schlüssel",
        "api_key_help": "Geben Sie Ihren OpenAI API-Schlüssel für erweiterte KI-Vorschläge ein",
        "language_label": "Sprache auswählen",
        "model_selection": "Bewertungsmodell",
        "use_gpt": "GPT-4 für detaillierte Vorschläge verwenden",
    },
    "fr": {
        "title": "🎯 Évaluateur de CV alimenté par l'IA",
        "description": "Téléchargez votre CV et collez la description du poste pour obtenir une évaluation alimentée par l'IA.",
        "upload_cv": "📄 Télécharger votre CV",
        "job_desc": "📋 Description du poste",
        "job_desc_placeholder": "Collez la description complète du poste avec les responsabilités et les exigences...",
        "evaluate_btn": "🚀 Évaluer le CV",
        "results_title": "📊 Résultats de l'évaluation",
        "overall_score": "Score de correspondance global",
        "semantic_relevance": "Pertinence sémantique",
        "keyword_match": "Correspondance des mots-clés",
        "detailed_feedback": "💡 Commentaires détaillés",
        "ai_suggestions": "🤖 Suggestions alimentées par l'IA",
        "matched_keywords": "✅ Mots-clés correspondants",
        "missing_keywords": "❌ Mots-clés manquants",
        "error_no_cv": "Veuillez télécharger un fichier CV",
        "error_no_job": "Veuillez entrer une description de poste",
        "error_no_api": "Veuillez entrer votre clé API OpenAI dans la barre latérale",
        "analyzing": "Analyse de votre CV... Cela peut prendre un moment.",
        "api_key_label": "Clé API OpenAI",
        "api_key_help": "Entrez votre clé API OpenAI pour des suggestions IA améliorées",
        "language_label": "Sélectionner la langue",
        "model_selection": "Modèle d'évaluation",
        "use_gpt": "Utiliser GPT-4 pour des suggestions détaillées",
    },
    "es": {
        "title": "🎯 Evaluador de CV con IA",
        "description": "Sube tu CV y pega la descripción del trabajo para obtener una evaluación impulsada por IA.",
        "upload_cv": "📄 Subir tu CV",
        "job_desc": "📋 Descripción del trabajo",
        "job_desc_placeholder": "Pega la descripción completa del trabajo con responsabilidades y requisitos...",
        "evaluate_btn": "🚀 Evaluar CV",
        "results_title": "📊 Resultados de la evaluación",
        "overall_score": "Puntuación de coincidencia general",
        "semantic_relevance": "Relevancia semántica",
        "keyword_match": "Coincidencia de palabras clave",
        "detailed_feedback": "💡 Retroalimentación detallada",
        "ai_suggestions": "🤖 Sugerencias impulsadas por IA",
        "matched_keywords": "✅ Palabras clave coincidentes",
        "missing_keywords": "❌ Palabras clave faltantes",
        "error_no_cv": "Por favor sube un archivo de CV",
        "error_no_job": "Por favor ingresa una descripción del trabajo",
        "error_no_api": "Por favor ingresa tu clave API de OpenAI en la barra lateral",
        "analyzing": "Analizando tu CV... Esto puede tomar un momento.",
        "api_key_label": "Clave API de OpenAI",
        "api_key_help": "Ingresa tu clave API de OpenAI para sugerencias IA mejoradas",
        "language_label": "Seleccionar idioma",
        "model_selection": "Modelo de evaluación",
        "use_gpt": "Usar GPT-4 para sugerencias detalladas",
    },
    "it": {
        "title": "🎯 Valutatore CV alimentato dall'IA",
        "description": "Carica il tuo CV e incolla la descrizione del lavoro per ottenere una valutazione alimentata dall'IA.",
        "upload_cv": "📄 Carica il tuo CV",
        "job_desc": "📋 Descrizione del lavoro",
        "job_desc_placeholder": "Incolla la descrizione completa del lavoro con responsabilità e requisiti...",
        "evaluate_btn": "🚀 Valuta CV",
        "results_title": "📊 Risultati della valutazione",
        "overall_score": "Punteggio di corrispondenza complessivo",
        "semantic_relevance": "Rilevanza semantica",
        "keyword_match": "Corrispondenza parole chiave",
        "detailed_feedback": "💡 Feedback dettagliato",
        "ai_suggestions": "🤖 Suggerimenti alimentati dall'IA",
        "matched_keywords": "✅ Parole chiave corrispondenti",
        "missing_keywords": "❌ Parole chiave mancanti",
        "error_no_cv": "Si prega di caricare un file CV",
        "error_no_job": "Si prega di inserire una descrizione del lavoro",
        "error_no_api": "Si prega di inserire la chiave API OpenAI nella barra laterale",
        "analyzing": "Analisi del tuo CV... Questo potrebbe richiedere un momento.",
        "api_key_label": "Chiave API OpenAI",
        "api_key_help": "Inserisci la tua chiave API OpenAI per suggerimenti IA avanzati",
        "language_label": "Seleziona lingua",
        "model_selection": "Modello di valutazione",
        "use_gpt": "Usa GPT-4 per suggerimenti dettagliati",
    }
}

# KEYS ARE NOW STRICTLY STRINGS
FEEDBACK_TEMPLATES = {
    "en": {
        "excellent": "Excellent match! Your CV aligns very well with the job requirements.",
        "good": "Good match. Your CV shows relevant experience with room for improvement.",
        "moderate": "Moderate match. Consider tailoring your CV more specifically to this role.",
        "low": "Low match. Your CV may not align well with this position's requirements.",
        "relevance_high": "Strong semantic alignment with job description.",
        "relevance_medium": "Moderate alignment. Consider using similar terminology from the job posting.",
        "relevance_low": "Weak alignment. Restructure your CV to better reflect job requirements.",
        "keywords_high": "Good keyword coverage from the job description.",
        "keywords_medium": "Moderate keyword presence. Add more relevant skills and terms.",
        "keywords_low": "Low keyword match. Include more specific skills mentioned in the job posting."
    },
    "de": {
        "excellent": "Ausgezeichnete Übereinstimmung! Ihr Lebenslauf entspricht sehr gut den Anforderungen.",
        "good": "Gute Übereinstimmung. Ihr Lebenslauf zeigt relevante Erfahrung mit Verbesserungspotenzial.",
        "moderate": "Mittlere Übereinstimmung. Passen Sie Ihren Lebenslauf gezielter an diese Rolle an.",
        "low": "Geringe Übereinstimmung. Ihr Lebenslauf passt möglicherweise nicht gut zu dieser Position.",
        "relevance_high": "Starke semantische Übereinstimmung mit der Stellenbeschreibung.",
        "relevance_medium": "Mittlere Übereinstimmung. Verwenden Sie ähnliche Begriffe aus der Stellenausschreibung.",
        "relevance_low": "Schwache Übereinstimmung. Strukturieren Sie Ihren Lebenslauf neu.",
        "keywords_high": "Gute Schlüsselwort-Abdeckung aus der Stellenbeschreibung.",
        "keywords_medium": "Mittlere Schlüsselwort-Präsenz. Fügen Sie mehr relevante Begriffe hinzu.",
        "keywords_low": "Geringe Schlüsselwort-Übereinstimmung. Fügen Sie spezifische Fähigkeiten hinzu."
    },
    "fr": {
        "excellent": "Excellente correspondance! Votre CV correspond très bien aux exigences du poste.",
        "good": "Bonne correspondance. Votre CV montre une expérience pertinente avec des améliorations possibles.",
        "moderate": "Correspondance modérée. Adaptez votre CV plus spécifiquement à ce rôle.",
        "low": "Faible correspondance. Votre CV ne correspond peut-être pas bien à ce poste.",
        "relevance_high": "Forte alignement sémantique avec la description du poste.",
        "relevance_medium": "Alignement modéré. Utilisez une terminologie similaire de l'offre d'emploi.",
        "relevance_low": "Faible alignement. Restructurez votre CV pour mieux refléter les exigences.",
        "keywords_high": "Bonne couverture des mots-clés de la description du poste.",
        "keywords_medium": "Présence modérée de mots-clés. Ajoutez plus de termes pertinents.",
        "keywords_low": "Faible correspondance de mots-clés. Incluez plus de compétences spécifiques."
    },
    "es": {
        "excellent": "¡Excelente coincidencia! Tu CV se alinea muy bien con los requisitos del trabajo.",
        "good": "Buena coincidencia. Tu CV muestra experiencia relevante con margen de mejora.",
        "moderate": "Coincidencia moderada. Considera adaptar tu CV más específicamente a este rol.",
        "low": "Baja coincidencia. Tu CV puede no alinearse bien con los requisitos de esta posición.",
        "relevance_high": "Fuerte alineación semántica con la descripción del trabajo.",
        "relevance_medium": "Alineación moderada. Considera usar terminología similar de la oferta de trabajo.",
        "relevance_low": "Alineación débil. Reestructura tu CV para reflejar mejor los requisitos.",
        "keywords_high": "Buena cobertura de palabras clave de la descripción del trabajo.",
        "keywords_medium": "Presencia moderada de palabras clave. Añade más términos relevantes.",
        "keywords_low": "Baja coincidencia de palabras clave. Incluye más habilidades específicas."
    },
    "it": {
        "excellent": "Corrispondenza eccellente! Il tuo CV si allinea molto bene con i requisiti del lavoro.",
        "good": "Buona corrispondenza. Il tuo CV mostra esperienza rilevante con margine di miglioramento.",
        "moderate": "Corrispondenza moderata. Considera di adattare il tuo CV più specificamente a questo ruolo.",
        "low": "Bassa corrispondenza. Il tuo CV potrebbe non allinearsi bene con i requisiti di questa posizione.",
        "relevance_high": "Forte allineamento semantico con la descrizione del lavoro.",
        "relevance_medium": "Allineamento moderato. Considera l'uso di terminologia simile dall'offerta di lavoro.",
        "relevance_low": "Allineamento debole. Ristruttura il tuo CV per riflettere meglio i requisiti.",
        "keywords_high": "Buona copertura di parole chiave dalla descrizione del lavoro.",
        "keywords_medium": "Presenza moderata di parole chiave. Aggiungi più termini rilevanti.",
        "keywords_low": "Bassa corrispondenza di parole chiave. Includi più competenze specifiche."
    }
}

# ==================== MODEL LAYER ====================

@dataclass
class EvaluationResult:
    """Data class for evaluation results"""
    overall_score: float
    relevance_score: float
    keyword_match_score: float
    detailed_feedback: Dict[str, str]
    matched_skills: List[str]
    missing_skills: List[str]
    ai_suggestions: Optional[str] = None

class CVParser:
    """Handles CV document parsing"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    @staticmethod
    def extract_text(file) -> str:
        """Extract text from uploaded file"""
        file_type = file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            return CVParser.extract_text_from_pdf(file)
        elif file_type in ['docx', 'doc']:
            return CVParser.extract_text_from_docx(file)
        elif file_type == 'txt':
            return file.getvalue().decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

class GPTEvaluator:
    """GPT-4 based evaluation for detailed suggestions"""
    
    def __init__(self, api_key: str, language_code: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.language_code = language_code
    
    def get_detailed_suggestions(self, cv_text: str, job_desc: str, 
                                eval_result: EvaluationResult) -> str:
        """Get detailed AI-powered suggestions using GPT-4"""
        
        language_prompts = {
            "en": "English",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian"
        }
        
        # Use string code directly
        lang_name = language_prompts.get(self.language_code, "English")
        
        prompt = f"""You are an expert career advisor. Analyze the following CV against the job description and provide specific, actionable suggestions for improvement.

Job Description:
{job_desc[:2000]}

CV Content:
{cv_text[:2000]}

Current Scores:
- Overall Match: {eval_result.overall_score*100:.1f}%
- Semantic Relevance: {eval_result.relevance_score*100:.1f}%
- Keyword Match: {eval_result.keyword_match_score*100:.1f}%

Matched Skills: {', '.join(eval_result.matched_skills[:5])}
Missing Skills: {', '.join(eval_result.missing_skills[:5])}

Please provide:
1. Top 3 specific improvements to make the CV more competitive
2. Key skills or experiences to emphasize
3. Recommended CV structure changes
4. Any red flags or gaps to address

Respond in {lang_name} language. Keep it concise and actionable."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a professional career advisor providing feedback in {lang_name}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating AI suggestions: {str(e)}"

class CVEvaluator:
    """Core evaluation logic using transformer models"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """Use multilingual model for better language support"""
        self.model = SentenceTransformer(model_name)
    
    def calculate_semantic_similarity(self, cv_text: str, job_desc: str) -> float:
        """Calculate semantic similarity between CV and job description"""
        cv_embedding = self.model.encode(cv_text, convert_to_tensor=True)
        job_embedding = self.model.encode(job_desc, convert_to_tensor=True)
        
        similarity = util.cos_sim(cv_embedding, job_embedding)
        return float(similarity[0][0])
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        keywords = []
        text_lower = text.lower()
        
        words = text_lower.split()
        keywords = [word.strip('.,!?;:()[]{}') for word in words if len(word) > 3]
        return list(set(keywords))
    
    def calculate_keyword_match(self, cv_text: str, job_desc: str) -> Tuple[float, List[str], List[str]]:
        """Calculate keyword overlap between CV and job description"""
        cv_keywords = set(self.extract_keywords(cv_text))
        job_keywords = set(self.extract_keywords(job_desc))
        
        if not job_keywords:
            return 0.0, [], []
        
        matched = cv_keywords.intersection(job_keywords)
        missing = job_keywords - cv_keywords
        
        match_score = len(matched) / len(job_keywords)
        return match_score, list(matched)[:15], list(missing)[:15]
    
    def evaluate(self, cv_text: str, job_desc: str, language_code: str) -> EvaluationResult:
        """Complete evaluation of CV against job description"""
        
        relevance_score = self.calculate_semantic_similarity(cv_text, job_desc)
        keyword_score, matched_skills, missing_skills = self.calculate_keyword_match(cv_text, job_desc)
        
        overall_score = (relevance_score * 0.7) + (keyword_score * 0.3)
        
        # NOTE: We pass the language_code string here, NOT the Enum
        feedback = self._generate_feedback(overall_score, relevance_score, keyword_score, language_code)
        
        return EvaluationResult(
            overall_score=overall_score,
            relevance_score=relevance_score,
            keyword_match_score=keyword_score,
            detailed_feedback=feedback,
            matched_skills=matched_skills,
            missing_skills=missing_skills
        )
    
    def _generate_feedback(self, overall: float, relevance: float, 
                          keyword: float, language_code: str) -> Dict[str, str]:
        """Generate human-readable feedback in selected language"""
        
        # DOUBLE CHECK: Ensure we are working with a string
        safe_lang = get_safe_lang_code(language_code)
            
        # Access the dictionary with the STRING key
        templates = FEEDBACK_TEMPLATES.get(safe_lang, FEEDBACK_TEMPLATES["en"])
        
        feedback = {}
        
        if overall >= 0.75:
            feedback['overall'] = templates["excellent"]
        elif overall >= 0.60:
            feedback['overall'] = templates["good"]
        elif overall >= 0.45:
            feedback['overall'] = templates["moderate"]
        else:
            feedback['overall'] = templates["low"]
        
        if relevance >= 0.70:
            feedback['relevance'] = templates["relevance_high"]
        elif relevance >= 0.50:
            feedback['relevance'] = templates["relevance_medium"]
        else:
            feedback['relevance'] = templates["relevance_low"]
        
        if keyword >= 0.40:
            feedback['keywords'] = templates["keywords_high"]
        elif keyword >= 0.25:
            feedback['keywords'] = templates["keywords_medium"]
        else:
            feedback['keywords'] = templates["keywords_low"]
        
        return feedback

# ==================== CONTROLLER LAYER ====================

class CVEvaluationController:
    """Orchestrates the evaluation process"""
    
    def __init__(self):
        self.parser = CVParser()
        self.evaluator = None
        self.gpt_evaluator = None
    
    @st.cache_resource
    def load_model(_self):
        """Load model with caching"""
        return CVEvaluator()
    
    def process_evaluation(self, cv_file, job_description: str, 
                          language: Language, api_key: Optional[str] = None,
                          use_gpt: bool = False) -> EvaluationResult:
        """Process complete evaluation workflow"""
        
        # =========================================================
        # CRITICAL FIX: CONVERT ENUM TO STRING IMMEDIATELY
        # This strips away the Enum object and passes only the string "de", "en", etc.
        # =========================================================
        lang_code = get_safe_lang_code(language)
        
        if self.evaluator is None:
            self.evaluator = self.load_model()
        
        cv_text = self.parser.extract_text(cv_file)
        
        # Pass the STRING to evaluate
        result = self.evaluator.evaluate(cv_text, job_description, lang_code)
        
        if use_gpt and api_key:
            try:
                # Use string comparison
                cached_lang_val = self.gpt_evaluator.language_code if self.gpt_evaluator else None
                
                if self.gpt_evaluator is None or cached_lang_val != lang_code:
                    self.gpt_evaluator = GPTEvaluator(api_key, lang_code)
                
                result.ai_suggestions = self.gpt_evaluator.get_detailed_suggestions(
                    cv_text, job_description, result
                )
            except Exception as e:
                result.ai_suggestions = f"Error: {str(e)}"
        
        return result

# ==================== VIEW LAYER (Streamlit UI) ====================

def get_text(key: str, language: Language) -> str:
    """Get translated text"""
    lang_code = get_safe_lang_code(language)
    return TRANSLATIONS.get(lang_code, TRANSLATIONS["en"]).get(key, key)

def render_sidebar(language: Language):
    """Render sidebar with settings"""
    st.sidebar.title("⚙️ Settings")
    
    # Language options
    lang_options = [
        ("English", Language.ENGLISH),
        ("Deutsch", Language.GERMAN),
        ("Français", Language.FRENCH),
        ("Español", Language.SPANISH),
        ("Italiano", Language.ITALIAN)
    ]
    
    # Safely find current index by comparing string values
    current_code = get_safe_lang_code(language)
    current_index = 0
    
    for i, (_, lang) in enumerate(lang_options):
        if lang.value == current_code:
            current_index = i
            break
    
    selected_lang = st.sidebar.selectbox(
        get_text("language_label", language),
        options=lang_options,
        format_func=lambda x: x[0],
        index=current_index
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(get_text("model_selection", language))
    
    use_gpt = st.sidebar.checkbox(
        get_text("use_gpt", language),
        value=False
    )
    
    api_key = None
    if use_gpt:
        api_key = st.sidebar.text_input(
            get_text("api_key_label", language),
            type="password",
            help=get_text("api_key_help", language)
        )
    
    return selected_lang[1], use_gpt, api_key

def render_header(language: Language):
    """Render application header"""
    st.title(get_text("title", language))
    st.markdown(get_text("description", language))

def render_input_section(language: Language) -> Tuple:
    """Render input section for CV and job description"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(get_text("upload_cv", language))
        cv_file = st.file_uploader(
            "",
            type=['pdf', 'docx', 'doc', 'txt'],
            help="PDF, DOCX, TXT"
        )
    
    with col2:
        st.subheader(get_text("job_desc", language))
        job_description = st.text_area(
            "",
            height=200,
            placeholder=get_text("job_desc_placeholder", language),
            label_visibility="collapsed"
        )
    
    return cv_file, job_description

def render_results(result: EvaluationResult, language: Language):
    """Render evaluation results"""
    st.subheader(get_text("results_title", language))
    
    score_percentage = result.overall_score * 100
    if score_percentage >= 75:
        color = "green"
    elif score_percentage >= 60:
        color = "orange"
    else:
        color = "red"
    
    st.markdown(
        f"### {get_text('overall_score', language)}: "
        f"<span style='color:{color}'>{score_percentage:.1f}%</span>", 
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(get_text("semantic_relevance", language), 
                 f"{result.relevance_score*100:.1f}%")
    
    with col2:
        st.metric(get_text("keyword_match", language), 
                 f"{result.keyword_match_score*100:.1f}%")
    
    st.subheader(get_text("detailed_feedback", language))
    for category, feedback in result.detailed_feedback.items():
        st.info(f"**{category.title()}:** {feedback}")
    
    if result.ai_suggestions:
        st.subheader(get_text("ai_suggestions", language))
        st.markdown(result.ai_suggestions)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(get_text("matched_keywords", language))
        if result.matched_skills:
            for skill in result.matched_skills[:10]:
                st.success(skill)
        else:
            st.write("-")
    
    with col2:
        st.subheader(get_text("missing_keywords", language))
        if result.missing_skills:
            for skill in result.missing_skills[:10]:
                st.warning(skill)
        else:
            st.write("-")

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="CV Evaluator",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize language safely
    if 'language' not in st.session_state:
        st.session_state.language = Language.ENGLISH
    
    # Safely recover language from session state
    current_lang = st.session_state.language
    
    language, use_gpt, api_key = render_sidebar(current_lang)
    st.session_state.language = language
    
    controller = CVEvaluationController()
    
    render_header(language)
    cv_file, job_description = render_input_section(language)
    
    if st.button(get_text("evaluate_btn", language), type="primary", use_container_width=True):
        if cv_file is None:
            st.error(get_text("error_no_cv", language))
        elif not job_description.strip():
            st.error(get_text("error_no_job", language))
        elif use_gpt and not api_key:
            st.error(get_text("error_no_api", language))
        else:
            with st.spinner(get_text("analyzing", language)):
                try:
                    result = controller.process_evaluation(
                        cv_file, job_description, language, api_key, use_gpt
                    )
                    render_results(result, language)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("*Powered by Sentence Transformers & GPT-4*")

if __name__ == "__main__":
    main()
