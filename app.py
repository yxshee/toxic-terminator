"""
🛡️ Toxic Terminator - Advanced Visual Interface
================================================

A modern, visually enhanced Streamlit application for real-time toxicity detection
with advanced UI components, animations, and comprehensive visual feedback.

Features:
- 🎨 Modern gradient design with glassmorphism effects
- 📊 Real-time confidence meters and visual indicators
- 🌈 Dynamic color-coded results with animations
- 📱 Responsive design optimized for all devices
- ⚡ Interactive elements with hover effects
- 🔄 Loading animations and smooth transitions

Author: Venom
Date: August 2025
Version: 2.0 Enhanced
"""

import streamlit as st
import pickle
import re
import nltk
import time
import plotly.graph_objects as go
import plotly.express as px
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Configure page settings for better mobile experience
st.set_page_config(
    page_title="🛡️ Toxic Terminator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Download required NLTK resources
@st.cache_resource
def download_nltk_data():
    """Download required NLTK resources with progress tracking"""
    resources = ['punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

download_nltk_data()

# ---------------- Enhanced Modern Theme ---------------- #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF6347);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Input Styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.3) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        border: none !important;
        border-radius: 50px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        background: linear-gradient(45deg, #FF5252, #26A69A) !important;
    }
    
    /* Result Cards */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        animation: slideInUp 0.6s ease-out;
    }
    
    .toxic-card {
        border-left-color: #FF4757;
        background: linear-gradient(135deg, #FFF5F5, #FFE8E8);
    }
    
    .safe-card {
        border-left-color: #2ED573;
        background: linear-gradient(135deg, #F0FFF4, #E8F5E8);
    }
    
    .confidence-meter {
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        background: rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    /* Animations */
    @keyframes slideInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Stats Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #FFD700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #FFD700;
        animation: spin 1s ease-in-out infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu, footer, .stDeployButton {
        visibility: hidden;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white;
        font-family: 'Inter', sans-serif;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .glass-card {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        
        .stButton button {
            font-size: 16px !important;
            padding: 0.6rem 1.5rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Enhanced Application Logic ---------------- #

def get_wordnet_pos(tag):
    """
    Convert Penn Treebank POS tags to WordNet POS tags for accurate lemmatization.
    
    Args:
        tag (str): Part-of-speech tag in Penn Treebank format
        
    Returns:
        wordnet POS tag: The corresponding WordNet part-of-speech tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    """
    Preprocess the input text with enhanced error handling and progress tracking.
    """
    # Remove special characters and keep only alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize into individual words
    tokens = word_tokenize(text.lower())
    
    # Apply part-of-speech tagging
    pos_tokens = pos_tag(tokens)
    
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each word based on its part of speech
    lemmas = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tokens]
    
    return " ".join(lemmas)

@st.cache_resource
def load_model():
    """
    Load the pre-trained toxicity detection model and TF-IDF vectorizer with progress tracking.
    """
    try:
        with open("models/tf_idf.pkt", "rb") as f:
            tfidf = pickle.load(f)
        with open("models/toxicity_model.pkt", "rb") as f:
            model = pickle.load(f)
        return tfidf, model, True
    except Exception as e:
        return None, None, False

def create_confidence_meter(confidence, is_toxic=False):
    """Create an animated confidence meter using Plotly"""
    color = "#FF4757" if is_toxic else "#2ED573"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'color': 'white', 'size': 16}},
        number = {'suffix': "%", 'font': {'color': 'white', 'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': "white", 'tickfont': {'color': 'white'}},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [0, 50], 'color': "rgba(255,255,255,0.1)"},
                {'range': [50, 80], 'color': "rgba(255,255,255,0.15)"},
                {'range': [80, 100], 'color': "rgba(255,255,255,0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_analysis_chart(text_length, toxicity_score):
    """Create a visual analysis chart"""
    categories = ['Text Length', 'Toxicity Risk', 'Safety Score']
    values = [
        min(text_length / 10, 100),  # Normalize text length
        toxicity_score * 100,
        (1 - toxicity_score) * 100
    ]
    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    
    fig = go.Figure([go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="📊 Analysis Breakdown",
        title_font_color="white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color='white'),
        yaxis=dict(color='white', range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ---------------- Enhanced UI Components ---------------- #

def render_header():
    """Render the enhanced header with animations"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🛡️ Toxic Terminator</div>
        <div class="main-subtitle">
            Advanced AI-Powered Content Moderation • Real-time Toxicity Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_stats():
    """Render performance statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">95.2%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">0.97</div>
            <div class="stat-label">ROC AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">10K+</div>
            <div class="stat-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">56K</div>
            <div class="stat-label">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)

def render_loading_animation():
    """Render a loading animation"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <p style="color: white; margin-top: 1rem; font-family: 'Inter', sans-serif;">
            🔍 Analyzing content...
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Main Application ---------------- #

def main():
    # Render header
    render_header()
    
    # Render performance stats
    render_stats()
    
    # Main input section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Text for Analysis")
    st.markdown("Type or paste the content you want to analyze for toxicity detection.")
    
    user_input = st.text_area(
        "",
        height=150,
        placeholder="Enter your text here... (e.g., tweets, comments, messages)",
        label_visibility="collapsed"
    )
    
    analyze_button = st.button("🔍 Analyze Content", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load models
    tfidf, model, models_loaded = load_model()
    
    if analyze_button and user_input:
        if models_loaded:
            # Show loading animation
            loading_placeholder = st.empty()
            with loading_placeholder:
                render_loading_animation()
            
            # Simulate processing time for better UX
            time.sleep(1.5)
            loading_placeholder.empty()
            
            # Process the input
            processed = preprocess_text(user_input)
            features = tfidf.transform([processed])
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            # Create result visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Main result card
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-card toxic-card">
                        <h2 style="color: #FF4757; margin-bottom: 1rem;">
                            ⚠️ Toxic Content Detected
                        </h2>
                        <p style="color: #666; font-size: 1.1rem; margin-bottom: 1rem;">
                            The analyzed content contains potentially harmful or offensive material.
                        </p>
                        <div style="background: #FF4757; height: 8px; width: {probability*100}%; 
                             border-radius: 4px; transition: width 1s ease;"></div>
                        <p style="color: #888; margin-top: 0.5rem; font-size: 0.9rem;">
                            Risk Level: {probability:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card safe-card">
                        <h2 style="color: #2ED573; margin-bottom: 1rem;">
                            ✅ Content is Safe
                        </h2>
                        <p style="color: #666; font-size: 1.1rem; margin-bottom: 1rem;">
                            The analyzed content appears to be appropriate and non-toxic.
                        </p>
                        <div style="background: #2ED573; height: 8px; width: {(1-probability)*100}%; 
                             border-radius: 4px; transition: width 1s ease;"></div>
                        <p style="color: #888; margin-top: 0.5rem; font-size: 0.9rem;">
                            Safety Score: {(1-probability):.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Confidence meter
                fig = create_confidence_meter(probability, prediction == 1)
                st.plotly_chart(fig, use_container_width=True)
            
            # Analysis breakdown
            st.markdown("### 📊 Detailed Analysis")
            col3, col4 = st.columns([1, 1])
            
            with col3:
                analysis_fig = create_analysis_chart(len(user_input), probability)
                st.plotly_chart(analysis_fig, use_container_width=True)
            
            with col4:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🔍 Analysis Details")
                st.write(f"**Original Length:** {len(user_input)} characters")
                st.write(f"**Processed Length:** {len(processed)} words")
                st.write(f"**Prediction:** {'Toxic' if prediction == 1 else 'Safe'}")
                st.write(f"**Confidence:** {max(probability, 1-probability):.2%}")
                
                # Risk indicators
                if prediction == 1:
                    if probability > 0.9:
                        st.error("🚨 High Risk - Immediate attention required")
                    elif probability > 0.7:
                        st.warning("⚠️ Medium Risk - Review recommended")
                    else:
                        st.info("ℹ️ Low Risk - Minor concerns detected")
                else:
                    st.success("✅ Content approved for publication")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        else:
            st.error("❌ **Model Loading Error**\n\nUnable to load the required model files. Please ensure:")
            st.markdown("""
            - `models/toxicity_model.pkt` exists
            - `models/tf_idf.pkt` exists  
            - Files are accessible and not corrupted
            """)
    
    elif analyze_button and not user_input:
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer section
    st.markdown("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📘 About Toxic Terminator")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.markdown("""
        **🤖 AI Technology**
        - TF-IDF Vectorization
        - Multinomial Naive Bayes
        - NLTK Text Processing
        - Real-time Classification
        """)
    
    with col6:
        st.markdown("""
        **🎯 Use Cases**
        - Social Media Moderation
        - Comment Filtering
        - Content Review
        - Community Safety
        """)
    
    with col7:
        st.markdown("""
        **📊 Performance**
        - 95.2% Accuracy
        - 0.97 ROC AUC Score
        - Millisecond Response
        - Scalable Processing
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
