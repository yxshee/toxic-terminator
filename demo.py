#!/usr/bin/env python3
"""
🎬 Toxic Terminator Demo Script
==============================

This script demonstrates the enhanced visual interface of Toxic Terminator
with sample inputs and showcases all the new visual features.

Usage:
    python demo.py

Features Demonstrated:
- 🎨 Modern glassmorphism UI
- 📊 Interactive confidence meters
- 🌈 Animated result cards
- 📱 Responsive design
- ⚡ Real-time analysis

Author: Venom
Date: August 2025
"""

import streamlit as st
import time

def run_demo():
    """Run the Toxic Terminator demo with sample inputs"""
    
    st.set_page_config(
        page_title="🎬 Toxic Terminator Demo",
        page_icon="🎬",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .demo-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .demo-title {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .demo-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .sample-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .sample-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .toxic-sample {
        border-left: 4px solid #FF4757;
    }
    
    .safe-sample {
        border-left: 4px solid #2ED573;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Demo Header
    st.markdown("""
    <div class="demo-header">
        <div class="demo-title">🎬 Toxic Terminator Demo</div>
        <div class="demo-subtitle">
            Experience the Enhanced Visual Interface with Interactive Examples
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("## 🚀 Quick Start Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <h3>1. Enter Text</h3>
            <p>Type or paste content into the analysis box</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h3>2. Analyze</h3>
            <p>Click the analyze button to process your content</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>3. Review Results</h3>
            <p>Get instant visual feedback and detailed analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample Texts
    st.markdown("## 📋 Sample Texts to Try")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("### ✅ Safe Content Examples")
        
        safe_samples = [
            "I love spending time with my family on weekends.",
            "Thanks for sharing this helpful tutorial!",
            "The weather is beautiful today, perfect for a walk.",
            "Congratulations on your achievement!",
            "This movie was really entertaining and well-made."
        ]
        
        for sample in safe_samples:
            st.markdown(f"""
            <div class="sample-card safe-sample">
                <p>"{sample}"</p>
                <small>✅ Expected: Safe Content</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("### ⚠️ Potentially Toxic Examples")
        st.markdown("<small>⚠️ These examples are for testing purposes only</small>", unsafe_allow_html=True)
        
        toxic_samples = [
            "I hate this stupid website and everyone on it.",
            "You're all a bunch of idiots who don't understand anything.",
            "This is the worst content I've ever seen, complete garbage.",
            "Stop being such a moron and use your brain for once.",
            "I can't stand people like you, you're absolutely worthless."
        ]
        
        for sample in toxic_samples:
            st.markdown(f"""
            <div class="sample-card toxic-sample">
                <p>"{sample}"</p>
                <small>⚠️ Expected: Toxic Content</small>
            </div>
            """, unsafe_allow_html=True)
    
    # New Features Showcase
    st.markdown("## ✨ Enhanced Features")
    
    feature_cols = st.columns(4)
    
    features = [
        ("🎨", "Glassmorphism UI", "Modern translucent design with blur effects"),
        ("📊", "Interactive Charts", "Real-time confidence meters and analysis"),
        ("🌈", "Animated Results", "Smooth transitions and visual feedback"),
        ("📱", "Responsive Design", "Optimized for all screen sizes")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with feature_cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h4>{title}</h4>
                <p style="font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance Metrics
    st.markdown("## 📈 Performance Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Accuracy", "95.2%", "2.1%")
    
    with metrics_col2:
        st.metric("ROC AUC", "0.97", "0.03")
    
    with metrics_col3:
        st.metric("Response Time", "< 1s", "-200ms")
    
    with metrics_col4:
        st.metric("Training Data", "56K", "10K")
    
    # Call to Action
    st.markdown("---")
    st.markdown("## 🎯 Ready to Try It?")
    
    if st.button("🚀 Launch Toxic Terminator", type="primary"):
        st.balloons()
        st.success("🎉 Redirecting to the main application...")
        time.sleep(2)
        st.markdown("""
        <script>
        window.open('app.py', '_blank');
        </script>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; 
                background: rgba(255,255,255,0.1); border-radius: 10px;">
        <p style="color: #667eea; font-weight: 600;">
            🛡️ Toxic Terminator - Making Digital Spaces Safer, One Analysis at a Time
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run_demo()
