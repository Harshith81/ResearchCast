import sys
import streamlit as st
sys.path.append(".")

try:
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a')
    st.success("✅ Kokoro loaded successfully.")
except Exception as e:
    st.error(f"❌ Kokoro error: {e}")
