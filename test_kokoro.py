import sys
import subprocess
import streamlit as st

# Debug information
st.write("Python Path:", sys.path)
result = subprocess.run(["pip", "list"], capture_output=True, text=True)
st.write("Installed packages:")
st.code(result.stdout)

# Try importing and show error details
try:
    import kokoro
    st.success("Kokoro imported successfully!")
except ImportError as e:
    st.error(f"Failed to import kokoro: {str(e)}")
    
# Try the other possible import path
try:
    import kokoro_tts
    st.success("kokoro_tts imported successfully!")
except ImportError as e:
    st.error(f"Failed to import kokoro_tts: {str(e)}")
