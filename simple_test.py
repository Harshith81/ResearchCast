import sys
sys.path.append(".")

try:
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a')
    success("✅ Kokoro loaded successfully.")
except Exception as e:
    error(f"❌ Kokoro error: {e}")
