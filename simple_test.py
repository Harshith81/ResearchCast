import sys
sys.path.append(".")

try:
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a')
    print("✅ Kokoro loaded successfully.")
except Exception as e:
    print(f"❌ Kokoro error: {e}")
