import streamlit as st
from llama_cpp import Llama
import json

# --- 1. SET UP STREAMLIT DASHBOARD PAGE ---
st.set_page_config(page_title="GGUF API Server", page_icon="🤖")
st.title("🤖 Live GGUF API Engine")
st.caption("Running seamlessly on Streamlit Community Cloud")

# --- 2. SINGLETON PATTERN TO CACHE THE MODEL ---
# This ensures the multi-gigabyte model loads ONCE into RAM and doesn't reload on every request
@st.cache_resource
def load_gguf_model():
    repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    
    # from_pretrained handles the live URL loading natively without local storage
    return Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=512,         # Small context window keeps memory use way below 2.7GB
        n_gpu_layers=0,    # CPU processing only (Streamlit Free doesn't have GPUs)
        verbose=False
    )

try:
    with st.spinner("Loading GGUF model into cloud RAM..."):
        llm = load_gguf_model()
    st.success("✅ Engine Online & Model Loaded!")
except Exception as e:
    st.error(f"❌ Initialization Failed: {e}")
    st.stop()

# --- 3. THE API ENTRANCE TRICK ---
# Streamlit lets us catch query parameters from the URL string.
# Example: https://share.streamlit.io/user/repo/main/app?prompt=hello
query_params = st.query_params

if "prompt" in query_params:
    # If a prompt is detected, clear the page UI and return raw JSON data instead
    user_prompt = query_params["prompt"]
    max_tokens = int(query_params.get("max_tokens", 100))
    
    # Run text generation
    response = llm(f"Q: {user_prompt} A:", max_tokens=max_tokens, stop=["Q:", "\n"])
    result_text = response["choices"][0]["text"].strip()
    
    # Output perfectly formatted JSON directly to the browser window
    st.json({"text": result_text})
    st.stop()  # Halt execution immediately so the default UI elements don't print

# --- 4. BACKEND MONITOR (What you see when visiting manually) ---
st.write("---")
st.markdown("### How to query this API from Vercel/Frontend:")
st.markdown("Send a standard HTTP GET request to this URL with a `prompt` query parameter.")

# Display a dynamic sample link based on your active server URL
current_url = "https://your-streamlit-app-url.streamlit.app"
st.code(f"{current_url}/?prompt=What+is+the+capital+of+Japan%3F", language="text")

st.markdown("#### Expected JSON Output:")
st.code(json.dumps({"text": "Tokyo"}, indent=2), language="json")
