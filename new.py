import streamlit as st     
from dotenv import load_dotenv      
load_dotenv()    

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate    
from langchain_community.vectorstores import FAISS        
import google.generativeai as genai  
   
import re
import os
from gtts import gTTS
import tempfile
import requests
from bs4 import BeautifulSoup
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log
from functools import lru_cache
import logging   
import pickle
import time,math
from pathlib import Path    
import sys
import warnings
# Add new imports for enhanced URL handling
from newspaper import Article
import trafilatura
import urllib.parse
import soundfile as sf

# For key term extraction     
from keybert import KeyBERT
import yake
from rake_nltk import Rake   

import nltk
nltk.data.path.append('./nltk_data')
resources = ['punkt', 'stopwords', 'wordnet']
for res in resources:
    try:
        if res == 'punkt':
            nltk.data.find(f'tokenizers/{res}')
        else:
            nltk.data.find(f'corpora/{res}')
    except LookupError:
        try:
            nltk.download(res, download_dir='./nltk_data', quiet=True)
        except Exception as e:
            st.error(f"Error downloading {res}: {str(e)}")
            st.stop()
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import numpy as np
import random

import urllib.parse
from collections import Counter
import xml.etree.ElementTree as ET  

# For Google Scholar integration
# from serpapi.google_scholar_search import GoogleScholarSearch 
#from serpapi.google_search_results import GoogleSearch
#from serpapi import GoogleSearch



# import scholarly
from scholarly import ProxyGenerator, scholarly

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import string

import subprocess

# Define the path where ffmpeg will be stored
FFMPEG_PATH = os.path.join(os.getcwd(), "ffmpeg")

# Download ffmpeg only if it's not already present
if not os.path.exists(FFMPEG_PATH):
    print("Downloading FFmpeg...")
    subprocess.run(
        "wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && "
        "tar -xf ffmpeg-release-amd64-static.tar.xz && "
        "mv ffmpeg-*-static/ffmpeg . && "
        "rm -rf ffmpeg-*-static ffmpeg-release-amd64-static.tar.xz",
        shell=True,
        check=True,
    )

# Set ffmpeg path for pydub
from pydub import AudioSegment
AudioSegment.converter = os.path.join(os.getcwd(), "ffmpeg")


warnings.filterwarnings("ignore", category=UserWarning)   
sys.setrecursionlimit(10000)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Set the Streamlit page config.
st.set_page_config(page_title="ResearchCast 🎙️", layout="wide") 

# Add custom headers for better web scraping
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0'
}

# Configure the Gemini API with your API key.
def initialize_apis():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key not found. Please check your .env file.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure Google API: {str(e)}")
        return False  

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

@st.cache_data(persist="disk", show_spinner=False, hash_funcs={PdfReader: lambda _: None}, ttl=3600)
def get_pdf_text(pdf_files):
        text = ""
        total_files = len(pdf_files)
        progress_bar = st.progress(0)
        for i, pdf in enumerate(pdf_files, 1):
            try:
                pdf_reader = PdfReader(pdf)
                total_pages = len(pdf_reader.pages)
                for j, page in enumerate(pdf_reader.pages, 1):
                    text += page.extract_text() or ""
                    progress_bar.progress((j / total_pages) * (i / total_files))
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
        progress_bar.empty()
        if not text.strip():
            st.error("No text could be extracted from the PDF.")
            return None
        return text

@retry(wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING))
def get_text_from_url(url):
    try:
        # Normalize URL
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc.lower()

        if "arxiv.org" in domain:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            abstract = soup.find("blockquote", class_="abstract mathjax")
            sections = soup.find_all("div", class_="mathjax")
            content = "\n\n".join(section.get_text(strip=True) for section in sections)
            if abstract or content:
                return (abstract.get_text(strip=True).replace("Abstract: ", "") if abstract else "") + "\n\n" + content
            else:
                st.error("Could not extract content from the provided arXiv URL.")  
                return None

        elif "pubmed.ncbi.nlm.nih.gov" in domain:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            abstract = soup.find("div", class_="abstract-content selected")
            sections = soup.find_all("div", class_="section")
            content = "\n\n".join(section.get_text(strip=True) for section in sections)
            if abstract or content:
                return (abstract.get_text(strip=True) if abstract else "") + "\n\n" + content
            else:
                st.error("Could not extract content from the provided PubMed URL.")
                return None

        elif "journals.plos.org" in domain:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            abstract = soup.find("div", class_="abstract")
            if abstract:
                return abstract.get_text(strip=True)
            else:
                st.error("Could not extract text from the provided PLOS URL.")
                return None

        elif "doi.org" in domain:
            doi = url.split("doi.org/")[-1]
            api_url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(api_url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                data = response.json()
                abstract = data.get("message", {}).get("abstract", None)
                title = data.get("message", {}).get("title", ["No title available"])[0]
                if abstract:
                    abstract = re.sub(r"<[^>]*>", "", abstract)
                    return f"Title: {title}\n\nAbstract: {abstract}"
                else:
                    return f"Title: {title}\n\nNo abstract available for this DOI."
            else:
                st.error("Failed to fetch data from CrossRef API.")
                return None

        elif "semanticscholar.org" in domain:
            paper_id = url.split("/")[-1]
            headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
            response = requests.get(f"https://api.semanticscholar.org/v1/paper/{paper_id}", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                abstract = data.get("abstract", "No abstract available.")
                title = data.get("title", "No title available.")
                content = data.get("fieldsOfStudy", [])
                return f"Title: {title}\n\nAbstract: {abstract}\n\nFields of Study: {', '.join(content)}"
            else:
                st.error("Failed to fetch data from Semantic Scholar.")
                return None

        elif "doaj.org" in domain:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            abstract = soup.find("meta", {"name": "description"})
            if abstract:
                return abstract["content"]
            else:
                st.error("Could not extract text from the provided DOAJ URL.")
                return None

        elif "researchgate.net" in domain:
            # Enhanced ResearchGate extraction
            try:
                # Try newspaper3k first
                article = Article(url)
                article.download()
                article.parse()
                if article.text and len(article.text.split()) > 50:   
                    return article.text

                # Fallback to direct HTML parsing
                response = requests.get(url, headers=HEADERS, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Try multiple selectors  
                content_selectors = [
                    "div.research-detail-middle",
                    "div.publication-detail-author-info",
                    "div.publication-abstract",
                    "div.research-detail-header-section",
                    "div[itemprop='description']",
                    "div.nova-e-text"
                ]
                
                for selector in content_selectors:
                    content = soup.select_one(selector)
                    if content:
                        text = content.get_text(strip=True)
                        if len(text.split()) > 50:  # Basic validation
                            return text

                # Final fallback to trafilatura
                extracted = trafilatura.extract(response.text)
                if extracted:
                    return extracted

                st.error("Could not extract text from the provided ResearchGate URL.")
                return None

            except Exception as e:
                logger.error(f"ResearchGate extraction error: {str(e)}")
                st.error("Failed to extract content from ResearchGate.")
                return None

        else:
            # Handle general articles (news, blogs, etc.)
            try:
                # Try newspaper3k first
                article = Article(url)
                article.download()
                article.parse()
                
                if article.text and len(article.text.split()) > 50:
                    return article.text
                
                # Fallback to trafilatura
                response = requests.get(url, headers=HEADERS, timeout=10)
                extracted = trafilatura.extract(response.text)
                
                if extracted:
                    return extracted
                
                # Final fallback to basic HTML parsing
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove unwanted elements
                for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Try to find main content
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post'))
                
                if main_content:
                    return main_content.get_text(strip=True)
                
                st.error("Could not extract text from the provided URL.")
                return None

            except Exception as e:
                logger.error(f"General article extraction error: {str(e)}")
                st.error("Failed to extract content from the URL.")
                return None

    except Exception as e:
        st.error(f"Error fetching text from URL: {str(e)}")
        return None

    time.sleep(4)

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, # 900 may be very constraintful
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_text(text)

@st.cache_resource
def load_or_create_embeddings(text_chunks, cache_file="embeddings.pkl"):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                index, cached_texts = pickle.load(f)
            if cached_texts == text_chunks:
                return index, cached_texts
            else:
                st.warning("Cache mismatch! Regenerating embeddings...")
        except Exception as e:
            st.error(f"Error loading cache: {str(e)}. Regenerating embeddings.")
    # from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_texts(text_chunks, embedding=embedding_model)
    with open(cache_file, "wb") as f:
        pickle.dump((index, text_chunks), f)
    return index, text_chunks

@lru_cache(maxsize=100)
def generate_conversation_script(text):
    try:
        if not text:
            raise ValueError("No text available for conversation generation.")
        logger.info("Generating conversation using Gemini API")  
        
        # change the values in conversation structure like 9-14 etc to somewhat lower values if the model is not able to generate as needed and in a complete fashion
        
        prompt = """
        Create an engaging podcast conversation between a Host (H) and a Guest Expert (G) based on this research paper.
        
        Conversation Structure:
        1. Brief introduction (4-5 exchanges)
        2. Main topic discussion (11-15 exchanges)    
        3. Key findings and implications (9-10 exchanges)
        4. Conclusion (4-5 exchanges)
        This conversation should be long enough to produce approximately 10-15 minutes of audio.


        Guidelines:
        - Keep each exchange conversational and informative
        - Use clear, accessible language
        - Highlight key research findings
        - Maintain a natural dialogue flow
        - Natural back-and-forth dialogue flow
        - Real-world examples and applications
        - Clear explanations of complex concepts
        - Follow-up Questions and clarifications

        Format:
        H: [Host's dialogue]
        G: [Guest's dialogue]

        Research content: {text}
        """
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_output_tokens": 2048  # 1024, 2048, 4096(too large trasncript but highly unstable because of our model capacity)    
            
            # for testing i am using 2048 change this back to 1024 if it raises any quota or 429 issues or errors
        }
        # Use Gemini model via google.generativeai
        model = genai.GenerativeModel("gemini-1.5-pro-latest", generation_config=generation_config)
        
        # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)

        response = model.generate_content(prompt.format(text=text[:4000]))  # change this back to 3000 or 3500 if the model is not able to generate the script properly or if it raises any quota or 429 issues
        conversation = response.text
        conversation = re.sub(r'Host:', 'H:', conversation)  
        conversation = re.sub(r'Guest:', 'G:', conversation)
        return conversation
    except Exception as e:
        st.error(f"Error generating conversation: {str(e)}")   
        return None
    
### -------------
### Voice and Audio Settings
### -------------
    
def get_background_tracks():
    tracks = [{'name': 'None', 'path': None}] 
    audio_dir = Path("assets/audio")
    if audio_dir.exists():
        for file in audio_dir.glob("*.mp3"):
            name = file.stem.replace("_", " ").title()
            tracks.append({'name': name, 'path': str(file)})
    
    return tracks


def generate_audio(script, voice_settings=None, background_settings=None):
    if not script:
        st.error("Script is empty. Cannot generate audio.")   
        return None, None

    if voice_settings is None:
        voice_settings = {
            'host': {'voice_id': 'alice'},
            'guest': {'voice_id': 'alloy'}  
        }
    
    if background_settings is None:
        background_settings = {  
            'enabled': False,
            'track': None,
            'volume': 0.1
        }
    
    try:
        temp_dir = Path(tempfile.mkdtemp())
        segments = [seg.strip() for seg in script.split("\n") if seg.strip()]
        final_audio = AudioSegment.empty()
        timestamps = []
        current_time = 0
        total_segments = len(segments)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Group segments by speaker to reduce API calls and improve performance
        grouped_segments = []
        current_speaker = None
        current_text = ""
        current_speaker_segments = []
        
        for segment in segments:
            # Default to Host if no speaker marker
            if not (segment.startswith("H:") or segment.startswith("G:")):
                segment = "H: " + segment
            
            speaker = "host" if segment.startswith("H:") else "guest"
            text = segment[2:].strip()
            
            if speaker != current_speaker and current_speaker is not None:
                grouped_segments.append({
                    'speaker': current_speaker,
                    'text': current_text,
                    'original': current_speaker_segments
                })
                current_text = text
                current_speaker = speaker
                current_speaker_segments = [segment]
            else:
                # First segment or continuing with same speaker
                current_speaker = speaker
                if not current_text:
                    current_text = text
                    current_speaker_segments = [segment]
                else:
                    current_text += ". " + text
                    current_speaker_segments.append(segment)
        
        if current_speaker is not None:
            grouped_segments.append({
                'speaker': current_speaker,
                'text': current_text,
                'original': current_speaker_segments
            })
        
        # Initialize Kokoro TTS client
        kokoro_client = KokoroTTS()
        
        for i, group in enumerate(grouped_segments):
            voice = voice_settings[group['speaker']]
            temp_file = temp_dir / f"group_{i}.mp3"
            
            # Try with Kokoro TTS first
            try:
                # Generate speech with Kokoro
                kokoro_client.tts(
                    text=group['text'],
                    voice_id=voice['voice_id'],
                    rate=voice.get('rate', 1.0),
                    output_path=str(temp_file)
                )
                
                # Check if the file was created successfully
                if not temp_file.exists() or temp_file.stat().st_size == 0:
                    raise Exception("Failed to generate audio with Kokoro TTS")
                    
            except Exception as e:
                st.warning(f"Could not generate audio with Kokoro TTS: {str(e)}. Trying fallback...")
                try:   
                    # Fallback to Google TTS
                    time.sleep(2)  # Add delay to avoid rate limiting
                    tts = gTTS(
                        text=group['text'],    
                        lang='en',
                        tld='com',
                        slow=False
                    )
                    tts.save(str(temp_file))
                    
                    if not temp_file.exists() or temp_file.stat().st_size == 0:
                        raise Exception("gTTS fallback also failed")
                except Exception as fallback_e:
                    st.error(f"All TTS attempts failed: {str(fallback_e)}")
                    return None, None
            
            group_audio = AudioSegment.from_mp3(str(temp_file))
            
            total_words = len(group['text'].split())
            audio_duration = len(group_audio)
            
            segment_durations = []
            segment_start_times = [0]
            running_total = 0   
            
            for j, segment in enumerate(group['original']):
                segment_text = segment[2:].strip()
                segment_words = len(segment_text.split())
                segment_duration = int((segment_words / total_words) * audio_duration)
                segment_durations.append(segment_duration)
                running_total += segment_duration
                if j < len(group['original']) - 1:
                    segment_start_times.append(running_total)
            
            final_audio += group_audio
            for j, segment in enumerate(group['original']):
                start_time = current_time + segment_start_times[j]
                end_time = start_time + segment_durations[j]
                timestamps.append({
                    'start': start_time / 1000,
                    'end': end_time / 1000,
                    'text': segment
                })
            
            current_time += len(group_audio)
            temp_file.unlink()
            progress_bar.progress((i+1) / len(grouped_segments))
            time.sleep(0.1)  # Small delay to avoid UI freezing

        # Add background music if enabled
        if background_settings.get('enabled', False) and background_settings.get('track'):
            try:
                bg_track = AudioSegment.from_file(background_settings['track'])
                speech_duration = len(final_audio)
                bg_duration = len(bg_track)
                if bg_duration > 0:
                    repeats_needed = math.ceil(speech_duration / bg_duration)
                    bg_loop = bg_track * repeats_needed
                    bg_loop = bg_loop[:speech_duration]
                    
                    bg_loop = bg_loop - 20
                    speech_audio = final_audio + 4  
                    final_audio = speech_audio.overlay(bg_loop)
                else:
                    st.warning("Background track is empty or invalid")
                    
            except Exception as e:
                st.error(f"Error adding background music: {str(e)}")
                logger.error(f"Background music error: {str(e)}")
                
        progress_bar.empty()  
        status_text.empty()
        output_file = temp_dir / "podcast.mp3"  
        final_audio.export(str(output_file), format="mp3", bitrate="192k")
        
        return str(output_file), timestamps
            
    except Exception as e:
        st.error(f"❌ Error generating audio: {str(e)}")
        logger.error(f"Audio generation error: {str(e)}")
        return None, None

def setup_audio_settings():
    if "voice_settings" not in st.session_state:
        st.session_state.voice_settings = {
            'host': {'voice_id': 'alice','name': 'Alice (Host)'},
            'guest': {'voice_id': 'alloy','name': 'Alloy (Guest)'}
        }   
    
    if "background_settings" not in st.session_state:
        st.session_state.background_settings = {
            'enabled': False,
            'track': None,
            'track_name': 'None',
            'volume': 0.12,
            'auto_select': True
        }
    
    # Define updated Kokoro TTS voice options for KPipeline 
    host_voice_options = [
        {'name': 'Alice (Br-Female)', 'voice_id': 'alice'},
        {'name': 'Daniel (Br-Male)', 'voice_id': 'daniel'},
        {'name': 'Isabella (Br-Female)', 'voice_id': 'isabella'},
        {'name': 'Fable (Br-Male)', 'voice_id': 'fable'},
        {'name': 'Lily (Br-Female)', 'voice_id': 'lily'},
        {'name': 'George (Br-Male)', 'voice_id': 'george'},
        
    ]
    
    guest_voice_options = [
        {'name': 'Alloy (Am-Female)', 'voice_id': 'alloy'},
        {'name': 'Echo (Am-Male)', 'voice_id': 'echo'},
        {'name': 'Heart (Am-Female)', 'voice_id': 'heart'},
        {'name': 'Liam (Am-Male)', 'voice_id': 'liam'},
        {'name': 'Sarah (Am-Female)', 'voice_id': 'sarah'},
        {'name': 'Puck (Am-Male)', 'voice_id': 'puck'},
    ]
    
    with st.expander("🔊 Voice & Background Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Host Voice")
            # Find the current voice index in host voices
            host_voice_index = next((
                i for i, v in enumerate(host_voice_options) 
                if v['voice_id'] == st.session_state.voice_settings['host']['voice_id']
            ), 0)
            
            # Create a select box for host voice
            selected_host_voice = st.selectbox(
                "Select voice for the host",
                options=range(len(host_voice_options)),
                format_func=lambda i: host_voice_options[i]['name'],
                index=host_voice_index,
                key="host_voice_select"
            )
            st.session_state.voice_settings['host'] = {
                'voice_id': host_voice_options[selected_host_voice]['voice_id'],
                'name': host_voice_options[selected_host_voice]['name']
            }
        
        with col2:
            st.subheader("Guest Voice")
            # Find the current voice index in guest voices
            guest_voice_index = next((
                i for i, v in enumerate(guest_voice_options) 
                if v['voice_id'] == st.session_state.voice_settings['guest']['voice_id']
            ), 0)
            
            # Create a select box for guest voice
            selected_guest_voice = st.selectbox(
                "Select voice for the guest",
                options=range(len(guest_voice_options)),
                format_func=lambda i: guest_voice_options[i]['name'],
                index=guest_voice_index,
                key="guest_voice_select"
            )
            st.session_state.voice_settings['guest'] = {
                'voice_id': guest_voice_options[selected_guest_voice]['voice_id'],
                'name': guest_voice_options[selected_guest_voice]['name']
            }
        
        # Background music section
        st.subheader("Background Music")
        
        # Background music options
        bg_option = st.radio(
            "Background Music:",
            ["No Background Music", "Use Background Music"],
            index=1 if st.session_state.background_settings.get('enabled', False) else 0,
            key="bg_option"
        )
        
        if bg_option == "No Background Music":
            # Disable background music
            st.session_state.background_settings = {
                'enabled': False,
                'track': None,
                'track_name': 'None',
                'auto_select': False
            }
        else:  
            # Use Background Music
            st.info("A random background track will be automatically selected when generating the podcast audio.")
            
            # Save background settings with auto_select enabled
            st.session_state.background_settings = {
                'enabled': True,
                'track': None,  
                'track_name': 'Auto-selected',
                'volume': 0.03,
                'auto_select': True   
            }
    
    # Return current settings for use in generate_audio
    return {
        'voice_settings': st.session_state.voice_settings,
        'background_settings': st.session_state.background_settings
    }


class KokoroTTS:
    """
    Client for interacting with Kokoro TTS using the local KPipeline approach
    """
    def __init__(self):
        try:
            from kokoro import KPipeline
            self.pipeline_a = KPipeline(lang_code='a')  # American English
            self.pipeline_b = KPipeline(lang_code='b')  # British English
            self.initialized = True
            self.logger = logger  # Use your existing logger
        except ImportError as e:
            self.initialized = False
            logger.error(f"Could not import Kokoro package: {str(e)}")
        except Exception as e:
            self.initialized = False
            logger.error(f"Error initializing Kokoro: {str(e)}")
        
    def _get_pipeline_and_voice_id(self, voice_id):
        british_voices = ['alice','isabella','lily','daniel','fable','george']
        
        # Check if it's a British voice
        if voice_id in british_voices:
            # Use British pipeline with 'bf_' or 'bm_' prefix
            gender_prefix = 'bf_' if voice_id in ['alice','isabella','lily'] else 'bm_'
            return self.pipeline_b, f"{gender_prefix}{voice_id}"
        else:
            # Use American pipeline with 'af_' or 'am_' prefix
            gender_prefix = 'af_' if voice_id in ['alloy','heart','sarah'] else 'am_'
            return self.pipeline_a, f"{gender_prefix}{voice_id}"
    
    def tts(self, text, voice_id, style="narration", rate=1.0, output_path=None):
        if not self.initialized:
            raise Exception("Kokoro TTS not properly initialized")
        
        try:
            # Get appropriate pipeline and format voice ID
            pipeline, formatted_voice_id = self._get_pipeline_and_voice_id(voice_id)

            # Generate audio segments
            generator = pipeline(
                text, 
                voice=formatted_voice_id,
                speed=1.0,
            )
            
            # Collect and concatenate audio segments
            audio_segments = []
            for _, _, audio in generator:
                audio_segments.append(audio)
            
            if not audio_segments:
                raise Exception("No audio segments generated")

            final_audio = np.concatenate(audio_segments)
            sf.write(output_path, final_audio, 24000)
            return output_path
            
        except Exception as e:
            logger.error(f"Kokoro TTS error: {str(e)}")
            raise e


def check_audio_assets():
    assets_dir = Path("assets")
    audio_dir = assets_dir / "audio"

    if not assets_dir.exists():
        assets_dir.mkdir(exist_ok=True)   
    
    if not audio_dir.exists():
        audio_dir.mkdir(exist_ok=True)
        
    available_tracks = []
    for track in st.session_state.background_tracks:
        if track['path'] is not None:
            track_path = Path(track['path'])
            if track_path.exists():
                available_tracks.append(track['name'])

    if len(available_tracks) <= 1:
        st.warning("Background music tracks not found. Please add MP3 files to the 'assets/audio' directory.")
        
    return len(available_tracks) > 1


def select_random_background():
    random.seed(time.time())
    
    # Get available tracks (excluding 'None')
    available_tracks = [track for track in st.session_state.background_tracks if track['name'] != 'None' and track['path'] is not None]

    print(f"Available tracks: {[t['name'] for t in available_tracks]}")
    
    if not available_tracks:
        # No tracks available, return disabled settings
        st.warning("No background tracks available. Please add MP3 files to the assets/audio directory.")
        return {
            'enabled': False,
            'track': None,
            'track_name': 'None',
            'volume': 0.05
        }
    
    selected_track = random.choice(available_tracks)
    print(f"Selected track: {selected_track['name']}")
    
    track_path = Path(selected_track['path'])
    if not track_path.exists():
        # Track file doesn't exist, return disabled settings
        st.warning(f"Selected track file doesn't exist: {selected_track['path']}")
        return {
            'enabled': False,
            'track': None,
            'track_name': 'None',
            'volume': 0.05  
        }
    
    # Return complete background settings
    return {
        'enabled': True,
        'track': str(track_path.resolve()),  # Ensure absolute path
        'track_name': selected_track['name'],
        'volume': 0.03  # Fixed low volume
    }  
    
def check_kokoro_installation():
    """Check if Kokoro is installed and working properly"""
    try:
        from kokoro import KPipeline
        # Try to initialize with a simple test
        pipeline = KPipeline(lang_code='a')
        print("Kokoro TTS initialized successfully!")
        return True
    except ImportError:
        st.error("❌ Kokoro package is not installed. Voice generation will fall back to Google TTS.")
        st.info("To install Kokoro, run: pip install kokoro")
        return False
    except Exception as e:
        st.error(f"❌ Error initializing Kokoro: {str(e)}. Voice generation will fall back to Google TTS.")
        return False   

@st.cache_resource
def load_embeddings_model():
    # from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def initialize_qa_model():
    # We use Gemini for QA generation (if desired) or alternatively a simple chain using the Gemini API.
    return genai.GenerativeModel("gemini-1.5-pro-latest", generation_config={"temperature": 0.7, "top_p": 0.8, "max_output_tokens": 512})

@st.cache_resource 
def setup_qa_system(text_chunks):  
    try:
        logger.info("Setting up QA system...")
        if not text_chunks or len(text_chunks) == 0:
            raise ValueError("No text chunks available for embedding.")
        logger.info("Initializing embeddings...")
        embeddings = load_embeddings_model()
        logger.info("Creating vector store...")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        logger.info("QA system setup complete.")
        return vector_store
    except Exception as e:
        logger.error(f"Error setting up QA system: {str(e)}")
        st.error(f"Error setting up QA system: {str(e)}")
        return None
    
### -------------
### Q&A Section and Generate Summary 
### -------------

def generate_qa_answer(context, question):
    # Limit context to top 3 chunks to reduce response time
    top_context = "\n\n".join(context.split("\n\n")[:3])  
    prompt = f"""
    Using the following research paper context:
    {top_context}

    Answer the following question clearly in detail:
    {question}

    Final Answer:
    """
    # Use Gemini for QA answer generation.
    model = genai.GenerativeModel("gemini-1.5-pro-latest", generation_config={"temperature": 0.7, "top_p": 0.8, "max_output_tokens": 512})
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_summary(text):
    prompt = f"""
    Please provide a concise, structured, and insightful summary of the following research paper.
    
    The summary should include:
    - A brief overview of the topic.
    - Key findings and conclusions.
    - Significant implications and future directions.
    
    Summary:
    {text[:3000]}
    """
    model = genai.GenerativeModel("gemini-1.5-pro-latest", generation_config={"temperature": 0.7, "top_p": 0.8, "max_output_tokens": 512})
    response = model.generate_content(prompt)
    return response.text.strip()

### -------------
### Key Terms Extraction and finding Related papers 
### -------------

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re
from nltk.corpus import stopwords

def extract_research_keywords(text, num_keywords=10):
    # Load models (these will be cached after first run)
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small but effective model
    
    # Clean and preprocess text
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)  # Remove citations
    text = re.sub(r'References[\s\S]*$', '', text)    # Remove references section
    
    # Extract author keywords if available
    author_keywords = []
    keywords_pattern = re.search(r'(?i)(?:key ?words|key ?terms)[:;—–-]\s*(.*?)(?:\n\s*\n|\n\s*(?:introduction|abstract))', text, re.DOTALL)
    if keywords_pattern:
        keyword_text = keywords_pattern.group(1).strip()
        if ',' in keyword_text:
            author_keywords = [k.strip() for k in keyword_text.split(',')]
        elif ';' in keyword_text:
            author_keywords = [k.strip() for k in keyword_text.split(';')]
    
    # If we have enough author keywords, prioritize them
    if len(author_keywords) >= num_keywords:
        return author_keywords[:num_keywords]
    
    # Use NLP to extract noun phrases and named entities
    doc = nlp(text[:1000])  # Limit size for performance
    
    # Extract noun phrases and named entities as candidates
    candidates = []
    stop_words = set(stopwords.words('english'))
    domain_stopwords = {'study', 'research', 'paper', 'abstract', 'introduction', 'method', 'result'}
    stop_words.update(domain_stopwords)
    
    # Get noun phrases (noun chunks)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower()
        # Filter by length and stopwords
        words = phrase.split()
        if (2 <= len(words) <= 4 and 
            not all(word in stop_words for word in words) and
            not any(char.isdigit() for char in phrase)):
            candidates.append(phrase)
    
    # Add named entities
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
            candidates.append(ent.text.lower())
    
    # Add terms with specific POS patterns (ADJ + NOUN, etc.)
    pos_patterns = []
    for i in range(len(doc) - 1):
        if doc[i].pos_ == 'ADJ' and doc[i+1].pos_ == 'NOUN':
            term = (doc[i].text + ' ' + doc[i+1].text).lower()
            if not any(word in stop_words for word in term.split()):
                pos_patterns.append(term)
    
    # Add additional candidates from author keywords
    candidates.extend(author_keywords)
    
    # Remove duplicates and short candidates
    candidates = list(set([c for c in candidates if len(c) > 3]))
    
    if len(candidates) < num_keywords:
        # Not enough candidates, extract individual technical terms
        technical_terms = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                token.text.lower() not in stop_words and
                len(token.text) > 3):
                technical_terms.append(token.text.lower())
        candidates.extend(list(set(technical_terms)))
    
    # Still not enough candidates?
    if len(candidates) < num_keywords:
        return candidates  # Return what we have
    
    # Encode candidates to vectors
    candidate_embeddings = model.encode(candidates)
    
    # Cluster the embeddings
    num_clusters = min(num_keywords, len(candidates))
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(candidate_embeddings)
    cluster_centers = clustering_model.cluster_centers_
    
    # Find the closest candidates to each cluster center
    keywords = []
    for i in range(num_clusters):
        cluster_idx = np.where(clustering_model.labels_ == i)[0]
        if len(cluster_idx) > 0:
            # Find the candidate closest to cluster center
            closest_idx = cluster_idx[np.argmin(
                np.sum((candidate_embeddings[cluster_idx] - cluster_centers[i])**2, axis=1)
            )]
            keywords.append(candidates[closest_idx])
    
    # Prioritize author keywords if available
    final_keywords = []
    for keyword in author_keywords:
        if keyword not in final_keywords:
            final_keywords.append(keyword)
    
    # Add remaining unique keywords from clustering
    for keyword in keywords:
        if keyword not in final_keywords:
            final_keywords.append(keyword)
    
    return final_keywords[:num_keywords]

def find_related_papers(key_terms, num_papers=8):
    """Find related research papers using multiple methods with better handling"""
    if not key_terms or len(key_terms) == 0:
        return []
    
    error_indicators = ["insufficient", "unable", "error", "failed", "missing"]
    if isinstance(key_terms, list) and len(key_terms) == 1 and any(ind in key_terms[0].lower() for ind in error_indicators):
        logger.warning(f"Key term extraction failed with message: {key_terms[0]}")
        key_terms = ["research", "paper", "study", "analysis"]
        
    def prioritize_domain_terms(terms, domain=None):
        """Prioritize domain-specific terms in the search queries"""
        if not domain:
            domain_indicators = {
                'computer_science': ['algorithm', 'computation', 'programming', 'software', 'network', 'database', 'artificial intelligence', 'machine learning','deep learning','autonomous','cloud computing','data science','engineering'],
                'biology': ['cell', 'protein', 'gene', 'dna', 'rna', 'organism', 'species', 'enzyme','genomics','microbiome','bio technology','evolution','bio diversity'],
                'medicine': ['patient', 'treatment', 'clinical', 'disease', 'health', 'symptom', 'medical', 'therapy','healthcare','cancer','telemedicine','wellness','care'],
                'psychology': ['cognition', 'behavior', 'mental', 'perception', 'emotion', 'psychological', 'brain'],
                'physics': ['particle', 'quantum', 'relativity', 'energy', 'force', 'field', 'electron', 'velocity','mechanics','nuclear','motion'],
                'law': ['legal', 'court', 'justice', 'law', 'judge', 'judicial', 'statute', 'litigation', 'plaintiff', 'defendant', 'trial','attorney','prosecutor'],
                'economics': ['market', 'economic', 'inflation', 'price', 'monetary', 'finance', 'investment', 'banking','assests','capital','debt','stocks','mutual funds','shares','fixed deposists']
            }
            
            term_scores = {domain: 0 for domain in domain_indicators}
            for term in terms:
                term_lower = term.lower()
                for domain, indicators in domain_indicators.items():
                    for indicator in indicators:
                        if indicator in term_lower:
                            term_scores[domain] += 1
            
            max_domain = max(term_scores.items(), key=lambda x: x[1])
            if max_domain[1] > 0:
                domain = max_domain[0]
        
        return terms
    prioritized_terms = prioritize_domain_terms(key_terms)
    search_queries = [
        " ".join(prioritized_terms[:3]),  # Top 3 terms together
        f'"{prioritized_terms[0]}" {" ".join(prioritized_terms[1:3])}',  # First term in quotes with others
    ]
    
    papers = []
    for term in key_terms[:3]:
        if len(term.split()) > 1:  # Only use multi-word terms as individual queries
            search_queries.append(f'"{term}"')  # Use quotes for exact phrase
    
    # Try scholarly library (Google Scholar) with multiple backoff strategies
    for strategy in ['scholarly', 'crossref', 'arxiv']:
        if len(papers) >= num_papers:
            break
            
        if strategy == 'scholarly':
            try:
                logger.info("Trying scholarly (Google Scholar) search...")
                
                # Setup proxy rotation to avoid blocks
                pg = ProxyGenerator()
                success = pg.FreeProxies()
                if success:
                    scholarly.use_proxy(pg)
                else:
                    logger.warning("Failed to set up proxy for scholarly - may get blocked")
                
                # Try each query until we find enough papers
                for query in search_queries:
                    if len(papers) >= num_papers:
                        break
                        
                    try:
                        # Add random sleep to avoid detection
                        time.sleep(random.uniform(2.0, 4.0))
                        
                        # Search with timeout handling
                        search_results = scholarly.search_pubs(query)
                        result_count = 0
                        
                        for result in search_results:
                            if result_count >= num_papers // len(search_queries) + 1:
                                break
                                
                            try:
                                bib = result.get('bib', {})
                                # Only add if we have sufficient information
                                if bib.get('title'):
                                    paper = {
                                        'title': bib.get('title', 'Unknown Title'),
                                        'authors': bib.get('author', 'Unknown Author'),
                                        'year': bib.get('pub_year', 'Unknown Year'),
                                        'journal': bib.get('journal', 'N/A'),  # Fixed journal access
                                        'url': result.get('pub_url') or f"https://scholar.google.com/scholar?cluster={result.get('cluster_id')}" if result.get('cluster_id') else None,
                                        'citations': result.get('num_citations', 0),
                                        'venue': bib.get('venue', '')
                                    }
                                    # Only add if we don't have a duplicate
                                    if not any(p['title'].lower() == paper['title'].lower() for p in papers):
                                        papers.append(paper)
                                        result_count += 1
                            except Exception as e:
                                logger.warning(f"Error processing individual scholarly result: {str(e)}")
                                continue  
                                
                    except Exception as e:
                        logger.warning(f"Error with scholarly query '{query}': {str(e)}")
                
            except Exception as e:
                logger.error(f"Scholarly search failed completely: {str(e)}")
        
        # Try CrossRef API as backup
        elif strategy == 'crossref' and len(papers) < num_papers:
            try:
                logger.info("Trying CrossRef API for paper search...")
                for query in search_queries[:2]:  # Limit to just top queries
                    if len(papers) >= num_papers:
                        break
                        
                    encoded_query = urllib.parse.quote(query)
                    api_url = f"https://api.crossref.org/works?query={encoded_query}&rows=5"
                    
                    response = requests.get(api_url, headers=HEADERS, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('message', {}).get('items', [])
                        
                        for item in items:
                            if len(papers) >= num_papers:
                                break
                                
                            # Extract authors
                            authors = []
                            for author in item.get('author', []):
                                if 'given' in author and 'family' in author:
                                    authors.append(f"{author['given']} {author['family']}")
                                elif 'family' in author:
                                    authors.append(author['family'])
                            
                            author_str = ", ".join(authors) if authors else "Unknown Author"
                            
                            # Extract year properly
                            year = None
                            if 'published-print' in item and item['published-print'].get('date-parts'):
                                year = item['published-print']['date-parts'][0][0]
                            elif 'published-online' in item and item['published-online'].get('date-parts'):
                                year = item['published-online']['date-parts'][0][0]
                            elif 'created' in item and item['created'].get('date-parts'):
                                year = item['created']['date-parts'][0][0]
                            
                            # Get journal/venue information
                            journal = item.get('container-title', [''])[0] if item.get('container-title') else 'CrossRef'
                            
                            # Create paper entry
                            paper = {
                                'title': item.get('title', ['Unknown Title'])[0],
                                'authors': author_str,
                                'year': year or 'Unknown Year',
                                'journal': journal,
                                'url': item.get('URL', None),
                                'citations': item.get('is-referenced-by-count', 0),  # CrossRef does provide reference counts
                                'venue': journal
                            }
                            
                            # Don't add duplicate papers (case insensitive)
                            if not any(p['title'].lower() == paper['title'].lower() for p in papers):
                                papers.append(paper)
                    
                    time.sleep(1)  # Avoid rate limiting
                    
            except Exception as e:
                logger.error(f"CrossRef search failed: {str(e)}")
        
        # Try arXiv API as another backup
        elif strategy == 'arxiv' and len(papers) < num_papers:
            try:
                logger.info("Trying arXiv API for paper search...")
                for query in search_queries[:2]:  # Limit to just top queries
                    if len(papers) >= num_papers:
                        break
                        
                    encoded_query = urllib.parse.quote(query)
                    api_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=5"
                    
                    response = requests.get(api_url, timeout=10)
                    if response.status_code == 200:
                        # Parse XML response
                        from xml.etree import ElementTree as ET
                        root = ET.fromstring(response.text)
                        
                        # Define XML namespaces
                        ns = {
                            'atom': 'http://www.w3.org/2005/Atom',
                            'arxiv': 'http://arxiv.org/schemas/atom'
                        }
                        
                        for entry in root.findall('.//atom:entry', ns):
                            if len(papers) >= num_papers:
                                break
                                
                            # Extract paper details
                            title_elem = entry.find('./atom:title', ns)
                            title = title_elem.text if title_elem is not None else "Unknown Title"
                            
                            # Extract authors
                            authors = []
                            for author in entry.findall('./atom:author/atom:name', ns):
                                if author.text:
                                    authors.append(author.text)
                            
                            author_str = ", ".join(authors) if authors else "Unknown Author"
                            
                            # Extract URL
                            url = None
                            for link in entry.findall('./atom:link', ns):
                                if link.get('rel') == 'alternate' or link.get('title') == 'pdf':
                                    url = link.get('href')
                                    break
                            
                            # Extract publication date
                            published = entry.find('./atom:published', ns)
                            year = published.text[:4] if published is not None else "Unknown Year"
                            
                            # Extract journal info and categories
                            journal_ref = entry.find('./arxiv:journal_ref', ns)
                            journal = journal_ref.text if journal_ref is not None else "arXiv"
                            
                            # Extract categories for venue
                            categories = []
                            for category in entry.findall('./arxiv:primary_category', ns):
                                if category.get('term'):
                                    categories.append(category.get('term'))
                            
                            # Create paper entry
                            paper = {
                                'title': title,
                                'authors': author_str,
                                'year': year,
                                'journal': journal,
                                'url': url,
                                'citations': None,  # arXiv doesn't provide citation counts
                                'venue': 'arXiv: ' + ', '.join(categories) if categories else 'arXiv'
                            }
                            
                            # Don't add duplicate papers (case insensitive)
                            if not any(p['title'].lower() == paper['title'].lower() for p in papers):
                                papers.append(paper)
                    
                    time.sleep(1)  # Avoid rate limiting
                    
            except Exception as e:
                logger.error(f"arXiv search failed: {str(e)}")
    
    # Try Semantic Scholar as a last resort for better citation information
    if len(papers) < num_papers:
        try:
            logger.info("Trying Semantic Scholar API for better citation information...")
            # Try to enrich existing papers with citation information
            for i, paper in enumerate(papers):
                if paper.get('citations') is None or paper.get('citations') == 0:
                    try:
                        title = urllib.parse.quote(paper['title'])
                        api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&fields=title,authors,year,citationCount,venue,journal,url"
                        
                        response = requests.get(api_url, headers=HEADERS, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('data') and len(data['data']) > 0:
                                # Get first match as it's likely the best match
                                match = data['data'][0]
                                
                                # Update citation count if available
                                if 'citationCount' in match:
                                    papers[i]['citations'] = match['citationCount']
                                
                                # Update journal if not available
                                if (not paper.get('journal') or paper['journal'] == 'N/A') and match.get('journal'):
                                    papers[i]['journal'] = match['journal']
                                
                                # Update venue if not available
                                if (not paper.get('venue') or not paper['venue']) and match.get('venue'):
                                    papers[i]['venue'] = match['venue']
                                
                    except Exception as e:
                        logger.warning(f"Failed to enrich paper {paper['title']}: {str(e)}")
                    
                    time.sleep(1)  # Avoid rate limiting
                
            # Also fetch some new papers if needed
            if len(papers) < num_papers:
                for query in search_queries[:1]:  # Just try the main query
                    encoded_query = urllib.parse.quote(query)
                    api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,authors,year,citationCount,venue,journal,url&limit=5"
                    
                    response = requests.get(api_url, headers=HEADERS, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('data'):
                            for item in data['data']:
                                if len(papers) >= num_papers:
                                    break
                                
                                # Extract authors
                                authors = []
                                for author in item.get('authors', []):
                                    if author.get('name'):
                                        authors.append(author['name'])
                                
                                author_str = ", ".join(authors) if authors else "Unknown Author"
                                
                                # Create paper entry
                                paper = {
                                    'title': item.get('title', 'Unknown Title'),
                                    'authors': author_str,
                                    'year': item.get('year', 'Unknown Year'),
                                    'journal': item.get('journal', 'N/A'),
                                    'url': item.get('url'),
                                    'citations': item.get('citationCount', 0),
                                    'venue': item.get('venue', '')
                                }
                                
                                # Don't add duplicate papers (case insensitive)
                                if not any(p['title'].lower() == paper['title'].lower() for p in papers):
                                    papers.append(paper)
        
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {str(e)}")
    # If we still don't have enough papers, create ONE clear placeholder that indicates
    # the issue rather than fake papers that might mislead the user
    if not papers:
        logger.warning("No papers found from any source, creating a single placeholder")
        papers.append({
            'title': f"Search for papers related to '{', '.join(key_terms[:3])}'",
            'authors': "Search directly on Google Scholar",
            'year': "",
            'journal': "N/A",
            'url': f"https://scholar.google.com/scholar?q={urllib.parse.quote(' '.join(key_terms[:3]))}",
            'citations': None,
            'venue': "Multiple academic sources"
        })
    
    # Sort papers by citations if available, otherwise keep original order
    papers_with_citations = [p for p in papers if p.get('citations') is not None and p.get('citations') != 0]
    papers_without_citations = [p for p in papers if p.get('citations') is None or p.get('citations') == 0]
    
    sorted_papers = sorted(papers_with_citations, key=lambda x: x.get('citations', 0), reverse=True)
    sorted_papers.extend(papers_without_citations)
    
    # Ensure all papers have consistent fields before returning
    for paper in sorted_papers:
        paper['journal'] = paper.get('journal', 'Unknown Journal or Internet Source')
        paper['citations'] = paper.get('citations', 'Unknown')
        paper['year'] = paper.get('year', 'Unknown Year')
        
    return sorted_papers[:num_papers]


def main():
    st.title("🎙️ ResearchCast: Research Paper to Podcast Converter")
    st.markdown("#### Transform research papers into engaging podcast conversations 📚")   
    
    # Initialize session state variables if not exist   
    if "current_transcript" not in st.session_state:  
        st.session_state.current_transcript = None
    if "transcript_displayed" not in st.session_state:
        st.session_state.transcript_displayed = False
    if "audio_file" not in st.session_state:
        st.session_state.audio_file = None          
    if "audio_generation_requested" not in st.session_state:
        st.session_state.audio_generation_requested = False 
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = None
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = None
    if "key_terms" not in st.session_state:
        st.session_state.key_terms = None
    if "related_papers" not in st.session_state:
        st.session_state.related_papers = None
    if "current_input" not in st.session_state:  
        st.session_state.current_input = None
    if "kokoro_available" not in st.session_state:
        st.session_state.kokoro_available = False
    
    if "background_tracks" not in st.session_state:
        st.session_state.background_tracks = get_background_tracks()
    
    # Check audio assets
    check_audio_assets()
    
    # Check if Kokoro is installed and working
    st.session_state.kokoro_available = check_kokoro_installation()
    
    # Initialize APIs (keep your existing initialize_apis function)
    if not initialize_apis():
        st.stop()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("How to use:")   
        st.markdown("""\
        1. Upload a research paper (PDF) or provide a URL.
        2. Wait for the text to be processed (this may take a few moments).
        3. Customize audio settings (optional).   
        4. Generate podcast transcript and audio.
        5. Click "Get Summary" to get a brief overview of the provided article. 
        6. Ask questions if you have any in the Q&A section.
        7. Extract key terms and find related research papers.
        """)
        
        # Display TTS engine status
        if st.session_state.kokoro_available:
            print("Kokoro TTS is active and ready!")
        else:
            st.warning("Using Google TTS as fallback (Kokoro not available)")
            
        with st.expander("ℹ️ Supported Sources"):
            st.markdown("""
            Academic Sources:   
            - arXiv
            - PubMed
            - PLOS
            - DOI
            - Semantic Scholar
            - DOAJ
            - ResearchGate (Not Working Currently)
            
            General Sources:
            - News Articles (The Hindu, etc.)
            - Blog Posts (Medium, etc.)
            - General Web Articles
            """)
            
        # Button for generating only the summary (without reloading transcript/audio)
        if st.button("Get Summary"):
            if st.session_state.get("raw_text"):
                st.session_state.document_summary = generate_summary(st.session_state.raw_text)
            else:
                st.warning("No document text available for summary.")
    
    # 1. INPUT SECTION
    st.subheader("1️⃣ Input Research Paper")
    # Choose input method
    option = st.radio("Choose Input Method:", ["Upload PDF", "Provide URL"])
    raw_text = ""
    current_input = None
    
    if option == "Upload PDF":
        pdf_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")
        if pdf_file:
            current_input = f"pdf_{pdf_file.name}"
            # Only process if input has changed
            if current_input != st.session_state.current_input:
                raw_text = get_pdf_text([pdf_file])
                st.session_state.current_input = current_input
                # Reset previous results when input changes
                st.session_state.current_transcript = None
                st.session_state.audio_file = None
                st.session_state.document_summary = None
                st.session_state.key_terms = None
                st.session_state.related_papers = None
                st.session_state.audio_generation_requested = False
    elif option == "Provide URL":
        url = st.text_input("Enter Research Paper URL")
        if url:
            current_input = f"url_{url}"
            # Only process if input has changed
            if current_input != st.session_state.current_input:
                raw_text = get_text_from_url(url)
                st.session_state.current_input = current_input
                # Reset previous results when input changes
                st.session_state.current_transcript = None
                st.session_state.audio_file = None
                st.session_state.document_summary = None
                st.session_state.key_terms = None
                st.session_state.related_papers = None
                st.session_state.audio_generation_requested = False

    # Process text if we have new input
    if raw_text:
        st.session_state.raw_text = raw_text  # Save raw text in session state for later use
        with st.spinner("Processing Research Paper..."):
            text_chunks = get_text_chunks(raw_text)
            # Only generate transcript and audio if not already in session state
            if not st.session_state.get("current_transcript"):
                conversation_script = generate_conversation_script(raw_text)
                if conversation_script:
                    st.session_state.current_transcript = conversation_script
            
            # Replace extract_key_terms with extract_research_keywords
            if not st.session_state.get("key_terms"):
                with st.spinner("Extracting key terms..."):
                    try:
                        # Use the new function instead of the old ones
                        key_terms = extract_research_keywords(raw_text)
                        st.session_state.key_terms = key_terms if key_terms else []
                        
                        if not st.session_state.key_terms:
                            st.warning("Could not extract key terms from the document.")
                    except Exception as e:
                        st.error(f"Error extracting key terms: {str(e)}")
                        # Simple fallback without using the enhanced extraction
                        st.session_state.key_terms = ["General research", "Academic paper", "Study"]
            
            # Find related research papers if not already done
            if not st.session_state.get("related_papers") and st.session_state.key_terms:
                with st.spinner("Finding related research papers..."):
                    try:
                        st.session_state.related_papers = find_related_papers(st.session_state.key_terms)
                        if not st.session_state.related_papers:
                            st.warning("Could not find related papers.")
                    except Exception as e:
                        st.error(f"Error finding related papers: {str(e)}")
                        st.session_state.related_papers = []
    
    # Use raw_text from session state if available (for persistent processing)
    if not raw_text and "raw_text" in st.session_state:
        raw_text = st.session_state.raw_text
        text_chunks = get_text_chunks(raw_text)
    
    # Only display the remaining sections if we have processed content
    if raw_text or "raw_text" in st.session_state:
        
        # 2. TRANSCRIPT SECTION
        if st.session_state.get("current_transcript"):
            st.subheader("2️⃣ Podcast Transcript")
            st.text_area("", st.session_state.current_transcript, height=200)
            st.session_state.transcript_displayed = True
        
        # 3. AUDIO SETTINGS SECTION
        if st.session_state.get("current_transcript"):
            st.subheader("3️⃣ Audio Settings")
            
            # TTS engine information
            if st.session_state.kokoro_available:
                print("Using Kokoro TTS for high-quality voice generation")
            else:
                st.info("Using Google TTS for voice generation")   
            
            # Get audio settings from UI
            audio_settings = setup_audio_settings()
            
            # Generate audio button
            generate_audio_button = st.button("Generate Podcast Audio", type="primary")
            if generate_audio_button:
                st.session_state.audio_generation_requested = True
            
            # Generate audio if requested   
            if st.session_state.audio_generation_requested and not st.session_state.get("audio_file"):
                with st.spinner("Generating podcast audio... This may take a moment."):   
                    
                    voice_settings = audio_settings['voice_settings']
                    background_settings = audio_settings['background_settings']
        
                    if background_settings['auto_select'] and background_settings['enabled']:
                        background_settings = select_random_background() 
                        
                    audio_file, timestamps = generate_audio(
                        st.session_state.current_transcript,
                        voice_settings=voice_settings,
                        background_settings=background_settings
                    )
                       
                    if audio_file:
                        st.session_state.audio_file = audio_file
                        st.session_state.timestamps = timestamps
                        st.session_state.audio_generation_requested = False   
                        st.session_state.background_settings_used = background_settings
        
        # Display the summary if available
        if st.session_state.get("document_summary"):
            with st.expander("📄 Document Summary", expanded=False):
                st.write(st.session_state.document_summary)
        else:
            st.info("Click 'Get Summary' in the sidebar to generate a summary.")
            
        # 4. KEY TERMS SECTION - Display in 3 columns
        if st.session_state.get("key_terms") and st.session_state.key_terms:
            st.subheader("4️⃣ Key Terms")
            
            with st.expander("View Key Terms", expanded=False):
                # Calculate number of terms per column (rounded up)
                terms = st.session_state.key_terms
                total_terms = len(terms)
                terms_per_col = (total_terms + 2) // 3  # ceiling division to distribute evenly
                
                # Create 3 columns
                col1, col2, col3 = st.columns(3)
                
                # Distribute terms across columns
                with col1:
                    for i in range(0, min(terms_per_col, total_terms)):
                        st.write(f"• {terms[i]}")
                        
                with col2:
                    for i in range(terms_per_col, min(2 * terms_per_col, total_terms)):
                        st.write(f"• {terms[i]}")
                        
                with col3:
                    for i in range(2 * terms_per_col, total_terms):
                        st.write(f"• {terms[i]}")
        
        # 5. AUDIO PLAYBACK SECTION
        if st.session_state.get("audio_file"):
            st.subheader("5️⃣ Listen to Podcast")
            with open(st.session_state.audio_file, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
            
            # Display voice and background information
            if "voice_settings" in st.session_state:
                st.caption(f"Host Voice: {st.session_state.voice_settings['host']['name']} | " +
                          f"Guest Voice: {st.session_state.voice_settings['guest']['name']}")  
            
            if "background_settings_used" in st.session_state and st.session_state.background_settings_used['enabled']:
                st.caption(f"Background: {st.session_state.background_settings_used['track_name']} (Auto-selected)")
            
        # 6. Q&A SECTION
        if text_chunks:
            st.subheader("6️⃣ Ask Questions")
            vector_store = setup_qa_system(text_chunks)
            if vector_store:
                question = st.text_input("Type your question here ❓")
                if question:
                    time.sleep(1)  # Small delay to prevent rapid API calls
                    docs = vector_store.similarity_search(question) 
                    
                    context = "\n\n".join([doc.page_content for doc in docs if hasattr(doc, "page_content")])
                    if not context:
                        # If not, join directly as strings:
                        context = "\n\n".join(docs)   
                    
                    answer = generate_qa_answer(context, question)
                    st.markdown(f"**Answer:** {answer}")
    
        # 7. RELATED PAPERS SECTION - Using expanders for each paper
        if "related_papers" in st.session_state and st.session_state.related_papers:
            st.subheader("7️⃣ Related Research Papers")
        
            # Directly create expanders for each paper without nesting
            for i, paper in enumerate(st.session_state.related_papers, 1):
                with st.expander(f"{i}. {paper.get('title', 'Unknown Title')}", expanded=False):
                    st.write(f"**Authors:** {paper.get('authors', 'Unknown')}")
                    
                    # Journal information
                    journal_info = paper.get('journal', 'N/A')
                    if journal_info == 'N/A' or not journal_info:   
                        journal_info = paper.get('venue', 'Unknown Journal or Internet Source')  
                    
                    # Create two columns for metadata and link   
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Year:** {paper.get('year', 'Unknown')}")
                        st.write(f"**Citations:** {paper.get('citations', 'Unknown')}")  
                        st.write(f"**Journal:** {journal_info}")
                    
                    with col2:
                        if paper.get('url'):
                            st.markdown(f"[View Paper]({paper['url']})")     
if __name__ == "__main__":   
    main()

    
  
    
# https://arxiv.org/abs/2011.13801      
# https://arxiv.org/abs/2203.06207    
# https://pubmed.ncbi.nlm.nih.gov/34879315/         
# https://pubmed.ncbi.nlm.nih.gov/31923642/          
# https://journals.plos.org/climate/article?id=10.1371/journal.pclm.0000524   
# https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001397 
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0261230
# https://www.semanticscholar.org/paper/Quantum-computing-with-Qiskit-Javadi-Abhari-Treinish/ed1c8676a7c13421afa315af9702fad182ee2347
# https://doaj.org/article/3c847db3b1ae4ba1ae9f69bedf4ca3ce
# https://doaj.org/article/1c847db3b1ae4ba1ae9f69bedf4ca3ce
# https://doaj.org/article/4d847db3b1ae4ba1ae9f69bedf4ca3ce
# https://www.researchgate.net/publication/309543821_Applications_of_Cloud_Computing_in_Health_Systems --
# https://www.researchgate.net/publication/386049161_A_Revolution_in_Processing_Capabilities_and_Its_Possible_Uses_Quantum_Computing --   
# https://doi.org/10.1371/journal.pclm.0000524        
# https://doi.org/10.1016/j.cell.2019.01.023        


# https://www.bbc.com/news/articles/cg7zxgxdggjo
# https://indianexpress.com/article/technology/artificial-intelligence/how-sam-altman-sidestepped-elon-musk-to-win-over-donald-trump-9837292/
