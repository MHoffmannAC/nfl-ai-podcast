import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import time
import re
import os
import pickle
import num2words

# TTS Imports
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import streamlit.components.v1 as components

# Podcast App Imports
from llama_index.llms.groq import Groq
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Database Imports
from sqlalchemy import create_engine, text

# --- Configuration and Setup ---
st.set_page_config(page_title="The 'Neural Zone infraction' NFL Podcast")
st.html("""
    <style>
        /* General chat message bubble style */
        [data-testid="stChatMessage"] {
            max-width: 70%;     
            padding: 1rem 1rem;
            border-radius: 12px;
            margin: 0.5rem 0; 
        }

        /* Dave (left aligned) */
        [class*="st-key-Dave"] [data-testid="stChatMessage"] { 
            flex-direction: row;
            text-align: left;
            margin-right: auto;   /* push toward left */
            background-color: #222244; 
        }

        /* Julia (right aligned) */
        [class*="st-key-Julia"] [data-testid="stChatMessage"] {
            flex-direction: row-reverse;
            text-align: left;
            margin-left: auto;    /* push toward right */
            background-color: #442222;
        }
    </style>
""")

st.title("The 'Neural Zone infraction' NFL Podcast")
st.markdown("Listen to our AI co-hosts Dave and Julia discuss the latest NFL news!")

@st.cache_data
def get_news():
    """Fetches the latest NFL news from the database."""
    def query_db(_sql_engine, query, **params):
        """Executes a SQL query and returns the results.
        
        Note: The leading underscore on _sql_engine tells Streamlit not to hash
        this argument for caching purposes, which prevents the unhashable type error.
        """
        with _sql_engine.connect() as conn:
            result = conn.execute(text(query), params)
            if result.returns_rows:
                return [dict(row._mapping) for row in result]
            else:
                conn.commit()
                return None
    sql_engine = create_engine(f"mysql+pymysql://avnadmin:{st.secrets['aiven_pwd']}@mysql-nfl-mhoffmann-nfl.b.aivencloud.com:10448/nfl", pool_size=20, max_overflow=50)
    news = query_db(sql_engine, "SELECT * FROM news WHERE published >= NOW() - INTERVAL 7 DAY ORDER BY published DESC;")
    return news

# Fetch news from the database
news = get_news()

headline_to_story = {i["headline"]: i["story"] for i in news}
headline_to_id = {i["headline"]: i["news_id"] for i in news}

# --- TTS Model and Embeddings Setup ---
@st.cache_resource
def load_tts_models():
    """Load and cache the TTS models and vocoder."""
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return processor, model, vocoder

@st.cache_resource
def load_speaker_embeddings():
    """Load and cache the speaker embeddings for the hosts."""
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # Using different embeddings for different voices
    embeddings_dave = torch.tensor(embeddings_dataset[1146]["xvector"]).unsqueeze(0)
    embeddings_julia = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return {"Dave": embeddings_dave, "Julia": embeddings_julia}

processor, model, vocoder = load_tts_models()
speaker_embeddings = load_speaker_embeddings()

def generate_tts_and_duration(text, speaker_embeddings):
    """
    Generates speech from text and returns it as a base64 encoded string and duration.
    
    Args:
        text (str): The text to convert to speech.
        speaker_embeddings (torch.Tensor): The speaker embeddings for the voice.
    
    Returns:
        tuple: A tuple containing the base64-encoded string and the audio duration in seconds.
    """
    
    all_speech = []
    total_duration = 0.0

    # Split the text into chunks
    chunks = split_text_into_chunks(text)
    
    for chunk in chunks:
        inputs = processor(text=chunk, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        all_speech.append(speech.numpy())
        
    # Concatenate all generated speech tensors
    combined_speech = np.concatenate(all_speech)
    total_duration = len(combined_speech) / 16000.0  # Sample rate is 16000
    
    # Encode to base64
    return combined_speech, total_duration

def split_text_into_chunks(text, max_chars=400):
    """
    Splits a long text into a list of strings, ensuring each chunk is below a certain character limit.
    It attempts to split by sentences first, then falls back to a simpler split if needed.
    """
    sentences = re.split('(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " "
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def clean_response(text):
    """
    Cleaning:
       -    Converts all digits in a string to their English word equivalents.
       -    Removes hyphens/minus from the text.
       -    Replace NFL by N F L.
    """
    
    # Regex to find all numbers in the text
    pattern = re.compile(r'-?\d+\.?\d*')
    matches = pattern.findall(text)
    for match in matches:
        try:
            if '.' in match:
                number_in_words = num2words.num2words(float(match))
            else:
                number_in_words = num2words.num2words(int(match))
            
            text = text.replace(match, number_in_words, 1) # Only replace the first instance to handle duplicates
        except Exception as e:
            print(f"Could not convert number {match} to words: {e}")
            pass
    
    # Remove hyphens/minus
    text = text.replace("-", " ")
    text = text.replace("–", " ") # en-dash
    text = text.replace("—", " ") # em-dash
    
    # Replace NFL with N F L
    text = re.sub(r'\bNFL\b', 'N F L', text)
    
    return text

# --- State Management ---
if "news_selected" not in st.session_state:
    st.session_state.news_selected = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn" not in st.session_state:
    st.session_state.turn = "dave"
if "news_topic_index" not in st.session_state:
    st.session_state.news_topic_index = 0
if "topic_messages_count" not in st.session_state:
    st.session_state.topic_messages_count = 0
if "pre_generated_responses" not in st.session_state:
    st.session_state.pre_generated_responses = []
if "podcast_started" not in st.session_state:
    st.session_state.podcast_started = False
if "ready_to_start_podcast" not in st.session_state:
    st.session_state.ready_to_start_podcast = False
if "first_run" not in st.session_state:
    st.session_state.first_run = True

def reset_app():
    """Resets the application state to the initial news selection screen."""
    st.session_state.news_selected = False
    st.session_state.messages = []
    st.session_state.turn = "dave"
    st.session_state.news_topic_index = 0
    st.session_state.topic_messages_count = 0
    st.session_state.pre_generated_responses = []
    st.session_state.podcast_started = False
    st.session_state.ready_to_start_podcast = False
    st.session_state.first_run = True
    st.rerun()

# Initial news selection screen
if not st.session_state.news_selected:
    col_settings, col_load = st.columns(2, gap="large")

    with col_settings:
        st.header("Generate New Episode")
        selected_headlines = st.multiselect(
            "Select the news article(s) covered in this episode:",
            options=[i['headline'] for i in news],
            placeholder="Choose headlines...",
            help="Select one or more headlines to include in the podcast episode."
        )
        podcast_length = st.selectbox("Select podcast length:", options=["Teaser", "Short", "Medium", "Long"], index=None, help="Teaser: 2-3 dialogues per topic, Short: ~5 dialogues per topic, Medium: ~10 dialogues per topic, Long: ~20 dialogues per topic")
        groq_token = st.text_input("Enter your Groq API Key:", type="password", help="Get your API key from https://www.groq.com")
        model_name = st.selectbox("Select LLM Model:", options=["gemma2-9b-it", "llama-3.3-70b-versatile", "qwen/qwen3-32b"], index=None, help="Choose the large language model for dialogue generation.")
    
        if st.button("Generate Podcast Episode"):
            try:
                if selected_headlines and groq_token and podcast_length:
                    podcast_length = {"Teaser": 1, "Short": 2, "Medium": 5, "Long": 10}[podcast_length]
                    df = pd.DataFrame([
                        {"news_id": headline_to_id[h], "headline": h, "story": headline_to_story[h]}
                        for h in selected_headlines
                    ])
                    st.session_state.selected_news = df
                    
                    progress_bar = st.progress(0)

                    news_df = st.session_state.selected_news

                    llm = Groq(model=model_name, api_key=groq_token)

                    persona_a = f"""You are 'Dave,' a knowledgeable and analytical co-host of an NFL podcast. You provide thoughtful, fact-based insights on the latest news. You are often the more serious one, but you appreciate a good joke.
                    Your job is to introduce a topic, provide analysis, and respond to your co-host's points. You should engage in a natural conversation. You have the ultimate authority to decide when the conversation moves to the next news story. 
                    Keep your responses **concise**, no more than 2-3 sentences unless absolutely necessary.
                    """
                    dave_engine = SimpleChatEngine.from_defaults(llm=llm, memory=ChatMemoryBuffer.from_defaults(), prefix_messages=[ChatMessage(role=MessageRole.SYSTEM, content=persona_a)])

                    persona_b = f"""You are 'Julia,' the co-host of an NFL podcast. You are the witty and lighthearted one, always ready with a joke or a sarcastic comment. Your primary role is to react to the news presented by 'Dave' and keep the mood fun and entertaining.
                    You often use humor and irony in your responses. Don't be afraid to take a playful jab at 'Dave' or a particular team. You should engage in a natural conversation, reacting to Dave's points.
                    Feel free to end your turns by asking 'Dave' a question to get his thoughts on your point.
                    You tend to be a bit chattier, so your responses can be slightly longer, but still try to keep them to the point.
                    """
                    julia_engine = SimpleChatEngine.from_defaults(llm=llm, memory=ChatMemoryBuffer.from_defaults(), prefix_messages=[ChatMessage(role=MessageRole.SYSTEM, content=persona_b)])
                
                    # Pre-generate responses
                    current_turn = "dave"
                    current_message_content = ""
                    all_audio_segments = []
                    silent_segment = np.zeros(16000, dtype=np.float32) # 16000 samples for 1 second at 16kHz
                    all_audio_segments.append(silent_segment)
                    
                    while st.session_state.news_topic_index < news_df.shape[0]: # Generate a few responses per topic
                        with st.spinner("Generating podcast episode... This may take a moment."):
                            print(f"Generating for topic index {st.session_state.news_topic_index}, message count {st.session_state.topic_messages_count}, speeker {current_turn}")
                            if current_turn == "dave":
                                next_bot = True
                                current_news = news_df.iloc[st.session_state.news_topic_index]
                                
                                # Dynamically add the news story context to the prompt for the first message of a new topic
                                if st.session_state.topic_messages_count == 0:
                                    if st.session_state.news_topic_index == 0:
                                        prompt = f"Welcome everyone to the 'Neural Zone Infraction' NFL Podcast. Include the tagline 'Breaking down the week in football, at machine speed.'. Introduce yourself as well as your co-host 'Julia'. Introduce then the first news story of the day: '{current_news['headline']}'. The story is: '{current_news['story']}'. Provide your initial thoughts."
                                    else:
                                        prompt = f"Introduce the next news story: '{current_news['headline']}'. The story is: '{current_news['story']}'. Provide your initial thoughts."
                                # If enough messages have been exchanged on this topic, move to the next one or wrap up
                                elif st.session_state.topic_messages_count >= 2*podcast_length:
                                    if st.session_state.news_topic_index + 1 < news_df.shape[0]:
                                        prompt = f"Summarize the current news story '{current_news['headline']}' and provide a closing remark. Then transition to the next news story with the sentence: 'Alright, let's move on to the next big story...'"
                                    else: # Last news story, so just wrap up
                                        prompt = f"Summarize the current news story '{current_news['headline']}' and provide a closing remark to end the podcast episode."
                                    st.session_state.news_topic_index += 1
                                    st.session_state.topic_messages_count = -1
                                    next_bot = False
                                else:
                                    prompt = f"Your co-host Julia just said: '{current_message_content}'. Respond to his point, but also tie it back to the main news story or a related point. Don't just acknowledge his joke."

                                response = dave_engine.chat(prompt)
                                cleaned_response = clean_response(response.response)
                                audio_data, duration = generate_tts_and_duration(cleaned_response, speaker_embeddings["Dave"])
                                all_audio_segments.append(audio_data)
                                st.session_state.pre_generated_responses.append({'response': response.response, 'duration': duration, 'speaker_id': "Dave"})
                                current_message_content = response.response
                                if next_bot:
                                    current_turn = "julia"

                            elif current_turn == "julia":
                                prompt = f"Your co-host Dave just said: '{current_message_content}'. React to his analysis with a lighthearted or ironic take. Keep it brief and then ask him a question to get his his thoughts on your point."
                                response = julia_engine.chat(prompt)
                                cleaned_response = clean_response(response.response)
                                audio_data, duration = generate_tts_and_duration(cleaned_response, speaker_embeddings["Julia"])
                                all_audio_segments.append(audio_data)
                                st.session_state.pre_generated_responses.append({'response': response.response, 'duration': duration, 'speaker_id': "Julia"})
                                current_message_content = response.response
                                current_turn = "dave"

                        progress = (st.session_state.news_topic_index * 2 * podcast_length + st.session_state.topic_messages_count) / (news_df.shape[0] * 2 * podcast_length)
                        progress_bar.progress(progress)
                        st.session_state.topic_messages_count += 1
                            
                        combined_audio_numpy = np.concatenate(all_audio_segments)
                        # Save the combined audio to a BytesIO object in memory
                        wav_io = io.BytesIO()
                        sf.write(wav_io, combined_audio_numpy, samplerate=16000, format="WAV")
                        wav_io.seek(0)
                        
                        # Encode the final combined audio to base64
                        audio_base64 = base64.b64encode(wav_io.read()).decode('utf-8')
                        st.session_state.combined_audio_b64 = audio_base64
                        
                    st.session_state.news_selected = True
                    st.session_state.ready_to_start_podcast = True
                    st.session_state.dave_engine = dave_engine
                    st.session_state.julia_engine = julia_engine
                    st.rerun()
                else:
                    st.warning("Please choose all settings and select at least one headline.")
            except Exception as e:
                st.error(f"An error occurred during podcast generation. Please try again.")
                reset_app()
    with col_load:
        st.header("Load Old Episode")
        save_dir = "prerecorded_episodes"
        # Ensure the directory exists to prevent errors
        os.makedirs(save_dir, exist_ok=True)
        
        # List all .pkl files in the directory
        episode_files = [f for f in os.listdir(save_dir)] #  if f.endswith('.pkl')]
        
        if not episode_files:
            st.info("No prerecorded episodes found.")
        else:
            selected_episode_file = st.selectbox("Select an episode to load:", options=episode_files, index=None)
            
            if st.button("Load Old Episode"):
                if selected_episode_file:
                    file_path = os.path.join(save_dir, selected_episode_file)
                    
                    with open(file_path, "rb") as f:
                        episode_data = pickle.load(f)
                    
                    st.session_state.combined_audio_b64 = episode_data["combined_audio_b64"]
                    st.session_state.pre_generated_responses = episode_data["pre_generated_responses"]
                    st.session_state.news_selected = True
                    st.session_state.ready_to_start_podcast = True
                    st.rerun()
                else:
                    st.warning("Please select an episode to load.")
else:
    # Main podcast discussion

    # Wait for the user to press the start button
    if st.session_state.ready_to_start_podcast and not st.session_state.podcast_started:
        st.info("Podcast episode ready to play. Click 'Start Podcast' to begin listening.")
        if st.button("Start Podcast"):
            st.session_state.podcast_started = True
            # We don't need a rerun here, the script will continue execution

    # This block will execute once the "Start Podcast" button is clicked
    if st.session_state.podcast_started:
        # Play the single, combined audio file
        time.sleep(1)
        st.audio(io.BytesIO(base64.b64decode(st.session_state.combined_audio_b64)), format='audio/wav', autoplay=True)
        time.sleep(1)
        # Loop through the pre-generated responses and display them sequentially
        for idx, response_data in enumerate(st.session_state.pre_generated_responses):
            response = response_data['response']
            duration = response_data['duration']
            speaker_id = response_data['speaker_id']
            avatar = ":material/face:" if speaker_id == "Dave" else ":material/support_agent:"

            # Display the message
            with st.container(key=f"{speaker_id}-{idx}"):
                with st.chat_message(speaker_id, avatar=avatar):
                    st.write(response)
            
            # Save the message to history so it persists
            st.session_state.messages.append({"role": speaker_id, "content": response, "avatar": avatar})
            
            # Wait for the duration of this specific audio segment before showing the next message
            if st.session_state.first_run:
                time.sleep(duration + 1) # Adding a 1-second pause between speakers

        # Once the loop is finished, the podcast is over
        st.divider()
        st.session_state.first_run = False
        st.info("The hosts have covered all the news topics for today's show. You can save the current episode or start a new episode below.")

        col1, col2 = st.columns([1,2], gap="large")

        with col1:
            # Action 3: Start a new episode
            if st.button("🔄 Start New Episode"):
                reset_app()

        with col2:
            user_filename_input = st.text_input(
                label="Enter a name for the podcast:",
                value=None,
                max_chars=50,
                placeholder="e.g., nfl_podcast_aug_24_2025"
            )

            # Sanitize and Validate the filename
            if user_filename_input:
                sanitized_filename = user_filename_input.replace(' ', '_')
                sanitized_filename = re.sub(r'[^\w]', '', sanitized_filename)
            else:
                sanitized_filename = ""

            # Action 1: Download the file to the user's local machine
            if not sanitized_filename:
                st.download_button("📥 Download Audio", data=b"", disabled=True)
            else:
                final_filename = f"{sanitized_filename}.wav"
                audio_bytes = base64.b64decode(st.session_state.combined_audio_b64)
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name=final_filename,
                    mime="audio/wav"
                )

            # Action 2: Save the file to the server's filesystem
            if not sanitized_filename:
                st.button("💾 Publish Episode to Server", disabled=True)
            else:
                if st.button("💾 Publish Episode to Server"):
                    save_dir = "prerecorded_episodes"
                    # Create the directory if it doesn't exist
                    os.makedirs(save_dir, exist_ok=True)
                    
                    final_filename = f"{sanitized_filename}"
                    file_path = os.path.join(save_dir, final_filename)
                    
                    episode_data = {
                            "combined_audio_b64": st.session_state.combined_audio_b64,
                            "pre_generated_responses": st.session_state.pre_generated_responses
                        }
                        
                    with open(file_path, "wb") as f:
                        pickle.dump(episode_data, f)
                    st.success(f"Saved episode data to: {file_path}")


