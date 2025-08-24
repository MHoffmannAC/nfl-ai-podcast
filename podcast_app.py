import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import time
import re
import os

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

import re

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

# --- State Management ---
if "news_selected" not in st.session_state:
    st.session_state.news_selected = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn" not in st.session_state:
    st.session_state.turn = "bot_a"
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

def reset_app():
    """Resets the application state to the initial news selection screen."""
    st.session_state.news_selected = False
    st.session_state.messages = []
    st.session_state.turn = "bot_a"
    st.session_state.news_topic_index = 0
    st.session_state.topic_messages_count = 0
    st.session_state.pre_generated_responses = []
    st.session_state.podcast_started = False
    st.session_state.ready_to_start_podcast = False
    st.rerun()

# Initial news selection screen
if not st.session_state.news_selected:
    selected_headlines = st.multiselect(
        "Select the news article(s) covered in this episode:",
        options=[i['headline'] for i in news],
        placeholder="Choose headlines..."
    )
    podcast_length = st.selectbox("Select podcast length:", options=["Short", "Medium", "Long"], index=None)
    groq_token = st.text_input("Enter your Groq API Key:", type="password")
    model_name = st.selectbox("Select LLM Model:", options=["gemma2-9b-it", "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"], index=None)
    
    if st.button("Generate Podcast Episode"):
        if selected_headlines and groq_token and podcast_length:
            podcast_length = {"Short": 1, "Medium": 5, "Long": 10}[podcast_length]
            df = pd.DataFrame([
                {"news_id": headline_to_id[h], "headline": h, "story": headline_to_story[h]}
                for h in selected_headlines
            ])
            st.session_state.selected_news = df
            
            with st.spinner("Generating podcast episode... This may take a moment."):
                news_df = st.session_state.selected_news

                llm = Groq(model=model_name, api_key=groq_token)

                persona_a = f"""You are 'Dave,' a knowledgeable and analytical co-host of an NFL podcast. You provide thoughtful, fact-based insights on the latest news. You are often the more serious one, but you appreciate a good joke.
                Your job is to introduce a topic, provide analysis, and respond to your co-host's points. You should engage in a natural conversation. You have the ultimate authority to decide when the conversation moves to the next news story. 
                Keep your responses **concise**, no more than 2-3 sentences unless absolutely necessary.
                """
                bot_a_engine = SimpleChatEngine.from_defaults(llm=llm, memory=ChatMemoryBuffer.from_defaults(), prefix_messages=[ChatMessage(role=MessageRole.SYSTEM, content=persona_a)])

                persona_b = f"""You are 'Julia,' the co-host of an NFL podcast. You are the witty and lighthearted one, always ready with a joke or a sarcastic comment. Your primary role is to react to the news presented by 'Dave' and keep the mood fun and entertaining.
                You often use humor and irony in your responses. Don't be afraid to take a playful jab at 'Dave' or a particular team. You should engage in a natural conversation, reacting to Dave's points.
                Feel free to end your turns by asking 'Dave' a question to get his thoughts on your point.
                You tend to be a bit chattier, so your responses can be slightly longer, but still try to keep them to the point.
                """
                bot_b_engine = SimpleChatEngine.from_defaults(llm=llm, memory=ChatMemoryBuffer.from_defaults(), prefix_messages=[ChatMessage(role=MessageRole.SYSTEM, content=persona_b)])
            
                # Pre-generate responses
                current_turn = "bot_a"
                current_message_content = ""
                all_audio_segments = []
                
                while st.session_state.news_topic_index < news_df.shape[0]: # Generate a few responses per topic
                    print(f"Generating for topic index {st.session_state.news_topic_index}, message count {st.session_state.topic_messages_count}, speeker {current_turn}")
                    if current_turn == "bot_a":
                        next_bot = True
                        current_news = news_df.iloc[st.session_state.news_topic_index]
                        
                        # Dynamically add the news story context to the prompt for the first message of a new topic
                        if st.session_state.topic_messages_count == 0:
                            if st.session_state.news_topic_index == 0:
                                prompt = f"Welcome everyone to the 'Neural Zone Infraction' NFL Podcast. Include the tagline 'Breaking down the week in football, at machine speed.'. Introduce yourself as well as your co-host 'Julia'. Introduce then the first news story of the day: '{current_news['headline']}'. The story is: '{current_news['story']}'. Provide your initial thoughts."
                            else:
                                prompt = f"Introduce the next news story: '{current_news['headline']}'. The story is: '{current_news['story']}'. Provide your initial thoughts."
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

                        response = bot_a_engine.chat(prompt)
                        audio_data, duration = generate_tts_and_duration(response.response, speaker_embeddings["Dave"])
                        all_audio_segments.append(audio_data)
                        st.session_state.pre_generated_responses.append({'response': response.response, 'duration': duration, 'speaker_id': "Dave"})
                        current_message_content = response.response
                        if next_bot:
                            current_turn = "bot_b"

                    elif current_turn == "bot_b":
                        prompt = f"Your co-host Dave just said: '{current_message_content}'. React to his analysis with a lighthearted or ironic take. Keep it brief and then ask him a question to get his his thoughts on your point."
                        response = bot_b_engine.chat(prompt)
                        audio_data, duration = generate_tts_and_duration(response.response, speaker_embeddings["Julia"])
                        all_audio_segments.append(audio_data)
                        st.session_state.pre_generated_responses.append({'response': response.response, 'duration': duration, 'speaker_id': "Julia"})
                        current_message_content = response.response
                        current_turn = "bot_a"

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
            st.session_state.bot_a_engine = bot_a_engine
            st.session_state.bot_b_engine = bot_b_engine
            st.rerun()
        else:
            st.warning("Please choose all settings and select at least one headline.")
else:
    # Main podcast discussion

    # Wait for the user to press the start button
    if st.session_state.ready_to_start_podcast and not st.session_state.podcast_started:
        st.info("Podcast episode generated. Click 'Start Podcast' to begin listening.")
        if st.button("Start Podcast"):
            st.session_state.podcast_started = True
            # We don't need a rerun here, the script will continue execution

    # This block will execute once the "Start Podcast" button is clicked
    if st.session_state.podcast_started:
        # Play the single, combined audio file
        st.audio(io.BytesIO(base64.b64decode(st.session_state.combined_audio_b64)), format='audio/wav', autoplay=True)

        # Loop through the pre-generated responses and display them sequentially
        for response_data in st.session_state.pre_generated_responses:
            response = response_data['response']
            duration = response_data['duration']
            speaker_id = response_data['speaker_id']
            avatar = "🎙️" if speaker_id == "Dave" else "🎧"

            # Display the message
            with st.chat_message(speaker_id, avatar=avatar):
                st.markdown(response)
            
            # Save the message to history so it persists
            st.session_state.messages.append({"role": speaker_id, "content": response, "avatar": avatar})
            
            # Wait for the duration of this specific audio segment before showing the next message
            #time.sleep(duration + 1) # Adding a 1-second pause between speakers

        # Once the loop is finished, the podcast is over
        st.info("The hosts have covered all the news topics for today's show. You can start a new episode below.")
        
        user_filename_input = st.text_input(
            label="Enter a name for the podcast:",
            value=None
        )

        # Sanitize and Validate the filename
        sanitized_filename = user_filename_input.replace(' ', '_')
        sanitized_filename = re.sub(r'[^\w]', '', sanitized_filename)

        # Use columns for the three action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            # Action 1: Download the file to the user's local machine
            if not sanitized_filename:
                st.download_button("📥 Download", data=b"", disabled=True)
            else:
                final_filename = f"{sanitized_filename}.wav"
                audio_bytes = base64.b64decode(st.session_state.combined_audio_b64)
                st.download_button(
                    label="📥 Download",
                    data=audio_bytes,
                    file_name=final_filename,
                    mime="audio/wav"
                )

        with col2:
            # Action 2: Save the file to the server's filesystem
            if not sanitized_filename:
                st.button("💾 Publish to Server", disabled=True)
            else:
                if st.button("💾 Publish to Server"):
                    save_dir = "prerecorded_episodes"
                    # Create the directory if it doesn't exist
                    os.makedirs(save_dir, exist_ok=True)
                    
                    final_filename = f"{sanitized_filename}.wav"
                    file_path = os.path.join(save_dir, final_filename)
                    
                    audio_bytes = base64.b64decode(st.session_state.combined_audio_b64)
                    with open(file_path, "wb") as f:
                        f.write(audio_bytes)
                    st.success(f"Saved to: {file_path}")

        with col3:
            # Action 3: Start a new episode
            if st.button("🔄 Start New"):
                reset_app()
        
        # Prevent this block from re-running after it's finished
        #st.session_state.podcast_started = False

