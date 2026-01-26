import pickle
import os
import base64
import io
import textwrap
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, concatenate_audioclips

# TODO:
# 2025_Nevada_Lawsuit
# 2025_Saints_lease_extended
# 2025_general_OvertimeRules (already ready)

# --- Configuration ---
PICKLE_FILE_PATH = "prerecorded_episodes/2025_Trump_Kickoff_Rules"
season_title="Season 2025/26"
main_title="Trump vs. Kickoff rules"

INTRO_IMAGE_PATH = "assets/intro_card.png"   
OUTRO_IMAGE_PATH = "assets/outro_card.png"  
INTRO_SONG_PATH  = "assets/intro_music.mp3" 
OUTRO_SONG_PATH  = "assets/outro_music.mp3"  
BG_IMAGE_PATH = "assets/background.png"
SPEAKER_IMAGES = {
    "Dave": "assets/dave.png",
    "Julia": "assets/julia.png"
}

# 2. VIDEO SETTINGS
INTRO_DURATION = 16.75  # seconds
OUTRO_DURATION = 16.75  # seconds
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 24
#SPEED_FACTOR = 1.2

# 3. OUTPUT FILENAME
FINAL_MOVIE_PATH = main_title.replace(" ", "_")+".mp4"

# 4. TEXT & APPEARANCE SETTINGS
SPEAKER_COLORS = {
    "Dave": "#222244", 
    "Julia": "#442222" 
}
TEXT_COLOR = "#e2e8f0"      # Light Gray/White
TEXT_PADDING = 50            # Padding inside the text box
FONT_SIZE_TEXT = 50        # Initial font size for dialogue text. Adjusted to fit box.


def wrap_text_to_box(text, font, max_width):
    """
    Wrap text to fit within a maximum pixel width using Pillow >= 10.
    Returns a list of lines.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = font.getbbox(test_line)  # Returns (x0, y0, x1, y1)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:  # avoid empty line
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def create_text_image(text, speaker, width, height, title=season_title, subtitle=main_title):
    """Creates a PIL Image with the speaker's name and dialogue, auto-scaling font to fit the box."""

    img = Image.open(BG_IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    # --- Fonts ---
    try:
        title_font = ImageFont.truetype("arial.ttf", 50)
        subtitle_font = ImageFont.truetype("arial.ttf", 70)
    except IOError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()

    # --- Draw title/subtitle ---
    text_color = TEXT_COLOR
    text_start_y = 50
    draw.text((320, text_start_y), title, font=title_font, fill=text_color)
    text_start_y += 100
    draw.text((320, text_start_y), subtitle, font=subtitle_font, fill=text_color)
    text_start_y += 160
    draw.line([(0, text_start_y-30), (width, text_start_y-30)], fill="white", width=2)

    # --- Box coordinates ---
    if speaker == "Dave":
        box_x0, box_x1 = 50, 800
        img_x = box_x1 + 20
    else:
        box_x0, box_x1 = 280, 1030
        img_x = box_x0 - 260
    box_y0 = text_start_y + 50

    max_box_height = height - box_y0 - 50  # leave space at bottom
    box_width = box_x1 - box_x0 - 2 * TEXT_PADDING

    # --- Auto-scale font ---
    font_size = FONT_SIZE_TEXT
    while font_size > 10:
        try:
            text_font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            text_font = ImageFont.load_default()

        # Wrap text
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            line_width = text_font.getbbox(test_line)[2] - text_font.getbbox(test_line)[0]
            if line_width <= box_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Compute total height
        line_height = text_font.getbbox("A")[3] - text_font.getbbox("A")[1]
        total_text_height = len(lines) * (line_height + 25) + 2*TEXT_PADDING

        if total_text_height <= max_box_height:
            break
        font_size -= 2

    box_y1 = box_y0 + total_text_height

    # --- Draw box ---
    fill_color = SPEAKER_COLORS.get(speaker, TEXT_COLOR)
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=fill_color, outline="white", width=4)

    # --- Draw text ---
    for i, line in enumerate(lines):
        y = box_y0 + TEXT_PADDING + i * (line_height + 25)
        draw.text((box_x0 + TEXT_PADDING, y), line, font=text_font, fill=TEXT_COLOR)

    # --- Draw speaker image ---
    if speaker in SPEAKER_IMAGES:
        speaker_img = Image.open(SPEAKER_IMAGES[speaker]).convert("RGBA")
        speaker_img.thumbnail((240, 240))
        img.paste(speaker_img, (img_x, box_y0), speaker_img)

    return np.array(img)


def create_cover_image(image_path, title, subtitle):

    cover_img_width, cover_img_height = 1080, 1920
    cover_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(cover_img)
    try:
        cover_font_main = ImageFont.truetype("arial.ttf", 75)
        cover_font_sub = ImageFont.truetype("arial.ttf", 55)
    except IOError:
        cover_font_main = ImageFont.load_default()
        cover_font_sub = ImageFont.load_default()
    
    title_text = title
    year_week_text = subtitle
#    box_coords = [
#          50, cover_img_height/2 + 200,
#        1030, cover_img_height/2 + 650
#    ]
#    draw.rectangle(box_coords, fill=(256,256,256,1))
    
    draw.text((cover_img_width/2, cover_img_height/2 + 300), year_week_text, font=cover_font_sub, fill="white", anchor="mm")
    draw.text((cover_img_width/2, cover_img_height/2 + 450), title_text, font=cover_font_main, fill="white", anchor="mm")
    
    return np.array(cover_img)

def create_info_image(image_path):
    info_img_width, info_img_height = 1080, 1920
    info_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(info_img)
    try:
        info_font_main = ImageFont.truetype("arial.ttf", 80)
        info_font_sub = ImageFont.truetype("arial.ttf", 60)
        info_font_subsub = ImageFont.truetype("arial.ttf", 50)
    except IOError:
        info_font_main = ImageFont.load_default()
        info_font_sub = ImageFont.load_default()
    
    text1 = f"Are you interested in more"
    draw.text((info_img_width/2, 900), text1, font=info_font_sub, fill="white", anchor="mm")
    text1b = f"NFL analytics?"
    draw.text((info_img_width/2, 980), text1b, font=info_font_sub, fill="white", anchor="mm")
    text2 = f"Do you enjoy Data Science?"
    draw.text((info_img_width/2, 1150), text2, font=info_font_sub, fill="white", anchor="mm")
    text2b = f"(Machine Learning, GenAI, ...)"
    draw.text((info_img_width/2, 1230), text2b, font=info_font_subsub, fill="white", anchor="mm")
    text3 = f"Then head over to"
    draw.text((info_img_width/2, 1400), text3, font=info_font_sub, fill="white", anchor="mm")
    text4 = f"nfl-gameday.streamlit.app"
    draw.text((info_img_width/2, 1550), text4, font=info_font_main, fill="white", anchor="mm")
    
    draw.line([(60,1588), (183,1588)], fill="white", width=4)
    draw.line([(235,1588), (475,1588)], fill="white", width=4)
    draw.line([(510,1588), (909,1588)], fill="white", width=4)
    draw.line([(934,1588), (955,1588)], fill="white", width=4)
    draw.line([(980,1588), (1018,1588)], fill="white", width=4)

    return np.array(info_img)

def main():
    """Main function to generate the podcast movie."""

    # --- 1. Load Data from Pickle File ---
    print(f"Loading episode data from {PICKLE_FILE_PATH}...")
    if not os.path.exists(PICKLE_FILE_PATH):
        print(f"Error: Pickle file not found at '{PICKLE_FILE_PATH}'. Please check the path.")
        return

    with open(PICKLE_FILE_PATH, "rb") as f:
        episode_data = pickle.load(f)
    
    responses = episode_data["pre_generated_responses"]
    audio_b64 = episode_data["combined_audio_b64"]

    # --- 2. Prepare Audio Files ---
    print("Preparing audio tracks...")
    
    # Decode the main dialogue and save to a temporary file
    audio_bytes = base64.b64decode(audio_b64)
    dialogue_audio_path = "temp_dialogue.wav"
    with open(dialogue_audio_path, "wb") as f:
        f.write(audio_bytes)

    # Load all audio clips
    intro_audio = AudioFileClip(INTRO_SONG_PATH).subclip(0, INTRO_DURATION)
    
    #y, sr = librosa.load(dialogue_audio_path, sr=None)
    #y_stretched = librosa.effects.time_stretch(y, rate=SPEED_FACTOR)
    #stretched_path = "temp_dialogue_stretched.wav"
    #sf.write(stretched_path, y_stretched, sr)
    #dialogue_audio = AudioFileClip(stretched_path)
    
    dialogue_audio = AudioFileClip(dialogue_audio_path)
    
    outro_audio = AudioFileClip(OUTRO_SONG_PATH).subclip(0, OUTRO_DURATION)

    # Concatenate audio clips into the final soundtrack
    final_audio = concatenate_audioclips([intro_audio, dialogue_audio, outro_audio])
    print(f"Final audio duration: {final_audio.duration:.2f} seconds")

    # --- 3. Create Video Clips ---
    print("Generating video clips for each message...")
    all_video_clips = []

    # Create Intro Clip
    cover_image = create_cover_image(INTRO_IMAGE_PATH, main_title, season_title)
    intro_clip = ImageClip(cover_image, duration=INTRO_DURATION)
    all_video_clips.append(intro_clip)

    # Create Dialogue Clips
    total_dialogue_duration = 0
    for msg in responses:
        speaker = msg['speaker_id']
        text = msg['response']
        duration = msg['duration']
        
        # Create an image for this piece of dialogue
        text_image_np = create_text_image(text, speaker, VIDEO_WIDTH, VIDEO_HEIGHT)
        img = Image.fromarray(text_image_np)
        img.save("text_image.png")
        # Create a video clip from the image
        adjusted_duration = duration # / SPEED_FACTOR
        dialogue_clip = ImageClip(text_image_np, duration=adjusted_duration)
        all_video_clips.append(dialogue_clip)
        total_dialogue_duration += duration

    print(f"Total dialogue video duration: {total_dialogue_duration:.2f} seconds")

    # Create Outro Clip
    outro_image = create_info_image(OUTRO_IMAGE_PATH)
    outro_clip = ImageClip(outro_image, duration=OUTRO_DURATION)
    all_video_clips.append(outro_clip)

    # --- 4. Assemble and Export Final Movie ---
    print("Concatenating all video clips...")
    final_video = concatenate_videoclips(all_video_clips, method="compose")
    
    # Set the combined audio to the final video
    final_video = final_video.set_audio(final_audio)
    final_video.fps = VIDEO_FPS
    
    print(f"Writing final movie to {FINAL_MOVIE_PATH}...")
    # Use 'libx264' for video and 'aac' for audio, common for MP4
    final_video.write_videofile(
        FINAL_MOVIE_PATH,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )
    
    # Clean up the temporary dialogue file
    os.remove(dialogue_audio_path)
    
    print("\nMovie generation complete! ✨")
    print(f"Your video is saved at: {FINAL_MOVIE_PATH}")


if __name__ == "__main__":
    main()