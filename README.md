# ============================================================
# Bambara Professional Video Dubber - صوت نقي بالوقار والثبات
# ============================================================

import streamlit as st
import tempfile
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from moviepy.editor import VideoFileClip, AudioFileClip
import re

# -------------------- إعداد الصفحة --------------------
st.set_page_config(
    page_title="دبلجة فيديو إلى البامبارا - صوت نقي",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 دبلجة فيديو احترافية إلى البامبارا")
st.markdown("""
**صوت نقي جداً • بدون قرقرة • بالوقار والثبات • قراءة صحيحة للأرقام والنص**
""")

# -------------------- تحميل النماذج --------------------
@st.cache_resource(show_spinner="جاري تحميل نماذج MALIBA-AI...")
def load_models():
    from whosper import WhosperTranscriber
    from maliba_ai.tts.inference import BambaraTTSInference
    from maliba_ai.config.settings import Speakers

    asr = WhosperTranscriber(model_id="MALIBA-AI/bambara-asr-v3")
    tts = BambaraTTSInference()
    return asr, tts, Speakers

try:
    asr_model, tts_model, Speakers = load_models()
    speakers_list = list(Speakers)
except Exception as e:
    st.error(f"فشل تحميل النماذج: {e}")
    st.stop()

# -------------------- دوال مساعدة --------------------
def extract_audio(video_path: str, audio_path: str):
    video = VideoFileClip(video_path)
    if video.audio is None:
        raise ValueError("الفيديو لا يحتوي على صوت")
    video.audio.write_audiofile(audio_path, logger=None, verbose=False)
    duration = video.duration
    video.close()
    return duration

def time_stretch(y, sr, target_duration):
    current = librosa.get_duration(y=y, sr=sr)
    if current <= 0:
        return y
    rate = current / target_duration
    return librosa.effects.time_stretch(y, rate=rate)

def clean_and_normalize_text(text: str) -> str:
    """تنظيف النص وتجهيزه للقراءة النقية"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # إزالة الرموز الغريبة مع الحفاظ على علامات الترقيم الأساسية
    text = re.sub(r'[^\w\s\.,!?;:؟٪%0-9ɛɔɲŋƐƆƝŊ\-]', ' ', text)
    return text.strip()

def split_into_sentences(text: str):
    """تقسيم النص إلى جمل قصيرة لجودة أعلى"""
    # تقسيم على النقاط وعلامات الاستفهام والتعجب
    sentences = re.split(r'(?<=[\.!?؟])\s+', text)
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # إذا كانت الجملة طويلة جداً نقسمها أكثر
        if len(s) > 180:
            parts = re.split(r'(?<=[,;:])\s+', s)
            result.extend([p.strip() for p in parts if p.strip()])
        else:
            result.append(s)
    return result

# -------------------- واجهة المستخدم --------------------
uploaded_file = st.file_uploader(
    "ارفع الفيديو (mp4 / mov / mkv / avi)",
    type=["mp4", "mov", "mkv", "avi"]
)

col1, col2 = st.columns(2)

with col1:
    selected_speaker = st.selectbox(
        "اختر الصوت (موصى به للنقاء والوقار)",
        options=speakers_list,
        format_func=lambda x: x.name,
        index=0  # افتراضي Bourama إن وُجد
    )

with col2:
    temperature = st.slider(
        "درجة الاستقرار (أقل = أنقى وأثبت)",
        min_value=0.45,
        max_value=0.85,
        value=0.62,
        step=0.01,
        help="0.55 - 0.68 تعطي أفضل نقاء وثبات بدون قرقرة"
    )

st.info("""
**أفضل الأصوات للوقار والثبات والنقاء:**
- **Bourama** → الأكثر استقراراً ودقة (موصى به بشدة)
- **Ibrahima** → هادئ ومتزن (مثالي للأدباء والمثقفين)
- **Moussa** → نطق واضح جداً
""")

use_translation = st.checkbox("ترجمة تلقائية إلى البامبارا عبر LLM (إذا كان النص بلغة أخرى)", value=False)

api_key = None
if use_translation:
    api_key = st.text_input("مفتاح API (OpenAI / DeepSeek / Grok)", type="password")

# -------------------- زر التشغيل --------------------
if uploaded_file and st.button("ابدأ الدبلجة الاحترافية", type="primary", use_container_width=True):

    with st.status("جاري المعالجة بجودة عالية...", expanded=True) as status:

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)

                # 1. حفظ الفيديو
                input_video = tmp / uploaded_file.name
                with open(input_video, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 2. استخراج الصوت
                st.write("1️⃣ استخراج الصوت من الفيديو...")
                original_audio_path = tmp / "original.wav"
                video_duration = extract_audio(str(input_video), str(original_audio_path))

                # 3. تحويل الكلام إلى نص
                st.write("2️⃣ تحويل الكلام إلى نص (Bambara ASR)...")
                result = asr_model.transcribe_audio(str(original_audio_path))
                original_text = result if isinstance(result, str) else result.get("text", str(result))
                st.write(f"**النص المستخرج:** {original_text[:350]}...")

                # 4. ترجمة (اختياري)
                bambara_text = clean_and_normalize_text(original_text)

                if use_translation and api_key:
                    st.write("3️⃣ ترجمة دقيقة إلى البامبارا (مع الأرقام)...")
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)

                    prompt = f"""Translate the following text into natural, formal, standard Bambara (Bamanankan).
Rules:
- Use correct modern orthography (ɛ ɔ ɲ ŋ)
- Convert all numbers, dates, percentages and times into proper Bambara words
- Keep a dignified, formal and clear style (suitable for educated speakers)
- Return ONLY the Bambara text, nothing else

Text:
{original_text}"""

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.15
                    )
                    bambara_text = clean_and_normalize_text(response.choices[0].message.content.strip())
                    st.write(f"**النص النهائي بالبامبارا:** {bambara_text[:400]}...")

                # 5. توليد الصوت النقي
                st.write("4️⃣ توليد الصوت بالبامبارا (نقي + وقار + ثبات)...")
                sentences = split_into_sentences(bambara_text)

                audio_segments = []
                progress = st.progress(0)

                for i, sentence in enumerate(sentences):
                    if not sentence:
                        continue

                    # إعدادات النقاء والثبات القصوى
                    audio = tts_model.generate_speech(
                        text=sentence,
                        speaker_id=selected_speaker,
                        temperature=temperature,      # منخفضة للنقاء
                        top_k=35,                       # مركّز
                        top_p=0.82,                     # مستقر
                        max_new_audio_tokens=1800,
                        output_filename=None
                    )
                    audio_segments.append(audio)
                    progress.progress((i + 1) / len(sentences))

                if not audio_segments:
                    raise ValueError("لم يتم توليد أي صوت")

                full_audio = np.concatenate(audio_segments)
                tts_path = tmp / "bambara_clean.wav"
                sf.write(str(tts_path), full_audio, 16000)

                # 6. مزامنة المدة
                st.write("5️⃣ مزامنة الصوت مع مدة الفيديو...")
                y, sr = librosa.load(str(tts_path), sr=None)
                y_stretched = time_stretch(y, sr, video_duration)
                stretched_path = tmp / "stretched.wav"
                sf.write(str(stretched_path), y_stretched, sr)

                # 7. دمج الفيديو
                st.write("6️⃣ دمج الصوت النهائي مع الفيديو...")
                output_video = tmp / "bambara_dubbed_professional.mp4"

                video = VideoFileClip(str(input_video))
                audio = AudioFileClip(str(stretched_path))
                final = video.set_audio(audio)
                final.write_videofile(
                    str(output_video),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                    threads=4,
                    verbose=False
                )
                video.close()
                audio.close()
                final.close()

                # النتيجة
                status.update(label="✅ اكتملت الدبلجة بجودة احترافية!", state="complete")
                st.success("تم إنشاء الفيديو المدبلج بصوت نقي وواضح وبالوقار")
                st.video(str(output_video))

                with open(output_video, "rb") as f:
                    st.download_button(
                        label="⬇️ تحميل الفيديو المدبلج (جودة عالية)",
                        data=f,
                        file_name="bambara_professional_dub.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"حدث خطأ: {e}")# 1. إنشاء بيئة
python -m venv venv
source venv/bin/activate          # ويندوز: venv\Scripts\activate

# 2. تثبيت المكتبات
pip install streamlit torch torchaudio transformers soundfile librosa moviepy openai numpy
pip install git+https://github.com/sudoping01/whosper.git
pip install maliba-ai
# أو
pip install git+https://github.com/MALIBA-AI/bambara-tts.git

# 3. تشغيل التطبيق
streamlit run app.py
         st.exception(e)# bambara-w-bay-l-mali
         
A ka fisa dɔɔnin 
