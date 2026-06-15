import gradio as gr
import torch
import time
import os

model_vad, utils_vad = None, None


def proceseaza_silero(cale_in):
    global model_vad, utils_vad
    if not cale_in:
        return None, "Încarcă un fișier."

    # Load Silero VAD from PyTorch Hub
    if model_vad is None:
        model_vad, utils_vad = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils_vad

    start_time = time.time()

    # Read audio
    wav = read_audio(cale_in)

    # Get the segments where there is actual speech
    speech_timestamps = get_speech_timestamps(wav, model_vad, sampling_rate=16000)

    # Merge the segments by eliminating the silence
    if len(speech_timestamps) > 0:
        wav_fara_liniste = collect_chunks(speech_timestamps, wav)

        # SOLUTION HERE: Generate a unique name using the exact time to avoid the Content-Length error in Gradio
        timestamp_unic = int(time.time())
        cale_out = f"audio_vad_optimizat_{timestamp_unic}.wav"

        save_audio(cale_out, wav_fara_liniste, sampling_rate=16000)
    else:
        return cale_in, "Nu s-a detectat voce."

    timp_exec = time.time() - start_time

    # Calculate how much time we saved
    durata_orig = len(wav) / 16000
    durata_noua = len(wav_fara_liniste) / 16000
    economie = durata_orig - durata_noua

    mesaj = f"⏱️ Optimizare: \nDurată originală: {durata_orig:.2f}s \nDurată după VAD: {durata_noua:.2f}s \nEconomie timp procesare AI: {economie:.2f} secunde!"

    return cale_out, mesaj


def render_silero_tab():
    with gr.TabItem("✂️ Voice Activity Detection (VAD)", id="tab_vad"):
        gr.Markdown("### Eliminarea inteligentă a liniștii înainte de Separarea Vocală")

        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(label="Audio Original (Pauze lungi)", type="filepath")
                btn_vad = gr.Button("PROCESARE VAD", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Audio Optimizat")
                raport_txt = gr.Textbox(label="Raport Eficiență")

        btn_vad.click(fn=proceseaza_silero, inputs=[audio_in], outputs=[audio_out, raport_txt])