# import standard libraries for time, memory management, matrix processing and ai
import time
import torch
import gc
import soundfile as sf
import numpy as np
import tempfile
import gradio as gr

# import the main sepformer class from the speechbrain library which actually does the voice separation
from speechbrain.inference.separation import SepformerSeparation

# import my classic DSP functions (resampling and denoise) and the drawing and archiving utilities
from procesare_semnale import rees_audio, aplicare_filtrare_zgomot
from utils import deseneaza_spectrograma, genereaza_arhiva_zip

# Global cache variables so as not to reload the AI model from disk on every click
LOADED_MODELS = {}
ULTIMELE_REZULTATE = {"voci": [None, None, None], "spectrograme": [None, None, None, None]}


# the function that takes everything from the interface, processes the audio and returns it in separated format
def proceseaza_audio(cale_audio, model_id, intensitate, filt_in, filt_out, norm_audio, frecventa_str, chk_trim,
                     trim_start, trim_end):
    global LOADED_MODELS, ULTIMELE_REZULTATE

    if not cale_audio:
        raise gr.Error("Te rog încarcă un fișier audio!")

    start_timp = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sistem] Hardware activ: {device.upper()}")

    try:
        # --- 1. PRE-PROCESSING ---
        audio_data, orig_fs = sf.read(cale_audio, dtype='float32')
        frecventa_aleasa = int(frecventa_str)

        # Truncation
        if chk_trim:
            start_idx = int(trim_start * orig_fs)
            end_idx = int(trim_end * orig_fs)
            end_idx = min(end_idx, len(audio_data))
            if start_idx < end_idx: audio_data = audio_data[start_idx:end_idx]

        # Initial spectrogram
        spec_orig_path = deseneaza_spectrograma(audio_data, orig_fs, "Spectrogramă Mixaj Original")
        ULTIMELE_REZULTATE["spectrograme"][0] = spec_orig_path

        # Pre-AI DSP filtering
        if filt_in:
            audio_data = aplicare_filtrare_zgomot(audio_data, orig_fs, intensitate)

        sig = torch.from_numpy(audio_data).unsqueeze(0) if len(audio_data.shape) == 1 else torch.from_numpy(
            audio_data).transpose(0, 1)

        # ==========================================================
        # --- 2. MUSICAL ENGINE (DEMUCS) ---
        # ==========================================================
        if "demucs" in model_id.lower():
            from demucs.apply import apply_model
            from demucs.pretrained import get_model

            frecventa_model_curent = 44100

            if model_id not in LOADED_MODELS:
                print(f"[sistem] Încarcare model Muzical avansat: {model_id}...")
                LOADED_MODELS[model_id] = get_model('htdemucs').to(device)

            model = LOADED_MODELS[model_id]
            model.eval()

            # Demucs requires Stereo format
            sig_demucs = rees_audio(sig, orig_fs, frecventa_model_curent)
            if sig_demucs.shape[0] == 1:
                sig_demucs = sig_demucs.repeat(2, 1)

            sig_demucs = sig_demucs.unsqueeze(0).to(device)

            with torch.no_grad():
                # Extracting the instruments
                sources_demucs = apply_model(model, sig_demucs, shifts=1, split=True, overlap=0.25)[0]

            # sources_demucs contains 4 sources: [Drums, Bass, Other, Vocals]. We convert them to Mono.
            sources_mono = sources_demucs.mean(dim=1)

            voce = sources_mono[3]
            instrumental = sources_mono[0] + sources_mono[1] + sources_mono[2]

            sources = torch.stack([voce, instrumental], dim=1).unsqueeze(0)

        # ==========================================================
        # --- 3. VOCAL ENGINE (SEPFORMER) ---
        # ==========================================================
        else:
            frecventa_model_curent = 8000

            if model_id not in LOADED_MODELS:
                nume_folder = model_id.replace("/", "_")
                cale_salvare = f"pretrained_models/{nume_folder}"
                print(f"[sistem] Încarcare model Vocal: {model_id}...")
                LOADED_MODELS[model_id] = SepformerSeparation.from_hparams(
                    source=model_id, savedir=cale_salvare, run_opts={"device": device}
                )

            model = LOADED_MODELS[model_id]

            # SepFormer requires Mono format
            sig_sep = rees_audio(sig, orig_fs, frecventa_model_curent)
            if sig_sep.shape[0] > 1: sig_sep = sig_sep.mean(dim=0, keepdim=True)

            with torch.no_grad():
                sources = model.separate_batch(sig_sep)
                sources = sources / torch.max(torch.abs(sources))

        # --- 4. POST-PROCESSING AND EXPORT ---
        num_sources = sources.shape[2]
        rezultate_audio = []
        rezultate_spectrograme = [None, None, None]

        for i in range(num_sources):
            out_tensor = sources[0, :, i].unsqueeze(0)

            if frecventa_aleasa != frecventa_model_curent:
                out_tensor = rees_audio(out_tensor, frecventa_model_curent, frecventa_aleasa)

            out_audio = out_tensor.squeeze(0).detach().cpu().numpy()

            if filt_out:
                out_audio = aplicare_filtrare_zgomot(out_audio, frecventa_aleasa, intensitate)

            if norm_audio:
                max_val = np.max(np.abs(out_audio))
                if max_val > 0: out_audio = out_audio / max_val

            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            nume_fisier = temp_wav.name
            temp_wav.close()
            sf.write(nume_fisier, out_audio, frecventa_aleasa)
            rezultate_audio.append(nume_fisier)

            eticheta = f"Sursa {i + 1}"
            if "demucs" in model_id.lower():
                eticheta = "Voce Versuri" if i == 0 else "Instrumental Negativ"

            nume_spec = deseneaza_spectrograma(out_audio, frecventa_aleasa, f"Spectrogramă {eticheta}")
            rezultate_spectrograme[i] = nume_spec

        durata_procesare = round(time.time() - start_timp, 2)
        durata_audio_secunde = len(audio_data) / orig_fs if orig_fs > 0 else 1
        rtf = round(durata_procesare / durata_audio_secunde, 2)

        mesaj_status = f"✅ Procesare în {durata_procesare}s (RTF: {rtf}x).\n▶ Model: {model_id} | Surse: {num_sources}"

        viz_audio = [gr.update(visible=False, value=None)] * 3
        viz_spec = [gr.update(visible=False, value=None)] * 3

        ULTIMELE_REZULTATE["voci"] = [None, None, None]
        ULTIMELE_REZULTATE["spectrograme"][1:] = [None, None, None]

        for i in range(num_sources):
            # Changing the audio label directly from the backend for a cleaner display
            lbl = "Versuri" if "demucs" in model_id and i == 0 else (
                "Instrumental" if "demucs" in model_id and i == 1 else f"Sursa {i + 1}")

            viz_audio[i] = gr.update(visible=True, value=rezultate_audio[i], label=lbl)
            viz_spec[i] = gr.update(visible=True, value=rezultate_spectrograme[i])

            ULTIMELE_REZULTATE["voci"][i] = rezultate_audio[i]
            ULTIMELE_REZULTATE["spectrograme"][i + 1] = rezultate_spectrograme[i]

        return viz_audio[0], viz_audio[1], viz_audio[2], mesaj_status, gr.update(value=spec_orig_path), viz_spec[0], \
            viz_spec[1], viz_spec[2], gr.update(visible=True)

    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def trigger_zip():
    global ULTIMELE_REZULTATE
    return genereaza_arhiva_zip(
        ULTIMELE_REZULTATE["voci"][0], ULTIMELE_REZULTATE["voci"][1], ULTIMELE_REZULTATE["voci"][2],
        ULTIMELE_REZULTATE["spectrograme"][0], ULTIMELE_REZULTATE["spectrograme"][1],
        ULTIMELE_REZULTATE["spectrograme"][2], ULTIMELE_REZULTATE["spectrograme"][3]
    )