import os
import sys
import torch
import numpy as np
import soundfile as sf
import torchaudio
import torch.nn.functional as F

# ==============================================================
# 1. RESOLVING PATHS AND REPEATED DOWNLOADS
# ==============================================================
# Automatically get the exact address of the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Force saving AI models directly into your project (one-time download)
CACHE_DIR = os.path.join(BASE_DIR, "modele_ai_descarcate")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

# AI imports (we do them AFTER setting the cache)
from speechbrain.inference.separation import SepformerSeparation
from demucs.apply import apply_model
from demucs.pretrained import get_model


# ==============================================================
# 2. DSP PROCESSING FUNCTIONS (Without clicks and distortions)
# ==============================================================
def rees_audio(tensor_audio, frecventa_curenta, frecventa_dorita):
    if frecventa_curenta == frecventa_dorita:
        return tensor_audio
    transform = torchaudio.transforms.Resample(orig_freq=frecventa_curenta, new_freq=frecventa_dorita)
    return transform(tensor_audio)


def aplicare_fade(audio_np, fs, fade_ms=50):
    fade_len = int((fade_ms / 1000.0) * fs)
    if len(audio_np) < 2 * fade_len: return audio_np
    audio_np[:fade_len] *= np.linspace(0, 1, fade_len)
    audio_np[-fade_len:] *= np.linspace(1, 0, fade_len)
    return audio_np


def normalizare_sigura(audio_np, target_db=-0.5):
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        factor = (10 ** (target_db / 20.0)) / max_val
        return audio_np * factor
    return audio_np


def mixeaza_cu_snr(s1, s2, snr_db=6.0):
    rms_s1 = np.sqrt(np.mean(s1 ** 2) + 1e-8)
    rms_s2 = np.sqrt(np.mean(s2 ** 2) + 1e-8)
    target_rms_s2 = rms_s1 / (10 ** (snr_db / 20.0))
    factor_atenuare = target_rms_s2 / rms_s2
    s12 = s1 + (s2 * factor_atenuare)
    return normalizare_sigura(s12)


def salveaza_fisier_sigur(cale_folder, nume_fisier, date_audio, sample_rate):
    """Salveaza fisierul si verifica daca s-a creat efectiv pe disc"""
    cale_completa = os.path.join(cale_folder, nume_fisier)
    sf.write(cale_completa, date_audio, sample_rate)
    if os.path.exists(cale_completa):
        print(f"  -> Salvat cu succes: {nume_fisier}")
    else:
        print(f"  ❌ EROARE CRITICĂ: Nu s-a putut salva {nume_fisier}!")


def prepara_si_mixeaza(nume_f1, nume_f2, fs_tinta, nume_dir_out, stereo=False):
    cale_dir_out = os.path.join(BASE_DIR, nume_dir_out)
    os.makedirs(cale_dir_out, exist_ok=True)
    print(f"\n📂 Cream fisierele in: {cale_dir_out}")

    data1, fs1 = sf.read(os.path.join(BASE_DIR, nume_f1), dtype='float32')
    data2, fs2 = sf.read(os.path.join(BASE_DIR, nume_f2), dtype='float32')

    if len(data1.shape) > 1: data1 = np.mean(data1, axis=1)
    if len(data2.shape) > 1: data2 = np.mean(data2, axis=1)

    data1 = data1[:10 * fs1]
    data2 = data2[:10 * fs2]

    data1 = aplicare_fade(data1, fs1, fade_ms=50)
    data2 = aplicare_fade(data2, fs2, fade_ms=50)

    t1 = torch.from_numpy(data1).unsqueeze(0)
    t2 = torch.from_numpy(data2).unsqueeze(0)

    t1 = rees_audio(t1, fs1, fs_tinta)
    t2 = rees_audio(t2, fs2, fs_tinta)

    max_len = max(t1.shape[1], t2.shape[1])
    t1 = F.pad(t1, (0, max_len - t1.shape[1]))
    t2 = F.pad(t2, (0, max_len - t2.shape[1]))

    s1 = normalizare_sigura(t1.squeeze(0).numpy())
    s2 = normalizare_sigura(t2.squeeze(0).numpy())
    s12 = mixeaza_cu_snr(s1, s2, snr_db=6.0)

    salveaza_fisier_sigur(cale_dir_out, "s1.wav", s1, fs_tinta)
    salveaza_fisier_sigur(cale_dir_out, "s2.wav", s2, fs_tinta)
    salveaza_fisier_sigur(cale_dir_out, "s12.wav", s12, fs_tinta)

    t_s12 = torch.from_numpy(s12).unsqueeze(0)
    if stereo: t_s12 = t_s12.repeat(2, 1)

    return t_s12, fs_tinta, cale_dir_out


# ==============================================================
# 3. ARTIFICIAL INTELLIGENCE ENGINES
# ==============================================================
def testul_1_voce_zgomot(voce_curata, zgomot):
    print("\n==================================================")
    print("▶ ÎNCEPE SCENARIUL 1: VOCE + ZGOMOT (SepFormer)")
    print("==================================================")
    dir_out = "Scenariul_1_Voce_Zgomot"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s12_tensor, fs, cale_dir_out = prepara_si_mixeaza(voce_curata, zgomot, 8000, dir_out, stereo=False)

    print("⏳ Incarcam AI-ul SepFormer (Va descarca o singura data, apoi va folosi memoria locala)...")
    cale_model = os.path.join(CACHE_DIR, "sepformer-whamr")
    model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr", savedir=cale_model,
                                             run_opts={"device": device})

    with torch.no_grad():
        print("🧠 Procesam separarea retelei neurale...")
        sources = model.separate_batch(s12_tensor)
        sources = sources / torch.max(torch.abs(sources))

    r1 = normalizare_sigura(sources[0, :, 0].squeeze(0).cpu().numpy())
    r2 = normalizare_sigura(sources[0, :, 1].squeeze(0).cpu().numpy())

    salveaza_fisier_sigur(cale_dir_out, "r1.wav", r1, fs)
    salveaza_fisier_sigur(cale_dir_out, "r2.wav", r2, fs)
    print("✅ SCENARIUL 1 A FOST FINALIZAT CU SUCCES!")


def testul_2_voce_muzica(voce_curata, muzica):
    print("\n==================================================")
    print("▶ ÎNCEPE SCENARIUL 2: VOCE + MUZICĂ (HTDemucs_FT)")
    print("==================================================")
    dir_out = "Scenariul_2_Voce_Muzica"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s12_tensor, fs, cale_dir_out = prepara_si_mixeaza(voce_curata, muzica, 44100, dir_out, stereo=True)
    s12_tensor = s12_tensor.unsqueeze(0).to(device)

    print("⏳ Incarcam AI-ul HTDemucs Fine-Tuned (Va descarca o singura data)...")
    model = get_model('htdemucs_ft').to(device)
    model.eval()

    with torch.no_grad():
        print("🧠 Extragem vocile la calitate Studio (Shifts=5, Overlap=0.8)...")
        sources_demucs = apply_model(model, s12_tensor, shifts=5, split=True, overlap=0.8)[0]

    sources_mono = sources_demucs.mean(dim=1)
    r1 = normalizare_sigura(sources_mono[3].cpu().numpy())
    r2 = normalizare_sigura((sources_mono[0] + sources_mono[1] + sources_mono[2]).cpu().numpy())

    salveaza_fisier_sigur(cale_dir_out, "r1.wav", r1, fs)
    salveaza_fisier_sigur(cale_dir_out, "r2.wav", r2, fs)
    print("✅ SCENARIUL 2 A FOST FINALIZAT CU SUCCES!")


if __name__ == "__main__":
    print(f"🚀 SE PORNEȘTE SISTEMUL DE ANALIZĂ 🚀")
    print(f"📌 Director curent: {BASE_DIR}")

    # Your file names (make sure they are named like this in the folder!)
    voce = "voce_vorbita_curata.wav"
    zgomot = "zgomot.wav"
    muzica = "muzica.wav"

    # RUN SCENARIO 1
    if os.path.exists(os.path.join(BASE_DIR, voce)) and os.path.exists(os.path.join(BASE_DIR, zgomot)):
        testul_1_voce_zgomot(voce, zgomot)
    else:
        print(f"\n⚠️ SĂRIT PESTE SCENARIUL 1: Lipsește fișierul '{voce}' sau '{zgomot}' din folder.")

    # RUN SCENARIO 2
    if os.path.exists(os.path.join(BASE_DIR, voce)) and os.path.exists(os.path.join(BASE_DIR, muzica)):
        testul_2_voce_muzica(voce, muzica)
    else:
        print(f"\n⚠️ SĂRIT PESTE SCENARIUL 2: Lipsește fișierul '{voce}' sau '{muzica}' din folder.")