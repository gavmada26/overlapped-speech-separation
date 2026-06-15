# import necessary libraries for working with files, math, and generating pictures
import os
import base64
import zipfile
import numpy as np
import soundfile as sf
import torch
import matplotlib
import tempfile
import gradio as gr

# Import testing and intelligibility libraries
from pesq import pesq
from pystoi import stoi
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F

# Cache for the SIM model so it doesn't download on every click
MODEL_SIM_CACHE = None

# =========================================================================
# FIX FOR MATPLOTLIB WARNING ("divide by zero encountered in log10")
# =========================================================================
import warnings

warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")

# bring in the speechmos package which automatically downloads and runs the Microsoft onnx in the background
try:
    from speechmos import dnsmos

    HAS_DNSMOS = True
except ImportError:
    HAS_DNSMOS = False
    print("[sistem] Avertisment: Pachetul 'speechmos' nu a fost găsit. DNSMOS va fi 0.00!")

# set the matplotlib backend to agg before importing pyplot so it doesn't block my web server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from procesare_semnale import rees_audio
from cfg import descrieri_modele


def schimba_ui_dupa_model(model_id):
    descriere = descrieri_modele.get(model_id, "")
    if "libri3mix" in model_id:
        return descriere, gr.update(visible=True)
    else:
        return descriere, gr.update(visible=False)


def incarca_poza_profil(cale="1.jpg"):
    if os.path.exists(cale):
        with open(cale, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    return "https://dummyimage.com/200x200/6a00ff/ffffff&text=Fara+Poza"


def deseneaza_spectrograma(y, sr, title, filename=None):
    if filename is None:
        filename = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    fig, ax = plt.subplots(figsize=(10, 3), dpi=80)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    cax, freqs, bins, im = ax.specgram(y, Fs=sr, NFFT=512, noverlap=256, cmap='magma')
    fig.colorbar(im, format='%+2.0f dB', ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename


def incarca_audio_robust_16k(cale):
    if not cale: return None
    try:
        data, fs = sf.read(cale, dtype='float32')
        if len(data) == 0: return np.zeros(16000, dtype='float32')
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        data = np.nan_to_num(data)
        if fs != 16000:
            tensor_sig = torch.from_numpy(data).unsqueeze(0)
            tensor_sig = rees_audio(tensor_sig, fs, 16000)
            data = tensor_sig.squeeze(0).cpu().numpy()

        return np.ascontiguousarray(data, dtype=np.float32).copy()
    except Exception:
        return np.zeros(16000, dtype='float32')


# ================= SUPREME ALIGNMENT FUNCTION =================
def sincronizare_si_egalizare_avansata(ref, est):
    # Remove direct current (DC offset)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    min_len = min(len(ref), len(est))
    if min_len == 0: return ref, est

    ref = ref[:min_len]
    est = est[:min_len]

    # Correlation on a large sample for sample-level alignment
    cadru = min(min_len, 16000 * 5)
    corelatie = np.correlate(ref[:cadru], est[:cadru], mode='full')
    lag = np.argmax(corelatie) - (cadru - 1)

    if lag > 0:
        est_aliniat = np.zeros_like(est)
        est_aliniat[lag:] = est[:-lag]
        est = est_aliniat
    elif lag < 0:
        lag = abs(lag)
        ref_aliniat = np.zeros_like(ref)
        ref_aliniat[lag:] = ref[:-lag]
        ref = ref_aliniat

    # RMS Normalization (Identical volume)
    rms_ref = np.sqrt(np.mean(ref ** 2))
    rms_est = np.sqrt(np.mean(est ** 2))

    if rms_est > 0:
        est = est * (rms_ref / rms_est)

    return ref, est


# =====================================================================


def calculeaza_pesq_tab(ref1, ref2, ref3, est1, est2, est3):
    try:
        voci_ref_crawled = [incarca_audio_robust_16k(ref1), incarca_audio_robust_16k(ref2),
                            incarca_audio_robust_16k(ref3)]
        voci_est_crawled = [incarca_audio_robust_16k(est1), incarca_audio_robust_16k(est2),
                            incarca_audio_robust_16k(est3)]

        voci_ref = [v for v in voci_ref_crawled if v is not None and np.sum(np.abs(v)) > 0]
        voci_est = [v for v in voci_est_crawled if v is not None and np.sum(np.abs(v)) > 0]
        num_voci = len(voci_est)

        if num_voci == 0:
            return "⚠️ Nu s-au detectat voci extrase pentru analiză. Rulează mai întâi separarea.", None

        raport = f"✅ Analiză Perceptuală PESQ și ESTOI ({num_voci} Voci):\n\n"

        fig = plt.figure(figsize=(12, 10), dpi=100)
        fig.patch.set_facecolor('#0a0514')
        gs = fig.add_gridspec(2, num_voci)

        culori = ["#00d4ff", "#6a00ff", "#ff0055"]

        for i in range(num_voci):
            r_brut = voci_ref[i]
            e_brut = voci_est[i]

            r, e = sincronizare_si_egalizare_avansata(r_brut, e_brut)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = pesq(16000, r, e, 'wb')
                    val = min(max(val, 1.0), 4.5)
            except:
                val = 1.0

            # NEW CALCULATION: ESTOI
            try:
                val_estoi = stoi(r, e, 16000, extended=True)
            except:
                val_estoi = 0.0

            raport += f"🔹 Vocea {i + 1}:\n"
            raport += f"   • Scor PESQ = {val:.2f} / 4.5 (Calitate)\n"
            raport += f"   • Scor ESTOI = {val_estoi:.2f} (Inteligibilitate)\n\n"

            ax_bar = fig.add_subplot(gs[0, i])
            ax_bar.set_facecolor('#140a23')

            ax_bar.bar(["Scor PESQ"], [val], color=culori[i], width=0.5)
            ax_bar.set_ylim(0, 5)
            ax_bar.set_ylabel("Valoare PESQ (ITU-T P.862)", color='#a0a0a0')
            ax_bar.set_title(f"Vocea {i + 1} (Fidelitate)", color='white')
            ax_bar.tick_params(colors='#a0a0a0')
            ax_bar.text(0, val + 0.2, f"{val:.2f}", ha='center', color='white', fontweight='bold')

            ax_err = fig.add_subplot(gs[1, i])
            S_ref = np.abs(plt.mlab.specgram(r, NFFT=512, Fs=16000)[0])
            S_est = np.abs(plt.mlab.specgram(e, NFFT=512, Fs=16000)[0])

            eroare = np.abs(10 * np.log10(S_ref + 1e-6) - 10 * np.log10(S_est + 1e-6))

            im = ax_err.imshow(eroare, aspect='auto', origin='lower', cmap='magma',
                               extent=[0, len(r) / 16000, 0, 8000])
            ax_err.set_title(f"Harta Distorsiunii V{i + 1}", color='#ff0055', fontsize=10)
            ax_err.set_xlabel("Timp (s)", color='#a0a0a0')
            if i == 0: ax_err.set_ylabel("Frecvență (Hz)", color='#a0a0a0')
            ax_err.tick_params(colors='#a0a0a0')

        fig.tight_layout(pad=3.0)
        cale_grafic = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(cale_grafic, facecolor='#0a0514')
        plt.close(fig)

        return raport, cale_grafic
    except Exception as e:
        return f"❌ Eroare la generarea analizei: {str(e)}", None


# ================= PURE MATHEMATICAL SI-SDR CALCULATION (WITHOUT TORCHMETRICS) =================
def calcul_sisdr_matematic(ref, est, eps=1e-8):
    # 1. Remove the DC component
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    # 2. Calculate the scale-invariant projection (s_target)
    ref_energy = np.sum(ref ** 2) + eps
    alpha = np.sum(ref * est) / ref_energy
    s_target = alpha * ref

    # 3. Calculate the error / noise (e_noise)
    e_noise = est - s_target

    # 4. The logarithmic ratio
    target_energy = np.sum(s_target ** 2)
    noise_energy = np.sum(e_noise ** 2) + eps

    return 10 * np.log10((target_energy + eps) / noise_energy)


def calculeaza_sisdr_tab(ref1, ref2, ref3, est1, est2, est3):
    try:
        voci_ref_crawled = [incarca_audio_robust_16k(ref1), incarca_audio_robust_16k(ref2),
                            incarca_audio_robust_16k(ref3)]
        voci_est_crawled = [incarca_audio_robust_16k(est1), incarca_audio_robust_16k(est2),
                            incarca_audio_robust_16k(est3)]

        voci_ref = [v for v in voci_ref_crawled if v is not None and np.sum(np.abs(v)) > 0]
        voci_est = [v for v in voci_est_crawled if v is not None and np.sum(np.abs(v)) > 0]
        num_voci = len(voci_est)

        if num_voci == 0:
            return "⚠️ Nu s-au detectat referințe sau voci extrase pentru analiză. Te rog încarcă fișierele originale sus.", None

        raport = f"✅ Analiză Matematică SI-SDR ({num_voci} Voci):\n\n"

        scoruri_sisdr = []
        nume_voci = []
        culori = ["#00d4ff", "#6a00ff", "#ff0055"]

        for i in range(num_voci):
            r_brut = voci_ref[i]
            e_brut = voci_est[i]

            # Temporally align the signals so that SI-SDR can correctly judge the purity
            r, e = sincronizare_si_egalizare_avansata(r_brut, e_brut)

            try:
                # Call our robust formula instead of the broken library
                val_sisdr = calcul_sisdr_matematic(r, e)
            except Exception as ex:
                print(f"[eroare SI-SDR Vocea {i + 1}] {ex}")
                val_sisdr = 0.0

            scoruri_sisdr.append(val_sisdr)
            nume_voci.append(f"Vocea {i + 1}")

            raport += f"🔹 Vocea {i + 1}:\n"
            raport += f"   • SI-SDR (Puritate Semnal): {val_sisdr:.2f} dB\n\n"

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.set_facecolor('#0a0514')
        fig.patch.set_facecolor('#0a0514')

        bare = ax.bar(nume_voci, scoruri_sisdr, color=culori[:num_voci], width=0.4)

        # Correctly scale the axes
        lim_sus = max(max(scoruri_sisdr) + 5, 20) if scoruri_sisdr else 20
        lim_jos = min(min(scoruri_sisdr) - 2, 0) if scoruri_sisdr else 0
        ax.set_ylim(lim_jos, lim_sus)

        ax.set_ylabel("Scor SI-SDR (dB)", color='#a0a0a0')
        ax.set_title("Puritatea Matematică a Semnalului Extras", color='white', fontsize=12)
        ax.tick_params(colors='#a0a0a0')
        ax.grid(axis='y', linestyle='--', alpha=0.2)

        for bara in bare:
            inaltime = bara.get_height()
            ax.text(bara.get_x() + bara.get_width() / 2., inaltime + 0.5 if inaltime >= 0 else inaltime - 1.5,
                    f'{inaltime:.2f} dB', ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

        plt.tight_layout()
        cale_grafic = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(cale_grafic, facecolor='#0a0514')
        plt.close(fig)

        return raport, cale_grafic
    except Exception as e:
        return f"❌ Eroare la calcul: {str(e)}", None


# ================= NEW FUNCTION FOR DNSMOS =================
def calculeaza_dnsmos_tab(est1, est2, est3):
    if not HAS_DNSMOS:
        return "❌ Eroare: Pachetul 'speechmos' nu este instalat. Te rog rulează 'pip install speechmos' în terminal.", None

    try:
        voci_est_crawled = [incarca_audio_robust_16k(est1), incarca_audio_robust_16k(est2),
                            incarca_audio_robust_16k(est3)]
        voci_est = [v for v in voci_est_crawled if v is not None and np.sum(np.abs(v)) > 0]
        num_voci = len(voci_est)

        if num_voci == 0:
            return "⚠️ Nu s-au detectat voci extrase pentru analiză. Rulează separarea mai întâi.", None

        raport = f"✅ Analiză DNSMOS (Juriu AI Fără Referință) - {num_voci} Voci:\n\n"

        nume_voci = []
        scoruri_sig = []
        scoruri_bak = []
        scoruri_ovrl = []

        for i in range(num_voci):
            e = np.ascontiguousarray(voci_est[i], dtype=np.float32).copy()
            e = e + 1e-7 * np.random.randn(*e.shape).astype(np.float32)
            e = np.nan_to_num(e)
            e = np.clip(e, -1.0, 1.0)

            if len(e) < 16000 * 2:
                e = np.pad(e, (0, max(0, 16000 * 2 - len(e))))

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rezultat = dnsmos.run(e, 16000)
                    sig = rezultat['sig_mos']
                    bak = rezultat['bak_mos']
                    ovrl = rezultat['ovrl_mos']
            except Exception as ex:
                sig, bak, ovrl = 1.0, 1.0, 1.0
                raport += f"⚠️ Eroare sistem la Vocea {i + 1}: {str(ex)}\n"

            nume_voci.append(f"Vocea {i + 1}")
            scoruri_sig.append(sig)
            scoruri_bak.append(bak)
            scoruri_ovrl.append(ovrl)

            raport += f"🔹 Vocea {i + 1}:\n"
            raport += f"   • Claritate Voce (SIG): {sig:.2f} / 5.0\n"
            raport += f"   • Suprimare Zgomot (BAK): {bak:.2f} / 5.0\n"
            raport += f"   • Calitate Generală (OVRL): {ovrl:.2f} / 5.0\n\n"

        fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
        ax.set_facecolor('#0a0514')
        fig.patch.set_facecolor('#0a0514')

        x = np.arange(num_voci)
        width = 0.25

        rects1 = ax.bar(x - width, scoruri_sig, width, label='Voce (SIG)', color='#00d4ff')
        rects2 = ax.bar(x, scoruri_bak, width, label='Zgomot (BAK)', color='#ff0055')
        rects3 = ax.bar(x + width, scoruri_ovrl, width, label='General (OVRL)', color='#6a00ff')

        ax.set_ylabel('Scor MOS (1-5)', color='#a0a0a0')
        ax.set_title('Evaluare Calitate Acustică (Microsoft DNSMOS)', color='white', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(nume_voci, color='white', fontweight='bold')
        ax.set_ylim(1, 5)

        legend = ax.legend(facecolor='#140a23', edgecolor='none')
        for text in legend.get_texts():
            text.set_color("white")

        ax.tick_params(colors='#a0a0a0')
        ax.grid(axis='y', linestyle='--', alpha=0.2)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='white',
                            fontsize=9, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        plt.tight_layout()
        cale_grafic = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(cale_grafic, facecolor='#0a0514')
        plt.close(fig)

        return raport, cale_grafic
    except Exception as e:
        return f"❌ Eroare la calcul: {str(e)}", None


def genereaza_arhiva_zip(v1, v2, v3, s_orig, s1, s2, s3):
    fisiere_de_arhivat = [
        ("voce_separata_1.wav", v1), ("voce_separata_2.wav", v2), ("voce_separata_3.wav", v3),
        ("spectrograma_original.png", s_orig), ("spectrograma_voce_1.png", s1),
        ("spectrograma_voce_2.png", s2), ("spectrograma_voce_3.png", s3)
    ]
    nume_arhiva = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(nume_arhiva, 'w') as arhiva:
        for nume_in_arhiva, cale_fisier in fisiere_de_arhivat:
            if cale_fisier and os.path.exists(cale_fisier):
                arhiva.write(cale_fisier, arcname=nume_in_arhiva)
    return gr.update(value=nume_arhiva, visible=True)


def afiseaza_notificare_si_muta_tab(fisier_audio):
    if fisier_audio is not None:
        gr.Info("✅ Fișierul a fost preluat cu succes! Ești gata de analiză.", duration=4)
        return gr.update(selected="tab_procesare")
    return gr.update()


# ================= SIM CALCULATION (SPEAKER SIMILARITY) =================
def calculeaza_sim_tab(ref1, ref2, ref3, est1, est2, est3):
    global MODEL_SIM_CACHE
    try:
        voci_ref_crawled = [incarca_audio_robust_16k(ref1), incarca_audio_robust_16k(ref2),
                            incarca_audio_robust_16k(ref3)]
        voci_est_crawled = [incarca_audio_robust_16k(est1), incarca_audio_robust_16k(est2),
                            incarca_audio_robust_16k(est3)]

        voci_ref = [v for v in voci_ref_crawled if v is not None and np.sum(np.abs(v)) > 0]
        voci_est = [v for v in voci_est_crawled if v is not None and np.sum(np.abs(v)) > 0]
        num_voci = len(voci_est)

        if num_voci == 0:
            return "⚠️ Nu s-au detectat voci extrase pentru analiză.", None

        if MODEL_SIM_CACHE is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            MODEL_SIM_CACHE = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                             savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                                             run_opts={"device": device})

        raport = f"✅ Analiză Identitate Vorbitor SIM ({num_voci} Voci):\n\n"
        scoruri_sim = []
        nume_voci = []
        culori = ["#00d4ff", "#6a00ff", "#ff0055"]

        for i in range(num_voci):
            r = torch.from_numpy(voci_ref[i]).unsqueeze(0)
            e = torch.from_numpy(voci_est[i]).unsqueeze(0)

            with torch.no_grad():
                emb_ref = MODEL_SIM_CACHE.encode_batch(r)
                emb_est = MODEL_SIM_CACHE.encode_batch(e)
                sim_score = F.cosine_similarity(emb_ref.squeeze(1), emb_est.squeeze(1)).item()

            # Simple normalization to ensure a [0, 1] range
            sim_score = max(0.0, min(1.0, sim_score))
            scoruri_sim.append(sim_score)
            nume_voci.append(f"Vocea {i + 1}")

            raport += f"🔹 Vocea {i + 1}:\n"
            raport += f"   • Similaritate Identitate (SIM): {sim_score:.2f} / 1.00\n\n"

        # Graph Generation
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.set_facecolor('#0a0514')
        fig.patch.set_facecolor('#0a0514')

        bare = ax.bar(nume_voci, scoruri_sim, color=culori[:num_voci], width=0.4)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Cosine Similarity (0 - 1)", color='#a0a0a0')
        ax.set_title("Păstrarea Identității Vorbitorului (SIM)", color='white', fontsize=12)
        ax.tick_params(colors='#a0a0a0')
        ax.grid(axis='y', linestyle='--', alpha=0.2)

        for bara in bare:
            inaltime = bara.get_height()
            ax.text(bara.get_x() + bara.get_width() / 2., inaltime + 0.05,
                    f'{inaltime:.2f}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

        plt.tight_layout()
        cale_grafic = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(cale_grafic, facecolor='#0a0514')
        plt.close(fig)

        return raport, cale_grafic
    except Exception as e:
        return f"❌ Eroare la calcul: {str(e)}", None