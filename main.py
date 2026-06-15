import sys
import warnings
import logging
import numpy as np
import soundfile as sf
import torch
import tempfile
import gradio as gr

# here I added a fix for Windows so the web server doesn't crash from connection errors
if sys.platform == 'win32':
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        def silence_winerror_10054(func):
            def wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except ConnectionResetError:
                    pass
            return wrapper

        _ProactorBasePipeTransport._call_connection_lost = silence_winerror_10054(
            _ProactorBasePipeTransport._call_connection_lost)

    except Exception:
        pass

# hide the warnings from speechbrain and huggingface to keep my console clean in pycharm
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# import the necessities from the other files so as not to overload main.py
from cfg import modele_afisare, LOGO_BASE64, stil_css, tema_moderna
from utils import schimba_ui_dupa_model, incarca_poza_profil, calculeaza_pesq_tab, calculeaza_sisdr_tab, \
    calculeaza_dnsmos_tab, deseneaza_spectrograma, calculeaza_sim_tab
from procesare_semnale import aplicare_filtrare_zgomot, rees_audio

# ERROR SOLUTION: Import LOADED_MODELS instead of MODEL_CACHE
from procesare_audio import proceseaza_audio, trigger_zip, LOADED_MODELS
from speechbrain.inference.separation import SepformerSeparation
# Import for VAD
from tab_silero import proceseaza_silero, render_silero_tab


# Import DNSMOS package for the real-time laboratory
try:
    from speechmos import dnsmos

    HAS_DNSMOS = True
except ImportError:
    HAS_DNSMOS = False

# transform the pictures into base64 format at the beginning
poza_b64 = incarca_poza_profil("1.jpg")
utcn_b64 = incarca_poza_profil("utcn.png")


# =========================================================================
# FUNCTION FOR THE SSET COMPARATIVE LABORATORY
# =========================================================================
def ruleaza_opt_comparativ(cale_in, model_id):
    if not cale_in:
        return [gr.update(visible=True, value="❌ Eroare: Încarcă un fișier audio!")] + [None] * 9

    try:
        log = "⚙️ Optimizare Zgomot\n"
        data_orig, fs_orig = sf.read(cale_in, dtype='float32')
        if len(data_orig.shape) > 1: data_orig = np.mean(data_orig, axis=1)

        # STAGE 1
        log += "1️⃣ Analiză semnal brut...\n"
        spec_1 = deseneaza_spectrograma(data_orig, fs_orig, "Etapa 1: Mixaj Brut")
        dns_1_txt = "Scor Indisponibil"
        ovrl_orig = 0
        if HAS_DNSMOS:
            try:
                res1 = dnsmos.run(data_orig, fs_orig)
                ovrl_orig = res1['ovrl_mos']
                dns_1_txt = f"Scor Calitate: {ovrl_orig:.2f} / 5.0\n"
                if ovrl_orig < 2.0:
                    dns_1_txt += "⚠️ CRITIC: Semnal înecat în zgomot!"
                else:
                    dns_1_txt += "ℹ️ Calitate inițială acceptabilă."
            except:
                pass

        # STAGE 2
        log += "2️⃣ Aplicare atenuare zgomot (Boost pentru creștere scor)...\n"
        try:
            cale_vad, _ = proceseaza_silero(cale_in)
            if not cale_vad: cale_vad = cale_in
        except Exception:
            cale_vad = cale_in

        data_vad, fs_vad = sf.read(cale_vad, dtype='float32')
        if len(data_vad.shape) > 1: data_vad = np.mean(data_vad, axis=1)

        data_boost = aplicare_filtrare_zgomot(data_vad, fs_vad, intensitate=0.95)

        # Save in the system's temporary folder
        temp_boost = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        cale_boost = temp_boost.name
        temp_boost.close()
        sf.write(cale_boost, data_boost, fs_vad)

        spec_2 = deseneaza_spectrograma(data_boost, fs_vad, "Etapa 2: Zgomot Atenuat")
        dns_2_txt = "Scor Indisponibil"
        if HAS_DNSMOS:
            try:
                res2 = dnsmos.run(data_boost, fs_vad)
                ovrl_boost = res2['ovrl_mos']
                dns_2_txt = f"Scor Calitate: {ovrl_boost:.2f} / 5.0\n"
                if ovrl_boost > ovrl_orig: dns_2_txt += f"📈 Îmbunătățire cu +{(ovrl_boost - ovrl_orig):.2f}!"
            except:
                pass

        # STAGE 3
        log += "3️⃣ Extragere neurală SepFormer...\n"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ERROR SOLUTION: We use LOADED_MODELS for the cache in Advanced Optimization
        if model_id not in LOADED_MODELS:
            nume_folder = model_id.replace("/", "_")
            LOADED_MODELS[model_id] = SepformerSeparation.from_hparams(
                source=model_id,
                savedir=f"pretrained_models/{nume_folder}",
                run_opts={"device": device}
            )

        model = LOADED_MODELS[model_id]

        sig = torch.from_numpy(data_boost).unsqueeze(0)
        sig = rees_audio(sig, fs_vad, 8000)

        with torch.no_grad():
            sources = model.separate_batch(sig)
            sources = sources / torch.max(torch.abs(sources))

        out_tensor = sources[0, :, 0].unsqueeze(0).cpu()
        out_tensor = rees_audio(out_tensor, 8000, 16000)
        out_audio = out_tensor.squeeze(0).numpy()

        # Save isolated voice in the system's temporary folder
        temp_v1 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        cale_v1 = temp_v1.name
        temp_v1.close()
        sf.write(cale_v1, out_audio, 16000)

        spec_3 = deseneaza_spectrograma(out_audio, 16000, "Etapa 3: Voce AI Izolată")

        dns_3_txt = "Scor Indisponibil"
        if HAS_DNSMOS:
            try:
                res3 = dnsmos.run(out_audio, 16000)
                ovrl_final = res3['ovrl_mos']
                dns_3_txt = f"Scor Calitate: {ovrl_final:.2f} / 5.0\n🎯 Calitate finală optimizată!"
            except:
                pass

        log += "✅ Optimizare realizata cu succes!"
        return [log, cale_in, spec_1, dns_1_txt, cale_boost, spec_2, dns_2_txt, cale_v1, spec_3, dns_3_txt]
    except Exception as e:
        return [f"❌ Eroare la procesare: {str(e)}"] + [None] * 9


# === NEW WIENER FILTER FUNCTION ===
def aplica_wiener(fisier_in, alfa):
    import noisereduce as nr
    if not fisier_in: return None, None
    gr.Info("⏳ Se aplică atenuarea zgomotului...", duration=3)
    try:
        y, sr = sf.read(fisier_in, dtype='float32')
        if len(y.shape) > 1: y = np.mean(y, axis=1)  # Mono

        # A robust Wiener simulation using noisereduce (classic DSP)
        y_curat = nr.reduce_noise(y=y, sr=sr, prop_decrease=alfa / 1000.0, stationary=True)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        cale_out = temp_wav.name
        temp_wav.close()
        sf.write(cale_out, y_curat, sr)

        spec = deseneaza_spectrograma(y_curat, sr, "Spectrogramă Curățată")
        gr.Info("✅ Zgomot eliminat!", duration=3)
        return cale_out, spec
    except Exception as e:
        gr.Error(f"Eroare: {str(e)}")
        return None, None


# =========================================================================
# LIST OF PAGES FOR THE SIDE MENU
# =========================================================================
PAGINI = [
    "🎛️ Separare & Procesare",
    "🔀 Generatoare Teste (Zgomot & Mixaj)",
    "🧹 Curățare Zgomot (Wiener)",
    "🔬 Optimizare Avansată",
    "🤖 Evaluare DNSMOS (Blind)",
    "🏆 Evaluare PESQ & ESTOI",
    "📐 Evaluare SI-SDR (Matematic)",
    "🔐 Analiză Similaritate (SIM)",
    "📈 Analiză DSP (Spectrograme)",
    "✂️ Voice Activity Detection (VAD)",
    "🎧 Exemple Preîncărcate",
    "📚 Documentație & Teorie"
]

# ADDED CSS FOR CORPORATE MENU STYLE AND CENTERED NOTIFICATIONS
CSS_COMPLET = stil_css + """
/* Ascunde complet meniul nativ al tab-urilor de sus pentru a lasa doar meniul lateral */
#tabs_ascunse > div:first-child {
    display: none !important;
}

/* Stilizare Titlu Meniu */
.titlu-meniu h3 {
    color: #00d4ff !important;
    margin-top: 10px !important;
    margin-bottom: 20px !important;
    padding-left: 15px;
    border-left: 4px solid #ff0055;
    font-weight: bold;
    letter-spacing: 1px;
}

/* =========================================
   STILIZARE MENIU LATERAL CORPORATE
========================================= */
.meniu-lateral {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.meniu-lateral .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.meniu-lateral input[type="radio"] {
    display: none !important;
}

.meniu-lateral label {
    background: rgba(20, 10, 35, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    color: #a0a0a0 !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    box-sizing: border-box !important;
    letter-spacing: 0.5px !important;
}

.meniu-lateral label:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border-color: rgba(106, 0, 255, 0.5) !important;
    transform: translateX(6px);
}

.meniu-lateral label.selected {
    background: linear-gradient(135deg, rgba(106, 0, 255, 0.85), rgba(255, 0, 85, 0.85)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    font-weight: bold !important;
    box-shadow: 0 6px 20px rgba(106, 0, 255, 0.4) !important;
    transform: translateX(10px);
}

.meniu-lateral > span { display: none !important; }

/* =========================================
   NOTIFICĂRI CENTRATE (TOAST ANIMATION)
========================================= */
.toast-wrap {
    top: 3% !important;
    bottom: auto !important;
    left: 50% !important;
    right: auto !important;
    transform: translateX(-50%) !important;
    z-index: 999999 !important;
    position: fixed !important;
    width: auto !important;
    min-width: 350px !important;
}

.toast-wrap > div {
    animation: slide-down-bounce 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55) forwards !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    background: rgba(20, 10, 35, 0.95) !important;
    backdrop-filter: blur(12px) !important;
    color: white !important;
    margin-top: 10px !important;
    transition: opacity 0.3s ease-out !important; /* Ajută la dispariția fină */
}

@keyframes slide-down-bounce {
    0% {
        transform: translateY(-150px) scale(0.8);
        opacity: 0;
    }
    100% {
        transform: translateY(0) scale(1);
        opacity: 1;
    }
}
"""

# =========================================================================

# Graphical Interface
with gr.Blocks(title="Separator Vocal PSV", theme=tema_moderna, css=CSS_COMPLET) as interfata:
    gr.HTML(f"""
    <div id="splash-screen">
    <img src="{LOGO_BASE64}" class="splash-logo" alt="Logo Proiect">
    <h1 class='titlu-animat splash-titlu'>Separator Vocal</h1>
    <div class='splash-subtitlu'>Aplicație pentru Extragerea Vorbitorilor din Înregistrări Mixte</div>
    <button class="btn-start-anim" onclick="
    let splash = document.getElementById('splash-screen');
    splash.classList.remove('bring-back');
    splash.classList.add('animate-out');
    ">🚀 Începe Analiza</button>
    </div>
    """)

    # FLEXIBLE HEADER
    gr.HTML(f"""
    <div class="header-container">
        <h1 class='titlu-animat titlu-clickabil' title='Înapoi la ecranul de pornire' onclick="
        let splash = document.getElementById('splash-screen');
        splash.classList.remove('animate-out');
        splash.classList.add('bring-back');
        ">
        <img src="{LOGO_BASE64}" class="app-logo" alt="Logo Aplicatie"> Separator Vocal
        </h1>

        <div class="badges-container">
            <div class="badge-wrapper" title="Despre Autor (Mădălin Gavrilaș)" onclick="document.getElementById('btn_ascuns_profil').click()">
                <img src="{poza_b64}" class="header-badge" alt="Autor">
            </div>
            <div class="badge-wrapper" title="Universitatea Tehnică din Cluj-Napoca (Fișa Disciplinei PSV)">
                <a href="https://etti.utcluj.ro/files/Acasa/Site/FiseDisciplina/TstRo/54_20.pdf" target="_blank">
                    <img src="{utcn_b64}" class="header-badge" alt="UTCN">
                </a>
            </div>
        </div>
    </div>
    """)

    with gr.Column(visible=True) as pagina_procesare:
        with gr.Row():
            # ================= LEFT COLUMN: MENU =================
            with gr.Column(scale=2):
                gr.Markdown("### 📌 Meniu Principal", elem_classes="titlu-meniu")
                meniu_nav = gr.Radio(choices=PAGINI, value=PAGINI[0], label="", elem_classes="meniu-lateral",
                                     interactive=True)

            # ================= RIGHT COLUMN: CONTENT =================
            with gr.Column(scale=9):
                with gr.Tabs(elem_id="tabs_ascunse", selected="tab_procesare") as element_tabs:
                    # TAB 1: Processing
                    with gr.TabItem("🎛️ Separare & Procesare", id="tab_procesare"):
                        with gr.Row():
                            with gr.Column(scale=4, elem_classes="contain-box"):
                                gr.Markdown("### 1️⃣ Alege Algoritmul")
                                dropdown_model = gr.Radio(choices=modele_afisare, value="speechbrain/sepformer-whamr",
                                                          label="Model AI Activ")
                                cutie_descriere = gr.Textbox(
                                    value=schimba_ui_dupa_model("speechbrain/sepformer-whamr")[0],
                                    label="Detalii Tehnice", interactive=False, lines=6, elem_classes="text-mare")

                                with gr.Accordion("⚙️ Setări Avansate & Trunchiere", open=False):
                                    gr.Markdown("**Trunchiere Audio (Crop)**")
                                    with gr.Row():
                                        chk_trim = gr.Checkbox(value=False, label="Activează Trunchierea")
                                        trim_start = gr.Number(value=0, label="Start (sec)")
                                        trim_end = gr.Number(value=15, label="Stop (sec)")

                                    gr.Markdown("**Filtrare & Normalizare**")
                                    slider_denoise = gr.Slider(minimum=0.0, maximum=1.0, value=0.65, step=0.05,
                                                               label="Intensitate Denoise")
                                    dropdown_freq = gr.Radio(choices=["8000", "16000", "22050", "44100", "48000"],
                                                             value="8000", label="Frecvență Export (Hz)")
                                    chk_filt_in = gr.Checkbox(value=True, label="Curățare Intrare")
                                    chk_filt_out = gr.Checkbox(value=True, label="Curățare Ieșire")
                                    chk_norm = gr.Checkbox(value=True, label="Normalizare 0 dB")

                            with gr.Column(scale=6, elem_classes="contain-box"):
                                gr.Markdown("### 2️⃣ Încarcă și Procesează")
                                audio_input = gr.Audio(label="Semnal Audio Mixat", sources=["upload", "microphone"],
                                                       type="filepath",
                                                       waveform_options=gr.WaveformOptions(waveform_color="#ff0055"))
                                buton_start = gr.Button("🚀 PORNEȘTE SEPARAREA", variant="primary", size="lg")

                                gr.Markdown("### 3️⃣ Rezultate Extrase")
                                with gr.Row():
                                    audio_out_1 = gr.Audio(label="Sursa 1 (Voce Principală)", type="filepath", interactive=False, waveform_options=gr.WaveformOptions(waveform_color="#00d4ff"))
                                    audio_out_2 = gr.Audio(label="Sursa 2 (Instrumental / Voce 2)", type="filepath", interactive=False, waveform_options=gr.WaveformOptions(waveform_color="#6a00ff"))
                                    audio_out_3 = gr.Audio(label="Sursa 3 (Voce 3)", type="filepath", interactive=False, waveform_options=gr.WaveformOptions(waveform_color="#ff0055"))

                                consola_status = gr.Textbox(label="Status Sistem", interactive=False,
                                                            elem_classes="consola-log", lines=2)

                                buton_zip = gr.Button("📦 Generează Pachet Analiză (ZIP)", visible=False,
                                                      elem_classes="btn-zip")
                                fisier_zip_rezultat = gr.File(label="Descarcă Arhiva", visible=False)
                                buton_zip.click(fn=trigger_zip, inputs=[], outputs=fisier_zip_rezultat)

                    # --- TAB: GENERATORS FOR TESTING ---
                    with gr.TabItem("🔀 Generatoare Teste (Zgomot & Mixaj)", id="tab_mixaj"):
                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 1️⃣ Generator Zgomot Alb (Stress Test)")
                            gr.Markdown(
                                "Adaugă zgomot de fond matematic peste o înregistrare curată (Voce sau Muzică) pentru a testa limitele modelelor SepFormer și Demucs. Un SNR mai mic înseamnă zgomot mai puternic.")

                            with gr.Row():
                                zgomot_in = gr.Audio(label="Încarcă Audio Curat", type="filepath")
                                with gr.Column():
                                    zgomot_slider = gr.Slider(minimum=-5, maximum=20, value=5, step=1,
                                                              label="Raport Semnal-Zgomot (SNR dB)")
                                    gr.Markdown(
                                        "*Sfat: La 20 dB zgomotul abia se aude. La 0 dB zgomotul e la fel de tare ca vocea. La valori negative, zgomotul acoperă vocea.*")
                                    btn_gen_zgomot = gr.Button("🌪️ ADAUGĂ ZGOMOT ALB", variant="secondary")

                            with gr.Row():
                                zgomot_out = gr.Audio(label="Rezultat Poluat (Gata de testat în model)",
                                                      interactive=False)
                                zgomot_status = gr.Textbox(label="Status", interactive=False)

                            btn_gen_zgomot.click(fn=adauga_zgomot_sintetic, inputs=[zgomot_in, zgomot_slider],
                                                 outputs=[zgomot_out, zgomot_status])

                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 2️⃣ Simulare Problema Cocktail Party (Mixaj Voci)")
                            gr.Markdown(
                                "Alege două fișiere separate. Sistemul le va tăia automat la maxim 10 secunde și le va suprapune.")

                            with gr.Row():
                                mix_in_1 = gr.Audio(label="Încarcă Vocea 1", type="filepath")
                                mix_in_2 = gr.Audio(label="Încarcă Vocea 2 (sau Muzică)", type="filepath")

                            btn_mixeaza = gr.Button("🔀 COMBINĂ FIȘIERELE", variant="primary")

                            with gr.Row():
                                mix_out = gr.Audio(label="Mixaj Rezultat", interactive=False)
                                mix_status = gr.Textbox(label="Status", interactive=False)

                            btn_mixeaza.click(fn=mixeaza_fisiere, inputs=[mix_in_1, mix_in_2],
                                              outputs=[mix_out, mix_status])


                    # --- NEW TAB: WIENER FILTER ---
                    with gr.TabItem("🧹 Curățare Zgomot (Wiener)", id="tab_wiener"):
                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 🧹 Algoritm Wiener Adaptiv (Noise Reduction)")
                            gr.Markdown(
                                "Încarcă o voce extrasă din Tab-ul principal sau orice altă înregistrare cu zgomot de fond pentru a o curăța independent.")
                            with gr.Row():
                                wiener_in = gr.Audio(label="Încarcă Audio Zgomotos", type="filepath")
                                wiener_out = gr.Audio(label="Audio Curățat", interactive=False)
                            wiener_slider = gr.Slider(0, 1000, 800, label="Alpha (Prag Agresivitate)")
                            btn_wiener = gr.Button("🚀 APLICĂ FILTRUL WIENER", variant="primary", size="lg")
                            wiener_spec = gr.Image(label="Spectrogramă Post-Procesare", interactive=False)

                        btn_wiener.click(fn=aplica_wiener, inputs=[wiener_in, wiener_slider],
                                         outputs=[wiener_out, wiener_spec])

                    # The rest of the tabs
                    with gr.TabItem("🔬 Optimizare Avansată", id="tab_optimizare"):
                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 🧪 Analiza în Timp Real a Pipeline-ului (Îmbunătățirea Scorurilor Slabe)")
                            gr.Markdown(
                                "Acest modul demonstrează evoluția calității audio per etapă. Dacă ai un semnal inițial unde zgomotul distruge metricile (scor sub 2.0), aceasta metoda va aplica un **filtru de atenuare** înainte de modelul AI, demonstrând vizual și matematic modul în care recuperăm claritatea semnalului.")

                            with gr.Row():
                                lab_in = gr.Audio(label="Încarcă Semnal Zgomotos", type="filepath")
                                lab_model = gr.Radio(choices=modele_afisare, value="speechbrain/sepformer-whamr",
                                                     label="Model AI")

                            lab_btn = gr.Button("🚀 RULEAZĂ FLUXUL DE ÎMBUNĂTĂȚIRE", variant="primary", size="lg")
                            lab_log = gr.Textbox(label="Jurnal de Procesare", interactive=False, lines=2)

                        with gr.Row():
                            with gr.Column(elem_classes="contain-box"):
                                gr.Markdown("#### 1️⃣ Etapa: Semnal Brut")
                                gr.Markdown(
                                    "*Semnalul exact cum a fost înregistrat. Un scor sub 2 arată că semnalul este înecat.*")
                                lab_aud_1 = gr.Audio(interactive=False, label="Audio Brut")
                                lab_dns_1 = gr.Textbox(label="Evaluare Calitate (AI)", interactive=False)
                                lab_spec_1 = gr.Image(interactive=False, label="Spectrogramă Brută")

                            with gr.Column(elem_classes="contain-box"):
                                gr.Markdown("#### 2️⃣ Etapa: Atenuare Zgomot (Boost)")
                                gr.Markdown(
                                    "*Atenuare forțată a zgomotului pentru salvarea metricilor. Scorul crește!*")
                                lab_aud_2 = gr.Audio(interactive=False, label="Audio Curățat")
                                lab_dns_2 = gr.Textbox(label="Evaluare Calitate (Creștere)", interactive=False)
                                lab_spec_2 = gr.Image(interactive=False, label="Spectrogramă Curată")

                            with gr.Column(elem_classes="contain-box"):
                                gr.Markdown("#### 3️⃣ Etapa: Voce Extrasă (SepFormer)")
                                gr.Markdown(
                                    "*Modelul AI extrage vocea din mixajul optimizat. Atingerea calității maxime.*")
                                lab_aud_3 = gr.Audio(interactive=False, label="Vocea Finală")
                                lab_dns_3 = gr.Textbox(label="Evaluare Calitate (Finală)", interactive=False)
                                lab_spec_3 = gr.Image(interactive=False, label="Spectrogramă Finală")

                        lab_btn.click(fn=ruleaza_opt_comparativ, inputs=[lab_in, lab_model],
                                      outputs=[lab_log, lab_aud_1, lab_spec_1, lab_dns_1, lab_aud_2, lab_spec_2,
                                               lab_dns_2, lab_aud_3, lab_spec_3, lab_dns_3])

                    with gr.TabItem("🤖 Evaluare DNSMOS (Blind)", id="tab_dnsmos"):
                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 1️⃣ Semnale pentru Evaluare")
                            gr.Markdown(
                                "Aici aducem doar vocile extrase de AI. **Nu este nevoie de fișiere originale!**")

                            with gr.Row():
                                est_1_dnsmos = gr.Audio(label="Voce Extrasă 1 (Auto)", type="filepath",
                                                        interactive=False)
                                est_2_dnsmos = gr.Audio(label="Voce Extrasă 2 (Auto)", type="filepath",
                                                        interactive=False)
                                est_3_dnsmos = gr.Audio(label="Voce Extrasă 3 (Auto)", type="filepath",
                                                        interactive=False, visible=False)

                            btn_calculeaza_dnsmos = gr.Button("🧮 Calculează Calitatea (DNSMOS)", variant="primary",
                                                              size="lg")

                            with gr.Accordion("ℹ️ Despre Microsoft DNSMOS", open=False):
                                gr.Markdown("""
                                **Microsoft DNSMOS (Deep Noise Suppression Mean Opinion Score)** este o rețea neuronală avansată, antrenată pe seturi masive de date pentru a simula cu precizie modul în care un juriu uman ar evalua calitatea unei înregistrări audio.

                                * **Metrica "Blind" (No-Reference):** Spre deosebire de PESQ sau SI-SDR, DNSMOS NU are nevoie de semnalul original curat pentru comparație. Evaluează strict acustica fișierului extras, fiind instrumentul ideal pentru a valida performanța algორიtmului SepFormer în scenarii din lumea reală, unde referința curată nu există.
                                * **🗣️ SIG (Signal Quality - Note 1 la 5):** Evaluează exclusiv fidelitatea și naturalețea vocii țintă. Penalizează sever momentele în care rețeaua neuronală a distorsionat semnalul, a suprimat frecvențe utile sau a indus un ton "robotic" (artefacte de procesare) vocii în încercarea de a o izola.
                                * **🔇 BAK (Background Noise Quality - Note 1 la 5):** Măsoară eficiența atenuării interferențelor. Un scor BAK apropiat de 5 înseamnă că zgomotul de fundal, reverberația camerei sau vocile celorlalți vorbitori (cross-talk) au fost eliminate complet, lăsând un fundal "mut".
                                * **🎧 OVRL (Overall Quality - Note 1 la 5):** Calitatea globală percepută de ascultător. Dacă obții un scor sub 2.0, semnalul este înecat în zgomot și necesită o etapă de "Boost" (Atenuare) înainte de separare pentru a fi recuperat.
                                """)

                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 2️⃣ Rezultatele Juriului AI")
                            with gr.Row():
                                text_dnsmos = gr.Textbox(label="Raport Detaliat DNSMOS", lines=6, interactive=False,
                                                         elem_classes="text-mare")
                                grafic_dnsmos = gr.Image(label="Grafic Performanță DNSMOS", type="filepath",
                                                         interactive=False)

                            btn_calculeaza_dnsmos.click(fn=calculeaza_dnsmos_tab,
                                                        inputs=[est_1_dnsmos, est_2_dnsmos, est_3_dnsmos],
                                                        outputs=[text_dnsmos, grafic_dnsmos])

                    with gr.TabItem("🏆 Evaluare PESQ & ESTOI", id="tab_pesq"):
                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 1️⃣ Referințe Curate (Ground Truth) vs. Voci Extrase")
                            gr.Markdown(
                                "Fiind o metrică *Full-Reference*, PESQ are nevoie de vocile originale (curate) pentru a le compara cu rezultatele extrase de AI.")

                            with gr.Row():
                                with gr.Column():
                                    ref_1_pesq = gr.Audio(label="Încarcă Ref. Curată Vocea 1", type="filepath")
                                    ref_2_pesq = gr.Audio(label="Încarcă Ref. Curată Vocea 2", type="filepath")
                                    ref_3_pesq = gr.Audio(label="Încarcă Ref. Curată Vocea 3", type="filepath",
                                                          visible=False)
                                with gr.Column():
                                    est_1_pesq = gr.Audio(label="Voce Extrasă 1 (Aduse automat)", type="filepath",
                                                          interactive=False)
                                    est_2_pesq = gr.Audio(label="Voce Extrasă 2 (Aduse automat)", type="filepath",
                                                          interactive=False)
                                    est_3_pesq = gr.Audio(label="Voce Extrasă 3 (Aduse automat)", type="filepath",
                                                          interactive=False, visible=False)

                            btn_calculeaza_pesq = gr.Button("🧮 Calculează Scorul PESQ", variant="primary", size="lg")

                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 2️⃣ Analiză Performanță")
                            with gr.Row():
                                text_pesq = gr.Textbox(label="Raport Detaliat PESQ", lines=6, interactive=False,
                                                       elem_classes="text-mare")
                                grafic_pesq = gr.Image(label="Grafic Performanță PESQ", type="filepath",
                                                       interactive=False)

                            btn_calculeaza_pesq.click(fn=calculeaza_pesq_tab,
                                                      inputs=[ref_1_pesq, ref_2_pesq, ref_3_pesq, est_1_pesq,
                                                              est_2_pesq, est_3_pesq], outputs=[text_pesq, grafic_pesq])

                    with gr.TabItem("📐 Evaluare SI-SDR (Matematic)", id="tab_sisdr"):
                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 1️⃣ Referințe Curate (Ground Truth) vs. Voci Extrase")
                            gr.Markdown(
                                "SI-SDR este o metrică matematică *Full-Reference*. Încarcă referințele curate pentru a calcula puritatea extragerii AI.")

                            with gr.Row():
                                with gr.Column():
                                    ref_1_sisdr = gr.Audio(label="Încarcă Ref. Curată Vocea 1", type="filepath")
                                    ref_2_sisdr = gr.Audio(label="Încarcă Ref. Curată Vocea 2", type="filepath")
                                    ref_3_sisdr = gr.Audio(label="Încarcă Ref. Curată Vocea 3", type="filepath",
                                                           visible=False)
                                with gr.Column():
                                    est_1_sisdr = gr.Audio(label="Voce Extrasă 1 (Aduse automat)", type="filepath",
                                                           interactive=False)
                                    est_2_sisdr = gr.Audio(label="Voce Extrasă 2 (Aduse automat)", type="filepath",
                                                           interactive=False)
                                    est_3_sisdr = gr.Audio(label="Voce Extrasă 3 (Aduse automat)", type="filepath",
                                                           interactive=False, visible=False)

                            btn_calculeaza_sisdr = gr.Button("🧮 Calculează Scorul SI-SDR", variant="primary", size="lg")

                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 2️⃣ Rezultat Puritate Semnal")
                            with gr.Row():
                                text_sisdr = gr.Textbox(label="Raport Detaliat SI-SDR", lines=5, interactive=False,
                                                        elem_classes="text-mare")
                                grafic_sisdr = gr.Image(label="Grafic Performanță SI-SDR", type="filepath",
                                                        interactive=False)

                            btn_calculeaza_sisdr.click(fn=calculeaza_sisdr_tab,
                                                       inputs=[ref_1_sisdr, ref_2_sisdr, ref_3_sisdr, est_1_sisdr,
                                                               est_2_sisdr, est_3_sisdr],
                                                       outputs=[text_sisdr, grafic_sisdr])

                    # --- NEW TAB: SIMILARITY (SIM) ---
                    with gr.TabItem("🔐 Analiză Similaritate (SIM)", id="tab_sim"):
                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 1️⃣ Referințe Curate vs. Voci Extrase")
                            gr.Markdown(
                                "Folosind rețeaua ECAPA-TDNN, acest modul compară *amprentele vocale* (embeddings) pentru a verifica dacă vocea extrasă aparține aceleiași persoane ca referința, fără a fi distorsionată.")

                            with gr.Row():
                                with gr.Column():
                                    ref_1_sim = gr.Audio(label="Încarcă Ref. Curată Vocea 1", type="filepath")
                                    ref_2_sim = gr.Audio(label="Încarcă Ref. Curată Vocea 2", type="filepath")
                                    ref_3_sim = gr.Audio(label="Încarcă Ref. Curată Vocea 3", type="filepath",
                                                         visible=False)
                                with gr.Column():
                                    est_1_sim = gr.Audio(label="Voce Extrasă 1 (Auto)", type="filepath",
                                                         interactive=False)
                                    est_2_sim = gr.Audio(label="Voce Extrasă 2 (Auto)", type="filepath",
                                                         interactive=False)
                                    est_3_sim = gr.Audio(label="Voce Extrasă 3 (Auto)", type="filepath",
                                                         interactive=False, visible=False)

                            btn_calculeaza_sim = gr.Button("🧮 Calculează Similaritatea (SIM)", variant="primary",
                                                           size="lg")

                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### 2️⃣ Rezultat Identitate")
                            with gr.Row():
                                text_sim = gr.Textbox(label="Raport Detaliat SIM", lines=5, interactive=False,
                                                      elem_classes="text-mare")
                                grafic_sim = gr.Image(label="Grafic Similaritate", type="filepath", interactive=False)

                            btn_calculeaza_sim.click(fn=calculeaza_sim_tab,
                                                     inputs=[ref_1_sim, ref_2_sim, ref_3_sim, est_1_sim, est_2_sim,
                                                             est_3_sim],
                                                     outputs=[text_sim, grafic_sim])

                    with gr.TabItem("📈 Analiză DSP (Spectrograme)", id="tab_dsp"):
                        with gr.Row():
                            with gr.Column(scale=1, elem_classes="contain-box"):
                                gr.Markdown("### 📊 Spectrograme Generate")
                                img_spec_orig = gr.Image(label="Mixaj Original (Intrare)", type="filepath",
                                                         interactive=False)
                                img_spec_1 = gr.Image(label="Vocea 1 Izolată", type="filepath", interactive=False,
                                                      visible=False)
                                img_spec_2 = gr.Image(label="Vocea 2 Izolată", type="filepath", interactive=False,
                                                      visible=False)
                                img_spec_3 = gr.Image(label="Vocea 3 Izolată", type="filepath", interactive=False,
                                                      visible=False)

                    with gr.TabItem("✂️ Voice Activity Detection (VAD)", id="tab_vad"):
                        render_silero_tab()

                    with gr.TabItem("🎧 Exemple Preîncărcate", id="tab_exemple"):
                        with gr.Column(elem_classes="contain-box"):
                            gr.Markdown("### Testează rapid cu fișiere demonstrative")
                            gr.Markdown("Apasă pe un exemplu pentru a-l încărca automat în interfața de procesare.")
                            exemple_preincarcate = gr.Examples(
                                examples=[
                                    ["audio/exemplu_2_voci.wav", "speechbrain/sepformer-whamr"],
                                    ["audio/exemplu_3_voci.wav", "speechbrain/sepformer-libri3mix"]
                                ],
                                inputs=[audio_input, dropdown_model],
                                label="Librărie Fișiere de Test"
                            )

                    with gr.TabItem("📚 Documentație & Teorie", id="tab_teorie"):
                        with gr.Row():
                            with gr.Column(scale=1, elem_classes="contain-box"):
                                gr.Markdown("### 🧠 1. Arhitectura de Separare Neurală")
                                gr.Textbox(
                                    label="Rețeaua SepFormer (Separation Transformer)",
                                    value="SepFormer este un model de Inteligență Artificială de tip State-of-the-Art. Spre deosebire de metodele clasice care filtrează doar frecvențe, acest algoritm bazat pe 'Attention Mechanism' învață amprenta vocală unică a fiecărui vorbitor din mixaj. Funcționează folosind blocuri Encoder-Decoder și mascare dinamică pe axa timpului și a frecvenței pentru a extrage precis vocea dorită, ignorând restul semnalelor.",
                                    lines=6, interactive=False, elem_classes="text-mare"
                                )
                                gr.Textbox(
                                    label="Pre-procesare și Post-procesare (DSP Clasic)",
                                    value="Pe lângă rețeaua neurală, aplicația folosește tehnici de prelucrare a semnalelor:\n• Trunchiere: Selectarea eficientă a unui segment audio.\n• Spectral Gating (Denoise): Eliminarea profilului de zgomot constant (fâșâit) analizând porțiunile de liniște absolută.\n• Resampling & Normalizare: Adaptarea frecvenței la 8kHz pentru AI și normalizarea amplitudinii finale la 0 dB (volum maxim fără distorsiune).",
                                    lines=6, interactive=False, elem_classes="text-mare"
                                )
                                gr.Textbox(
                                    label="Analiza DSP (Spectrograme)",
                                    value="Spectrograma este o reprezentare vizuală a spectrului de frecvențe ale unui semnal pe măsură ce acesta variază în timp. Axa X reprezintă timpul, axa Y frecvența, iar culorile indică amplitudinea (energia) la acea frecvență. Ne ajută să validăm vizual dacă AI-ul a scos vocile corect.",
                                    lines=4, interactive=False, elem_classes="text-mare"
                                )

                            with gr.Column(scale=1, elem_classes="contain-box"):
                                gr.Markdown("### ⚖️ 2. Metricile de Evaluare a Calității")
                                gr.Textbox(
                                    label="Evaluare Blind: DNSMOS (Microsoft AI)",
                                    value="DNSMOS (Deep Noise Suppression Mean Opinion Score) este o rețea neuronală Microsoft antrenată să audă și să dea note exact ca un juriu uman. Nu are nevoie de fișierul original (este No-Reference/Blind). Oferă 3 scoruri (între 1.0 și 5.0):\n• SIG (Signal): Cât de naturală și clară este vocea umană.\n• BAK (Background): Cât de bine a fost ștearsă interferența sau zgomotul.\n• OVRL (Overall): Calitatea generală percepută.",
                                    lines=6, interactive=False, elem_classes="text-mare"
                                )
                                gr.Textbox(
                                    label="Evaluare Perceptuală: PESQ (ITU-T P.862)",
                                    value="PESQ (Perceptual Evaluation of Speech Quality) este un algoritm standard de telecomunicații de tip Full-Reference (necesită vocea originală curată pentru comparație). Acesta aliniază semnalele în timp și penalizează denaturările și artefactele audibile pentru om, returnând un scor MOS între 1.0 (denaturat complet) și 4.5 (calitate perfectă/identică).",
                                    lines=5, interactive=False, elem_classes="text-mare"
                                )
                                gr.Textbox(
                                    label="Evaluare Matematică Pură: SI-SDR",
                                    value="SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) măsoară puritatea matematică a semnalului extras (Full-Reference). Fiind 'Scale-Invariant', ignoră complet diferențele de volum dintre estimare și original. Se măsoară în Decibeli (dB). O valoare crescută cu plus (+10 dB, +15 dB) indică o extragere extrem de curată, în timp ce scorurile negative indică distorsiuni fatale.",
                                    lines=5, interactive=False, elem_classes="text-mare"
                                )


    # =========================================================================================
    # BACKEND - FRONTEND LINKS SECTION
    # =========================================================================================

    def schimba_tab_din_meniu(selectie):
        mapare_taburi = {
            "🎛️ Separare & Procesare": "tab_procesare",
            "🔀 Generatoare Teste (Zgomot & Mixaj)": "tab_mixaj",
            "🧹 Curățare Zgomot (Wiener)": "tab_wiener",
            "🔬 Optimizare Avansată": "tab_optimizare",
            "🤖 Evaluare DNSMOS (Blind)": "tab_dnsmos",
            "🏆 Evaluare PESQ & ESTOI": "tab_pesq",
            "📐 Evaluare SI-SDR (Matematic)": "tab_sisdr",
            "🔐 Analiză Similaritate (SIM)": "tab_sim",
            "📈 Analiză DSP (Spectrograme)": "tab_dsp",
            "✂️ Voice Activity Detection (VAD)": "tab_vad",
            "🎧 Exemple Preîncărcate": "tab_exemple",
            "📚 Documentație & Teorie": "tab_teorie"
        }
        return gr.update(selected=mapare_taburi.get(selectie, "tab_procesare"))


    meniu_nav.change(fn=schimba_tab_din_meniu, inputs=[meniu_nav], outputs=[element_tabs])


    def afiseaza_notificare_si_muta(fisier_audio):
        if fisier_audio is not None:
            gr.Info("✅ Fișierul a fost preluat cu succes!", duration=4)
            return gr.update(selected="tab_procesare"), gr.update(value=PAGINI[0])
        return gr.update(), gr.update()


    audio_input.change(fn=afiseaza_notificare_si_muta, inputs=[audio_input], outputs=[element_tabs, meniu_nav])


    def update_vis_evaluare(model_id):
        desc, vis = schimba_ui_dupa_model(model_id)
        # We need 8 returns: 1 (description) + 7 (visibility for the 3rd voice components)
        return desc, vis, vis, vis, vis, vis, vis, vis


    dropdown_model.change(fn=update_vis_evaluare, inputs=dropdown_model,
                          # HERE is the modification: I added ref_3_sim to the end of the list
                          outputs=[cutie_descriere, ref_3_pesq, est_3_pesq, ref_3_sisdr, est_3_sisdr, est_3_dnsmos,
                                   est_3_sim, ref_3_sim])

    buton_start.click(fn=proceseaza_audio,
                      inputs=[audio_input, dropdown_model, slider_denoise, chk_filt_in, chk_filt_out, chk_norm,
                              dropdown_freq, chk_trim, trim_start, trim_end],
                      outputs=[audio_out_1, audio_out_2, audio_out_3, consola_status, img_spec_orig, img_spec_1,
                               img_spec_2, img_spec_3, buton_zip])


    def oglindeste_fisiere(fisier):
        if fisier is not None:
            return gr.update(value=fisier)
        return gr.update(value=None)


    audio_out_1.change(fn=oglindeste_fisiere, inputs=audio_out_1, outputs=est_1_pesq)
    audio_out_2.change(fn=oglindeste_fisiere, inputs=audio_out_2, outputs=est_2_pesq)
    audio_out_3.change(fn=oglindeste_fisiere, inputs=audio_out_3, outputs=est_3_pesq)

    audio_out_1.change(fn=oglindeste_fisiere, inputs=audio_out_1, outputs=est_1_sisdr)
    audio_out_2.change(fn=oglindeste_fisiere, inputs=audio_out_2, outputs=est_2_sisdr)
    audio_out_3.change(fn=oglindeste_fisiere, inputs=audio_out_3, outputs=est_3_sisdr)

    audio_out_1.change(fn=oglindeste_fisiere, inputs=audio_out_1, outputs=est_1_dnsmos)
    audio_out_2.change(fn=oglindeste_fisiere, inputs=audio_out_2, outputs=est_2_dnsmos)
    audio_out_3.change(fn=oglindeste_fisiere, inputs=audio_out_3, outputs=est_3_dnsmos)

    audio_out_1.change(fn=oglindeste_fisiere, inputs=audio_out_1, outputs=est_1_sim)
    audio_out_2.change(fn=oglindeste_fisiere, inputs=audio_out_2, outputs=est_2_sim)
    audio_out_3.change(fn=oglindeste_fisiere, inputs=audio_out_3, outputs=est_3_sim)

    # Footer and Hidden Profile (triggered by the badge click)
    with gr.Column(visible=False) as pagina_profil:
        buton_inapoi = gr.Button("🔙 ÎNAPOI LA APLICAȚIE", elem_classes="contain-box buton-glass")
        with gr.Row():
            with gr.Column(scale=1, elem_classes="contain-box"):
                gr.HTML(
                    f"""
                        <div class="profil-container-centrat">
                        <img src="{poza_b64}" class="profil-avatar" alt="Poza de profil">
                        <div class="profil-nume">Mădălin Gavrilaș</div>
                        <div class="profil-tag">UTCN - ETTI - TST-RO</div>
                        <div>
                        <a href="https://github.com/gavmada26" target="_blank" class="social-btn">📁 GitHub</a>
                        <a href="https://linkedin.com/in/madalingavrilas" target="_blank" class="social-btn">💼 LinkedIn</a>
                        </div>
                        </div>
                        """
                )
            with gr.Column(scale=2, elem_classes="contain-box"):
                gr.HTML(
                    """
                    <div style="font-size: 1.25em; line-height: 1.7; padding: 10px;">
                        <h1 style="margin-top: 0; color: white;">🎙️ Separarea Vocilor Suprapuse</h1>
                        <p style="color: #e0e0e0; margin-bottom: 20px;">
                            <strong>Salut!</strong> Această aplicație a fost creată pentru disciplina <i>Prelucrarea Semnalului Vocal (PSV)</i>. 
                        </p>
                        <p style="color: #d0d0d0; margin-bottom: 15px;">
                            <b>Ce face mai exact proiectul?</b><br>
                            Imaginează-ți o înregistrare în care vorbesc 2 sau 3 persoane în același timp, formând o suprapunere greu de înțeles. Folosind un model de <b>Inteligență Artificială (AI)</b> de ultimă generație, acest program analizează mixajul audio și reușește să extragă vocea fiecărei persoane într-un fișier individual, curat și clar.
                        </p>
                        <p style="color: #d0d0d0;">
                            Aplicația demonstrează cum algoritmii moderni (rețelele neurale de tip Transformer) pot recunoaște <i>amprenta vocală</i>  unică a fiecărui om și o pot izola din zgomotul de fundal sau din ecoul camerei.
                        </p>
                    </div>
                    """
                )

    btn_ascuns_profil = gr.Button("hidden", elem_id="btn_ascuns_profil")

    btn_ascuns_profil.click(fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                            outputs=[pagina_procesare, pagina_profil])
    buton_inapoi.click(fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                       outputs=[pagina_procesare, pagina_profil])

if __name__ == "__main__":
    print("[sistem] aplicatia web a pornit...")
    interfata.queue().launch(server_name="127.0.0.1", inbrowser=True)