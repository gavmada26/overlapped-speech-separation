import gradio as gr
import base64
import os

# --- Clean Names for UI ---
modele_afisare = [
    ("SepFormer WHAMR (Zgomot + Ecou)", "speechbrain/sepformer-whamr"),
    ("SepFormer Libri3Mix (3 Vorbitori)", "speechbrain/sepformer-libri3mix"),
    ("SepFormer WSJ02Mix (Calitate Studio)", "speechbrain/sepformer-wsj02mix"),
    ("SepFormer WHAM (Doar Zgomot Urban)", "speechbrain/sepformer-wham"),
    ("SepFormer Libri2Mix (Voci Narative)", "speechbrain/sepformer-libri2mix")
]

descrieri_modele = {
    "speechbrain/sepformer-whamr": (
        "🛡️ MODEL ROBUST (Zgomot + Ecou)\n\n"
        "▶ RECOMANDAT PENTRU: Înregistrări făcute în camere cu ecou puternic (reverberație) sau în medii foarte zgomotoase.\n\n"
        "⚙️ CUM FUNCȚIONEAZĂ: Identifică și elimină simultan zgomotul și ecoul. Izolează vocile principale, curățându-le de interferențe.\n"
        "ARHITECTURĂ: SepFormer cu mecanism de atenție duală."
    ),
    "speechbrain/sepformer-libri3mix": (
        "👥 MODEL TRI-VOCAL (3 Vorbitori)\n\n"
        "▶ RECOMANDAT PENTRU: Fișiere audio complexe unde exact 3 persoane vorbesc simultan.\n\n"
        "⚙️ CUM FUNCȚIONEAZĂ: Demultiplexează un semnal mixat în 3 fluxuri audio distincte pe baza amprentei vocale.\n"
        "ARHITECTURĂ: Transformer multi-sursă (Libri3Mix)."
    ),
    "speechbrain/sepformer-wsj02mix": (
        "🎙️ MODEL STUDIO (Calitate Maximă)\n\n"
        "▶ RECOMANDAT PENTRU: Înregistrări clare unde vocile se suprapun, dar NU există zgomot de fundal sau ecou.\n\n"
        "⚙️ CUM FUNCȚIONEAZĂ: Păstrează fidelitatea și timbrul natural al vocilor izolate la calitate de studio.\n"
        "ARHITECTURĂ: SepFormer clasic (State-of-the-Art)."
    ),
    "speechbrain/sepformer-wham": (
        "☕ MODEL URBAN (Filtrare Zgomot Ambiental)\n\n"
        "▶ RECOMANDAT PENTRU: Înregistrări în spații publice, pe stradă, acoperite de zgomote dinamice (trafic, vânt).\n\n"
        "⚙️ CUM FUNCȚIONEAZĂ: Se concentrează exclusiv pe suprimarea zgomotelor nestaționare.\n"
        "ARHITECTURĂ: Transformer cu suprimare asimetrică a zgomotului."
    ),
    "speechbrain/sepformer-libri2mix": (
        "📖 MODEL NARATIV (Discursuri Lungi)\n\n"
        "▶ RECOMANDAT PENTRU: Separarea a două voci care narează pe perioade lungi de timp.\n\n"
        "⚙️ CUM FUNCȚIONEAZĂ: Excelează în menținerea consistenței separării pe segmente lungi de vorbire.\n"
        "ARHITECTURĂ: SepFormer optimizat temporal."
    )
}

huggingface_models_dir = {
    "speechbrain/sepformer-whamr": "speechbrain/sepformer-whamr",
    "speechbrain/sepformer-libri3mix": "speechbrain/sepformer-libri3mix",
    "speechbrain/sepformer-wsj02mix": "speechbrain/sepformer-wsj02mix",
    "speechbrain/sepformer-wham": "speechbrain/sepformer-wham",
    "speechbrain/sepformer-libri2mix": "speechbrain/sepformer-libri2mix",
    "demucs/htdemucs": "demucs/htdemucs"
}

metrici_disponibile = ["PESQ (Calitate Perceptuală)", "ESTOI (Inteligibilitate)", "SI-SDR (Puritate Matematică)"]

input_dir = 'input_audio'
output_dir = 'output_audio'
directories = {
    'separat': os.path.join(output_dir, 'separat'),
    'vad': os.path.join(output_dir, 'vad'),
    'trim': os.path.join(output_dir, 'trim'),
    'mix': os.path.join(output_dir, 'mix'),
    'plots': os.path.join(output_dir, 'plots')
}

logo_svg_code = """
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#ff0055"/>
<stop offset="50%" stop-color="#6a00ff"/>
<stop offset="100%" stop-color="#00d4ff"/>
</linearGradient>
</defs>
<style>
.ring { transform-origin: 50px 50px; animation: spin 4s linear infinite; }
.ring2 { transform-origin: 50px 50px; animation: spin-rev 6s linear infinite; }
.bar { fill: url(#g); transform-origin: 50px 50px; animation: bounce 0.6s ease-in-out infinite alternate; }
.b1 { animation-delay: -0.4s; } .b2 { animation-delay: -0.2s; }
.b3 { animation-delay: 0s; } .b4 { animation-delay: -0.3s; } .b5 { animation-delay: -0.5s; }
@keyframes spin { 100% { transform: rotate(360deg); } }
@keyframes spin-rev { 100% { transform: rotate(-360deg); } }
@keyframes bounce { 0% { transform: scaleY(0.3); } 100% { transform: scaleY(0.9); } }
</style>
<circle class="ring" cx="50" cy="50" r="46" fill="none" stroke="url(#g)" stroke-width="4" stroke-dasharray="60 15 15 15" stroke-linecap="round"/>
<circle class="ring2" cx="50" cy="50" r="38" fill="none" stroke="url(#g)" stroke-width="2" stroke-dasharray="50 30" stroke-linecap="round" opacity="0.6"/>
<rect class="bar b1" x="24" y="25" width="6" height="50" rx="3"/>
<rect class="bar b2" x="36" y="25" width="6" height="50" rx="3"/>
<rect class="bar b3" x="48" y="25" width="6" height="50" rx="3"/>
<rect class="bar b4" x="60" y="25" width="6" height="50" rx="3"/>
<rect class="bar b5" x="72" y="25" width="6" height="50" rx="3"/>
</svg>
"""
LOGO_BASE64 = "data:image/svg+xml;base64," + base64.b64encode(logo_svg_code.encode('utf-8')).decode('utf-8')

stil_css = """
/* =========================================
1. STILURI GLOBALE & BACKGROUND
========================================= */
body, .gradio-container {
background: linear-gradient(135deg, #0a0514, #1a0b2e, #0f0524, #11051f) !important;
background-size: 400% 400% !important;
animation: gradient-miscare 15s ease infinite !important;
overflow-x: hidden;
}

@keyframes gradient-miscare {
0% { background-position: 0% 50%; }
50% { background-position: 100% 50%; }
100% { background-position: 0% 50%; }
}

/* =========================================
2. TIPOGRAFIE & ELEMENTE TEXT
========================================= */
.titlu-animat {
background: linear-gradient(270deg, #ff0055, #6a00ff, #00d4ff);
background-size: 400% 400%;
-webkit-animation: gradient-miscare 6s ease infinite;
animation: gradient-miscare 6s ease infinite;
color: transparent;
-webkit-background-clip: text;
background-clip: text;
text-align: left;
font-weight: 900;
padding: 5px 0;
font-size: 2.5em;
letter-spacing: 2px;
margin: 0 !important;
}

.titlu-clickabil {
cursor: pointer;
position: relative;
z-index: 100;
}

.app-logo {
height: 1.2em;
vertical-align: middle;
margin-right: 15px;
}

.text-mare textarea {
font-size: 1.15em !important;
line-height: 1.5em !important;
font-weight: 500 !important;
}

.consola-log {
background: #000000 !important;
color: #00d4ff !important;
font-family: 'Courier New', Courier, monospace !important;
border-left: 4px solid #00d4ff !important;
font-weight: bold;
}

/* =========================================
3. CONTAINERE & CARDURI (GLASSMORPHISM)
========================================= */
.contain-box {
background: rgba(20, 10, 35, 0.6) !important;
backdrop-filter: blur(15px) !important;
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
border: 1px solid rgba(255, 255, 255, 0.08) !important;
border-radius: 16px !important;
transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.contain-box:hover {
transform: translateY(-2px);
box-shadow: 0 12px 40px rgba(106, 0, 255, 0.3) !important;
}

/* =========================================
4. BUTOANE
========================================= */
.buton-glass {
color: #ffffff !important;
font-weight: bold !important;
font-size: 1.1rem !important;
margin-bottom: 20px !important;
padding: 10px !important;
text-align: center !important;
}

button.primary {
background: linear-gradient(90deg, #6a00ff, #ff0055) !important;
border: none !important;
color: white !important;
transition: all 0.2s ease !important;
text-transform: uppercase;
font-weight: bold !important;
font-size: 1.1rem !important;
letter-spacing: 1px;
}

button.primary:hover {
transform: scale(1.02);
box-shadow: 0px 0px 25px rgba(255, 0, 85, 0.7) !important;
}

button.btn-zip {
background: linear-gradient(90deg, #00d4ff, #6a00ff) !important;
border: none !important;
color: white !important;
transition: all 0.2s ease !important;
text-transform: uppercase;
font-weight: bold !important;
font-size: 1.1rem !important;
margin-top: 15px !important;
width: 100% !important;
}

button.btn-zip:hover {
transform: scale(1.02);
box-shadow: 0px 0px 25px rgba(0, 212, 255, 0.7) !important;
}

/* =========================================
5. HEADER BADGES (CREDITE ANIMATE)
========================================= */
.header-container { 
display: flex; 
justify-content: space-between; 
align-items: center; 
padding: 10px 20px; 
margin-bottom: 10px;
}

.badges-container { 
display: flex; 
gap: 15px; 
align-items: center; 
}

.badge-wrapper { 
cursor: pointer; 
position: relative; 
}

.header-badge { 
width: 55px; 
height: 55px; 
border-radius: 50%; 
object-fit: cover; 
border: 3px solid transparent; 
box-shadow: 0 0 15px rgba(106,0,255,0.6); 
transition: all 0.3s ease; 
animation: neon-border 2.5s infinite alternate; 
background-color: #0a0514;
}

.header-badge:hover { 
transform: scale(1.15) rotate(5deg); 
}

@keyframes neon-border {
0% { box-shadow: 0 0 5px #00d4ff, inset 0 0 5px #00d4ff; border-color: #00d4ff; }
50% { box-shadow: 0 0 15px #6a00ff, inset 0 0 10px #6a00ff; border-color: #6a00ff; }
100% { box-shadow: 0 0 5px #ff0055, inset 0 0 5px #ff0055; border-color: #ff0055; }
}

/* =========================================
6. FOOTER & PAGINA PROFIL
========================================= */
#btn_ascuns_profil { display: none !important; }

.profil-container-centrat { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }
.profil-avatar { width: 200px; height: 200px; border-radius: 50%; border: 4px solid #6a00ff; object-fit: cover; margin: 0 auto; display: block; box-shadow: 0 0 20px rgba(106, 0, 255, 0.5); }
.profil-nume { text-align: center; font-size: 2rem; font-weight: bold; margin-top: 15px; color: #ffffff; }
.profil-tag { text-align: center; color: #00d4ff; font-size: 1.1rem; margin-bottom: 20px; }

.social-btn { display: inline-block; padding: 10px 20px; margin: 5px; border-radius: 8px; text-decoration: none; color: white !important; font-weight: bold; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); transition: all 0.3s ease; }
.social-btn:hover { background: #6a00ff; box-shadow: 0 0 15px #6a00ff; }

/* =========================================
7. SPLASH SCREEN
========================================= */
#splash-screen {
position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
background: linear-gradient(135deg, #0a0514, #1a0b2e, #0f0524, #11051f); background-size: 400% 400%;
animation: gradient-miscare 12s ease infinite; z-index: 9999; display: flex; flex-direction: column;
align-items: center; justify-content: center; will-change: transform, opacity;
}

.splash-logo { width: 160px; height: 160px; margin-bottom: 20px; filter: drop-shadow(0 0 20px rgba(106, 0, 255, 0.8)); animation: pulse-glow 3s infinite alternate; }
@keyframes pulse-glow { 0% { filter: drop-shadow(0 0 15px rgba(106, 0, 255, 0.5)); transform: scale(0.98); } 100% { filter: drop-shadow(0 0 35px rgba(255, 0, 85, 0.9)); transform: scale(1.02); } }

.splash-titlu { font-size: 5em !important; margin-bottom: 10px !important; cursor: default; }
.splash-subtitlu { color: #00d4ff; font-size: 1.5em; margin-bottom: 50px; font-weight: bold; letter-spacing: 1px; text-align: center; }

.btn-start-anim { background: linear-gradient(90deg, #6a00ff, #ff0055); border: none; color: white; padding: 20px 50px; font-size: 1.5em; font-weight: bold; border-radius: 50px; cursor: pointer; box-shadow: 0 10px 30px rgba(106, 0, 255, 0.5); transition: transform 0.2s ease, box-shadow 0.2s ease; text-transform: uppercase; letter-spacing: 2px; }
.btn-start-anim:hover { transform: scale(1.05) translateY(-5px); box-shadow: 0 15px 40px rgba(255, 0, 85, 0.7); }

/* --- MODIFICARE DISPARITIE CENTRALA AICI (FADE/ZOOM) --- */
#splash-screen.animate-out { 
    animation: fade-out-central 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards; 
    pointer-events: none; 
}
@keyframes fade-out-central { 
    0% { transform: scale(1); opacity: 1; } 
    100% { transform: scale(1.1); opacity: 0; visibility: hidden; } 
}

#splash-screen.bring-back { 
    animation: fade-in-central 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards; 
    pointer-events: auto; 
}
@keyframes fade-in-central { 
    0% { transform: scale(1.1); opacity: 0; visibility: hidden; } 
    100% { transform: scale(1); opacity: 1; visibility: visible; } 
}

/* =========================================
   FIX PENTRU A PREVENI BLOCAREA ZOOM-ULUI PE IMAGINI
========================================= */
/* Ascunde butonul nativ de Fullscreen din Gradio */
button[aria-label="View fullscreen"], 
button[title="View fullscreen"],
.image-button[aria-label="View fullscreen"] {
    display: none !important;
}

/* Previne efectul de click si deschiderea pop-up-ului pe toate imaginile */
.gradio-container .image-frame img,
.gradio-container .image-container img {
    pointer-events: none !important;
}
"""

tema_moderna = gr.themes.Default(primary_hue="purple", neutral_hue="slate", text_size=gr.themes.sizes.text_md)