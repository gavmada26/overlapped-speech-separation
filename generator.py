import numpy as np
import soundfile as sf


def generator(nume_fisier="zgomot_pur.wav", durata_secunde=10, sample_rate=44100):
    print(f"⏳ Se generează {durata_secunde} secunde de zgomot alb...")

    # 1. Generate the mathematical noise (random values with normal distribution)
    zgomot = np.random.normal(0, 1, int(sample_rate * durata_secunde))

    # 2. Normalize the signal to prevent distortion (clipping)
    max_val = np.max(np.abs(zgomot))
    if max_val > 0:
        zgomot = zgomot / max_val

    # 3. Reduce its volume to 50% so it isn't deafening on the first listen
    zgomot = zgomot * 0.5

    # 4. Save the physical file to disk
    sf.write(nume_fisier, zgomot, sample_rate)
    print(f"✅ Gata! Fișierul a fost salvat cu succes sub numele: '{nume_fisier}'")


if __name__ == "__main__":
    # You can modify the name or duration here
    generator(nume_fisier="zgomot.wav", durata_secunde=15)