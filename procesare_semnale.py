import torchaudio
import noisereduce as nr

# function for changing the sampling frequency (resampling)
def rees_audio(tensor_audio, frecventa_curenta, frecventa_dorita):
    # if the frequency is already correct, return the intact signal
    if frecventa_curenta == frecventa_dorita:
        return tensor_audio

    # apply the transformation to the new frequency using torchaudio
    transform = torchaudio.transforms.Resample(orig_freq=frecventa_curenta, new_freq=frecventa_dorita)
    return transform(tensor_audio)

# function for background noise removal improved for voice
def aplicare_filtrare_zgomot(audio_numpy, rate, intensitate=0.65):
    # let the library dynamically calculate the fft windows based on the sample rate
    # keep stationary=True to be extremely efficient against white noise
    audio_curat = nr.reduce_noise(
        y=audio_numpy,
        sr=rate,
        prop_decrease=intensitate,
        stationary=True,
        time_mask_smooth_ms=64 # add the minimum error threshold required as a safety measure
    )
    return audio_curat