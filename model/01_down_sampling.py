# %%
import librosa
import soundfile as sf

from path_manager import PathManager

# %%
if __name__ == "__main__":
    # original の音源をdownsample
    print()
    # %%
    train_wav_list, test_wav_list = PathManager.train_test_wav()
    print(f"train wav files are:\n\t{train_wav_list}")
    print(f"test wav files are:\n\t{test_wav_list}")

    SR = 16_000
    for wav_i in train_wav_list + test_wav_list:
        y, sr = librosa.load(PathManager.data_path("original", wav_i), sr=None)
        y_16k = librosa.resample(y, sr, SR)
        sf.write(PathManager.data_path("downsample", wav_i), y_16k, SR)
