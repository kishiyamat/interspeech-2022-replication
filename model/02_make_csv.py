# %%
import numpy as np
import pandas as pd
import parselmouth
from plotnine import *
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from path_manager import PathManager
from utils import reset_index_by_time, run_length_encode

# %%
check_octave_jump = False
check_clustering = False  # 0か1かで十分分離できる

# %%
train_wav_list, test_wav_list = PathManager.train_test_wav()
data_list = []

for wav_i in train_wav_list + test_wav_list:
    # ファイル名から 1. 音素 2. ピッチラベル 3. 話者を取得
    phoneme, collapsed_pitches, speaker = wav_i.split("-")
    vowels = "".join(collapsed_pitches.split("_"))
    # 母音の数を取得(モーラの数ではない cf. esko)
    n_vowel = len(vowels)
    # pitch_floor と pitch_ceiling は可視化して調整(octave jump対策)
    pitch = parselmouth.Sound(str(PathManager.data_path("downsample", wav_i)))\
        .to_pitch_ac(pitch_floor=60, pitch_ceiling=200)\
        .selected_array['frequency']
    pitch[pitch == 0] = np.nan  # meanの計算/可視化で無視するので0はnanにする
    n_sample = len(pitch)
    time = np.arange(n_sample)  # クラスタリング/可視化で時間の軸が必要になる
    pipe = Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy='mean')),
        ("cluster", KMeans(n_clusters=n_vowel, random_state=0))])
    arr = np.array([pitch > 0, time]).T  # クラスタリングは0/1の方が好都合
    cluster = reset_index_by_time(pipe.fit_predict(arr))
    label = [vowels[cluster_i] for cluster_i in cluster]
    rle_label = [run_length_encode(vowels)[cluster_i] for cluster_i in cluster]
    data = pd.DataFrame({
        "stimuli": [wav_i] * n_sample,
        "is_train": [wav_i in train_wav_list]*n_sample,
        "pitch": pitch,
        "semitone": 12 * np.log(pitch/np.nanmedian(pitch)) / np.log(2),
        "silent": np.isnan(pitch),
        "time": time,
        "cluster": cluster,
        "label": label,
        "rle_cluster": reset_index_by_time(rle_label),
        "rle_label": rle_label,
        "phoneme": [phoneme]*n_sample,
        "collapsed_pitches": [collapsed_pitches]*n_sample,
        "speaker": [speaker]*n_sample,
    })
    data_list.append(data)
    if check_octave_jump or check_clustering:
        g = (
            ggplot(data, aes(x='time', y='semitone'))
            + facet_grid(". ~ cluster")
            + geom_point()
            + labs(x='time', y='semitone')
        )
        print(g)
# %%
data = pd.concat(data_list)
p = (ggplot(data, aes(x='time', y='semitone', color="label", shape="factor(cluster)"))
     + facet_wrap("~ collapsed_pitches")
     + geom_point()
     + labs(x='time', y='semitone'))
p.save(filename='artifacts/tone_by_cluster.png',
       height=8, width=8, units='cm', dpi=1000)

p = (ggplot(data, aes(x='time', y='semitone', color="rle_label", shape="factor(cluster)"))
     + facet_wrap("~ collapsed_pitches")
     + geom_point()
     + labs(x='time', y='semitone'))
p.save(filename='artifacts/tone_by_cluster_rle.png',
       height=8, width=8, units='cm', dpi=1000)

data.to_csv('artifacts/data.csv', index_label=False)

# %%
