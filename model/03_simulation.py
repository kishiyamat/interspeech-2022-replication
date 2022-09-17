# %%
import itertools

import numpy as np
import pandas as pd
import rle
from plotnine import *

from models import Model

# %%
data = pd.read_csv('artifacts/data.csv')
train_df = data.query("is_train == True")
test_df = data.query("is_train == False")
test_df["mora"] = test_df.collapsed_pitches.apply(len)
test_df_3mora = test_df.query("mora==3")
print(len(set(train_df.stimuli.values)))
print(len(set(test_df_3mora.stimuli.values)))
print(set(test_df_3mora.stimuli.values))
# %%
# 1. fit model by condition
# 2. make model inference on each stimuli
# 3. 推論結果がtokyo_patternかkinki_patternか
n_subjects = 10  # 20ずつ
use_durations = [True]  # Falseは話にならないのでTrueに限定
use_pi_conds = [True, False]  # topdown の検証用パラメータ
use_transitions = [True, False]  # topdown の検証用パラメータ
use_semitones = [True, False]  # 音の知覚の戦略. Falseは相対音感
tokyo_kinki_ratios = [-1, -0.5, 0, 0.5, 1]
conditions = itertools.product(
    use_semitones,
    use_durations,
    use_transitions,
    use_pi_conds,
    tokyo_kinki_ratios,
)

res = []
for use_semitone, use_duration, use_transition, use_pi, tokyo_kinki_ratio in list(conditions):
    for subj_idx in range(n_subjects):
        model_params = {
            "use_semitone": use_semitone,  # 相対/絶対が不明なので加える
            "use_duration": use_duration,
            "use_transition": use_transition,
            "use_pi": use_pi,
            "tokyo_kinki_ratio": tokyo_kinki_ratio,
            "subj_idx": subj_idx,
            "train_ratio": 0.5,
            "tmat_noise_ratio": 0.1,
        }
        model = Model(**model_params)
        X, y = model.df2xy(train_df)
        model.fit(X, y)
        # make model inference on the stimuli
        for _, df_by_stimuli in test_df_3mora.groupby("stimuli"):
            stimulus = df_by_stimuli.stimuli[0].split('.')[0]
            phoneme, pitch, speaker = stimulus.split("-")
            X, _ = model.df2xy(df_by_stimuli)
            X_flatten = np.concatenate(X)  # 実際の入力は区切られていないのでflatten
            y = model.percept(X_flatten)
            y_collapsed = tuple(rle.encode(y)[0])
            is_tokyo = y_collapsed in model.tokyo_pattern
            is_kinki = y_collapsed in model.ex_kinki_pattern
            n_success = is_tokyo or is_kinki
            res.append(pd.DataFrame(dict(
                tokyo_pref=[is_tokyo - is_kinki],
                subj_id=[subj_idx],
                stimulus=[stimulus],
                phoneme=[phoneme],
                pitch=[pitch],
                speaker=[speaker],
                use_semitone=[use_semitone],
                use_duration=[use_duration],
                use_transition=[use_transition],
                use_pi=[use_pi],
                tokyo_kinki_ratio=[tokyo_kinki_ratio],
                n_success=[n_success],
                pred=["".join(y_collapsed)],
            )))

res_df = pd.concat(res)
conditions = ["use_duration", "use_transition", "use_pi"]
plot_df = res_df.groupby(
    conditions+["tokyo_kinki_ratio", "pitch", "phoneme", "subj_id"]).mean().reset_index()

# transition がなくても右肩上がりの図は再現される...？
# 音響モデルとdurationで、ということになる。
for cond, df_g in plot_df.groupby(conditions):
    print(df_g.head())
    n_success = np.mean(df_g.n_success)
    print(len(df_g))
    print("\n".join(
        [f"c_str: {c_str}, c_bool: {c_bool}" for c_str, c_bool in zip(conditions, cond)]))
    g = (ggplot(df_g, aes(x='factor(tokyo_kinki_ratio)', y='tokyo_pref', color="pitch", fill="pitch"))
         + facet_grid("pitch~phoneme")
         + geom_violin()
         + ylim(-1, 1)
         + ggtitle(f"Properly inferenced: {n_success}")
         )
    print(g)

# TODO
# - 統計用のdfを出力
# - 統計で再現
# %%
res_df.to_csv("./results.csv")
# %%
res_df.head()
# %%
