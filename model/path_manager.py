# %%
from pathlib import Path

import pandas as pd


class PathManager:
    project_dir = Path("../")
    test_dir = Path("./tests/")
    label_by_encode = {
        "rle": {
            "HL": ["H1", "L1"],
            "HHL": ["H2", "H2", "L1"],
            "HLL": ["H1", "L2", "L2"],
            "LL": ["L2", "L2"],
            "LHH": ["L1", "H2", "H2"],
            "LLH": ["L2", "L2", "H1"],
        },
        "rle_delta": {
            "HL": ["H1", "dL1"],
            "HHL": ["H2", "H2", "dL1"],
            "HLL": ["H1", "dL1", "L1"],
            "LL": ["L2", "L2"],
            "LHH": ["L1", "dH1", "H1"],
            "LLH": ["L2", "L2", "dH1"],
        },
    }
    # https://www.akenotsuki.com/kyookotoba/accent/bumpu.html
    accept = {"tokyo": ["HL", "LH", "LHH", "HLL", ],
              # HLL を許すとHLLになる。むしろ、話者もそうなのか？
              "kinki": ["HL", "HH", "LH", "LL", "LLH", "HLL", "HHL", ]}

    @classmethod
    def data_path(cls, data_type, wav_path="", is_test=False) -> Path:
        """
        wav_path: wav_path じゃなくても良い(どのみち処理するので)
        """
        # 参照したいタイプを渡すとパスを返す
        if is_test:
            project_dir = cls.test_dir
        else:
            project_dir = cls.project_dir
        # NOTE: 予測に失敗した時、近畿のLHH拒否が原因かもしれない
        accepted_types = [
            "original", "downsample", "feature",
            "label_base", "label_rle",
            "axb",
        ]
        assert data_type in accepted_types
        data_path_map = {
            "original": project_dir / "src/audio/output" / wav_path,
            "downsample": project_dir / "model/wav" / wav_path,
            "feature": project_dir / "model/feature" / str(wav_path.split(".")[0]+".npy"),
            "label_base": project_dir / "model/label_base" / str(wav_path.split(".")[0]+".npy"),
            "label_rle": project_dir / "model/label_rle" / str(wav_path.split(".")[0]+".npy"),
        }
        return data_path_map[data_type]

    @classmethod
    def train_test_wav(cls, is_test=False):
        if is_test:
            tone_df = pd.read_csv(cls.test_dir/"src/list/axb_list.csv")\
                .query("type=='filler'")[["a", "x", "b"]]
        else:
            tone_df = pd.read_csv(cls.project_dir/"src/list/axb_list.csv")\
                .query("type=='filler'")[["a", "x", "b"]]
        wav_list_set = set(
            tone_df.a.values.tolist() +
            tone_df.x.values.tolist() +
            tone_df.b.values.tolist()
        )
        # _を含むのはH_Lのようなテストデータ
        train_wav_list = list(
            filter(lambda s: s.count("_") == 0,
                   wav_list_set)
        )
        test_wav_list = list(filter(lambda s: s.count("_") != 0, wav_list_set))
        return train_wav_list, test_wav_list
