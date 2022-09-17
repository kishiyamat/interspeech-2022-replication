from typing import List, Set, Tuple, Union

import numpy as np
import pandas as pd
from hsmmlearn.emissions import AbstractEmissions
from hsmmlearn.hsmm import HSMMModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class GaussianMultivariateMixtureModel(AbstractEmissions, BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.le = LabelEncoder()
        self.gmms = []
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def fit(self, X, y):
        """
        """
        X = self.imputer.fit_transform(X)
        y = self.le.fit_transform(y)
        for lab in self.le.classes_:
            self.gmms.append(GaussianMixture(n_components=self.n_components)
                             .fit(X[lab == y, :]))
        return self

    def predict(self, X):
        X = self.imputer.fit_transform(X)
        return self.le.inverse_transform(self.likelihood(X).argmax(axis=0))

    def likelihood(self, X):
        """
        X: (n_sample, n_features)
        """
        X = self.imputer.fit_transform(X)
        likelihood_arr = np.vstack(  # log-lik->likelihood
            [np.exp(gmm.score_samples(X)) for gmm in self.gmms])
        likelihood_arr += 0.001  # likelihood に0があると尤度のpathが途絶える
        likelihood_arr /= likelihood_arr.sum(axis=0)  # predict_proba
        if likelihood_arr.shape != (len(self.le.classes_), X.shape[0]):
            raise ValueError
        return likelihood_arr


class Model:
    def __init__(self,
                 subj_idx: int,
                 tokyo_kinki_ratio: float,
                 use_semitone: bool,
                 use_duration: bool,
                 use_transition: bool,
                 use_pi: bool,
                 train_ratio: float,
                 tmat_noise_ratio: float,
                 ):
        """[summary]

        Args:
            use_semitone (bool): [特徴量としてsemitoneを利用(被験者の発話ごとに標準化)するか]
            use_duration (bool): [ラベルに持続時間を埋め込むか: HHをH2とするかHHとするか]
            use_transition (bool): [遷移確率を利用するか: P(H->L2)とP(H2->L)を一様とするか]
            tokyo_kinki_ratio (float): [行動実験の要因とされた東京に居住した割合]
        """
        self.use_semitone = use_semitone
        self.use_duration = use_duration
        self.use_transition = use_transition
        self.use_pi = use_pi
        self.tokyo_kinki_ratio = tokyo_kinki_ratio  # tokyoの影響は必ず入る
        self.le = LabelEncoder()
        self.acoustic = GaussianMultivariateMixtureModel(n_components=2)
        self.n_buffer = 1  # 分布を作成する際の拡張上限
        self.smoothing_dur = 0.0001  # laplase
        self.smoothing_tmat = 0.5
        self.subject_id = subj_idx
        self.test_ratio = 1 - train_ratio
        self.tmat_noise_ratio = tmat_noise_ratio

    def df2xy(self, df: pd.DataFrame):
        """[Modelのパラメータに基づいてdfを整形]

        Args:
            df (pd.DataFrame): col でグループされたデータフレーム
                - stimuli と cluster_key(rle_cluster |cluster) をカラムに持つ
            col (str): df を区切る単位を示した列

        Returns:
            List[np.ndarray]: [description]
            X: (n_cluster, n_sample, n_dim)  # 一番外はList
                [
                    [
                        [1, 0],
                        [1, 0],
                        [1, 0],],
                    [
                        [1, 0],
                        [1, 0],],
                    :
                ]
            y: (n_cluster, n_sample)
                [[y, y, y], [y, y],...]
        """
        cluster_key = "rle_cluster" if self.use_duration else "cluster"
        nested_X = []
        nested_y = []
        for _, row in df.groupby(["stimuli", cluster_key]):
            tone = row.semitone if self.use_semitone else row.pitch
            sil = row.silent
            nested_X.append(np.array([tone, sil]).T)
            y = row.rle_label if self.use_duration else row.label
            nested_y.append(y.values)
        return nested_X, nested_y

    def fit(self, nested_X: List[np.ndarray], nested_y: List[np.array]):
        """[summary]

        Args:
            nested_X ([type]): [description]
            nested_y ([type]): [description]

        Returns:
            [type]: [description]
        """
        # X has pitch and sil indicator, y is label
        # 音響モデルの訓練時にresampling
        self._X = np.concatenate(nested_X)
        # resample: 人によって触れてきた音響情報は違う
        self._X = self.acoustic.imputer.fit_transform(self._X)
        self.nested_y = nested_y  # for duration
        self._y = self.le.fit_transform(np.concatenate(nested_y))
        self._X,  _, self._y, _ = train_test_split(
            self._X, self._y, test_size=self.test_ratio, random_state=self.subject_id)
        n_sample = 200
        # https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
        self.hsmm = HSMMModel(emissions=self.acoustic.fit(self._X, self._y),
                              durations=self.duration,
                              tmat=self.tmat,
                              startprob=self.startprob)
        return self

    @property
    def duration(self):
        """[summary]
        [
            [H, 2]
            [L, 3],
            :
        ]
        のような duration arr を作成し、
        そこから duration の確率を ガウス分布で作成、ビニング
        Returns:
            [type]: [description]
        """
        duration_arr = np.array([[y[0], len(y)]
                                for y in self.nested_y], dtype=object)
        max_len = max(duration_arr[:, 1])
        duration_proba = []
        for lab in self.le.classes_:
            X_lab = duration_arr[duration_arr[:, 0] == lab][:, 1:]\
                + self.smoothing_dur
            gm = GaussianMixture(n_components=1).fit(X_lab)
            base = np.arange(1, max_len + self.n_buffer).reshape(-1, 1)
            likelihoods = np.exp(gm.score_samples(base))  # log-lik->likelihood
            duration_proba.append(likelihoods/sum(likelihoods))

        duration_proba = np.vstack(duration_proba)
        return duration_proba

    @property
    def tokyo_pattern(self):
        if self.use_duration:
            return {
                ("H1", "L2"),
                ("L1", "H2"),
                ("L1", "H1"),
                ("H1", "L1"),
            }
        else:
            return {
                "HLL",
                "LHH",
                "LH",
                "HL",
            }

    @property
    def kinki_pattern(self):
        if self.use_duration:
            return {
                ("H1", "L2"),
                ("H2", "L1"),
                ("L2", "H1"),
                ("H1", "L1"),
                ("H1", "H1"),
                ("L1", "H1"),
                ("L1", "L1"),
            }
        else:
            return {
                "HLL",
                "HHL",
                "LLH",
                "HL",
                "HH",
                "LH",
                "LL",
            }

    @property
    def ex_kinki_pattern(self):
        if self.use_duration:
            return {
                ("H2", "L1"),
                ("L2", "H1"),
                ("H1", "L1"),
                ("H1", "H1"),
                ("L1", "H1"),
                ("L1", "L1"),
            }
        else:
            return {
                "HHL",
                "LLH",
                "HL",
                "HH",
                "LH",
                "LL",
            }

    @property
    def _tmat(self):
        K = self.le.classes_
        tmat_tokyo = np.eye(len(K))*0.00001  # 他に遷移しない場合、自身に遷移させる
        for exp in self.pattern2bigram(self.tokyo_pattern):
            src_idx, tgt_idx = self.le.transform(list(exp))
            tmat_tokyo[src_idx, tgt_idx] += 1
        tmat_kinki = np.eye(len(K))*0.00001  # 他に遷移しない場合、自身に遷移させる
        for exp in self.pattern2bigram(self.kinki_pattern):
            src_idx, tgt_idx = self.le.transform(list(exp))
            tmat_kinki[src_idx, tgt_idx] += 1
        tokyo_weight = (self.tokyo_kinki_ratio+1)/2  # 1なら1, 0なら0
        kinki_weight = 1 - tokyo_weight
        tmat = tmat_tokyo*tokyo_weight + tmat_kinki*kinki_weight

        tmat /= tmat.sum(axis=1).reshape(-1, 1)
        return tmat

    @property
    def tmat(self):
        K = self.le.classes_
        if self.use_transition:
            tmat = self._tmat
        else:
            # use_transition = False な場合は遷移を仮定しない
            tmat = np.eye(len(K))*0.00001  # 他に遷移しない場合、自身に遷移させる
            tmat = np.ones_like(tmat)
            tmat /= tmat.sum(axis=1).reshape(-1, 1)

        # 1 + (0.0 -- 1.0)*ratio
        np.random.seed(seed=self.subject_id)
        noise = np.ones_like(tmat)\
            + np.random.rand(len(K), len(K))*self.tmat_noise_ratio
        tmat *= noise
        tmat /= tmat.sum(axis=1).reshape(-1, 1)
        return tmat

    @property
    def pitch_pattern(self):
        """[summary]
            tokyo の場合は 他を許さない
            tokyo==1以外ならkinkiも許す
        """
        if 0 < self.tokyo_kinki_ratio <= 1:
            return self.tokyo_pattern
        elif self.tokyo_kinki_ratio == 0:
            return self.tokyo_pattern | self.kinki_pattern
        elif -1 <= self.tokyo_kinki_ratio < 0:
            return self.kinki_pattern
        else:
            raise NotImplementedError

    @property
    def startprob(self):
        K = self.le.classes_
        if self.use_pi:
            startprob = np.zeros(len(K))
            for exp in self.pitch_pattern:
                startprob[self.le.transform([exp[0]])] += self.smoothing_tmat
        else:
            startprob = np.ones(len(K))
        startprob /= startprob.sum()
        return startprob

    @staticmethod
    def pattern2bigram(patterns: Set[Union[str, Tuple[str]]]) -> Set[tuple]:
        """[観測されるセットを取得して認可と非認可を生成]
        Args:
            patterns (Union[str, List[str]]): [description]
        """
        bipitch = []
        for pattern in patterns:
            for idx, symbol in enumerate(pattern[:-1]):
                bipitch.append((symbol, pattern[idx + 1]))
        return set(bipitch)

    def mora(self, pred) -> int:
        if self.use_duration:
            return sum([int(p[-1]) for p in pred])
        else:
            return len(pred)

    def percept(self, X_flatten: np.array):
        return self.le.inverse_transform(self.hsmm.decode(self.acoustic.imputer.transform(X_flatten)))
