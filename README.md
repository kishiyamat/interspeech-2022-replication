# interspeech-2022-replication

https://www.isca-speech.org/archive/pdfs/interspeech_2022/kishiyama22_interspeech.pdf

# README

- Create a model for Tokyo dialect and Kinki dialect.
    - two patterns of (3x3) with labels and features as factors.
- Give each model a speech signal
- make predictions
- visualize

Label factors (based on RLE)

- RLE
- RLE + Delta

Feature Factor

- pitch (based on RLE)
- pitch:delta
- delta

Language model

- Tokyo
- Kinki

Experiment

- Test the inference by giving test speech (tone-illusory speech such as `L_H`).

## Supplementation

### Parameters to be passed to the model

HSMM itself is formed by the distribution of initial probability, transition probability, ejection probability, and persistence probability.
distributions of initial probability, transition probability, ejection probability, and persistence probability, each of which is necessary.
In addition, the injection probability is divided into the Gaussian case and the MultivariateGaussian case.
In addition, since the parameters required for the injection probability are different for the Gaussian case and the Multivariate Gaussian case, the following parameters are required for the HSMM
[emmition.py](https://github.com/kishiyamat/hsmmlearn/blob/master/hsmmlearn/emissions.py)
for the parameters required for the Gaussian and MultivarieteGaussian cases.
The explanation of each is given below.

In the Gaussian case
`GaussianEmissions`.

In the case of MultivariateGaussian, use
`MultivariateGaussianEmissions` in the case of Gaussian.

## Citation

```bibtex
@article{kishiyama2022,
  title={One-step models in pitch perception: Experimental evidence from Japanese},
  author={},
  journal={Proc. Interspeech 2022},
  pages={},
  year={2022}
}
```
