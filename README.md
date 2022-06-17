# interspeech-2021-replication

https://www.isca-speech.org/archive/pdfs/interspeech_2022/kishiyama22_interspeech.pdf

## Prerequisites

- Git
- Docker

## How to replicate the results

Clone this repository

```shell
$ git clone git@github.com:kishiyamat/interspeech-2020-replication.git
$ cd interspeech-2020-replication
```

Run the experiments

```shell
$ # Local terminal 1 @ interspeech-2021-replication
$ docker build -t kishiyamat/interspeech-2021-replication .
$ docker run -it --rm kishiyamat/interspeech-2021-replication bash
$ # Docker terminal
$ cd interspeech-2021-replication
$ make exp1     # dupoux et al. 1999
$ make exp2     # dupoux et al. 2011
$ # keep docker running
```

Copy the results

```shell
$ # Local terminal 2 @ interspeech-2021-replication
$ docker ps
CONTAINER ID        IMAGE                                     COMMAND             CREATED             STATUS              PORTS               NAMES
7609212cd78a        kishiyamat/interspeech-2021-replication   "bash"              9 minutes ago       Up 9 minutes        8787/tcp            sleepy_bell
$ # Use CONTAINER ID to find results
$ docker cp 7609212cd78a:/opt/app/interspeech-2021-replication/artifact/. artifact/
```

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
