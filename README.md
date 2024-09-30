# Implementierung und Untersuchung der Effekte von Social Influence auf kooperative Mulit-Agenten Systeme

*Die Strukutr dieses Repositories basiert auf dem google-deepmind meltingpot Repository (https://github.com/google-deepmind/meltingpot)*


<div align="center">
  <img src="docs/images/meltingpot_montage.gif"
       alt="Melting Pot substrates"
       height="250" width="250" />
</div>

## Installation

### `pip` install

[Melting Pot ist Verfügbar über PyPI](https://pypi.python.org/pypi/dm-meltingpot)
und kann mittels des folgenden Befehls ausgeführt werden:

```shell
pip install dm-meltingpot
```

NOTE: Melting Pot is built on top of [DeepMind Lab2D](https://github.com/google-deepmind/lab2d)
which is distributed as pre-built wheels. If there is no appropriate wheel for
`dmlab2d`, you will need to build it from source (see
[the `dmlab2d` `README.md`](https://github.com/google-deepmind/lab2d/blob/main/README.md)
for details).

### Manual install

If you want to work on the Melting Pot source code, you can perform an editable
installation as follows:

1.  Clone Melting Pot:

    ```shell
    git clone -b main https://github.com/google-deepmind/meltingpot
    cd meltingpot
    ```

2.  (Optional) Activate a virtual environment, e.g.:

    ```shell
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install Melting Pot:

    ```shell
    pip install --editable .[dev]
    ```

4.  (Optional) Test the installation:

    ```shell
    pytest --pyargs meltingpot
    ```


#### CUDA support

To enable CUDA support (required for GPU training), make sure you have the
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
package installed, and then run Docker with the `---gpus all` flag enabled. Note
that for GitHub Codespaces this isn't necessary, as it's done for you
automatically.

## Example usage

### Evaluation
The [evaluation](meltingpot/utils/evaluation/evaluation.py) library can be used
to evaluate [SavedModel](https://www.tensorflow.org/guide/saved_model)s
trained on Melting Pot substrates.

Instead a program for evaluating agents with the data specified in the Paper accompanying this repo
has been implemented

### Interacting with the substrates

You can try out the substrates interactively with the
[human_players](meltingpot/human_players) scripts. For example, to play
the `clean_up` substrate, you can run:

```shell
python meltingpot/human_players/play_clean_up.py
```

You can move around with the `W`, `A`, `S`, `D` keys, Turn with `Q`, and `E`,
fire the zapper with `1`, and fire the cleaning beam with `2`. You can switch
between players with `TAB`. There are other substrates available in the
[human_players](meltingpot/human_players) directory. Some have multiple
variants, which you select with the `--level_name` flag.

### Training agents

meltingpot provides two example scripts: one using
[RLlib](https://github.com/ray-project/ray), and another using
[PettingZoo](https://github.com/Farama-Foundation/PettingZoo) with
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3). Note
that Melting Pot is agnostic to how you train your agents, and as such, these
scripts are not meant to be a suggestion on how to achieve good scores in the
task suite.

For the approach of the accompanying thesis, a specific script has been implemented,
based on the Pettingzoo example, where it can also be found.

#### PettingZoo and Stable-Baselines3

This example uses a PettingZoo wrapper with a fully parameter shared PPO agent
from SB3.

The PettingZoo wrapper can be used separately from SB3 and
can be found [here](examples/pettingzoo/utils.py).

```shell
cd <meltingpot_root>
pip install -r examples/requirements.txt
cd examples/pettingzoo
python sb3_train_SI.py
```
adjustments for running different trainings, are described in detail in the comments.

## Documentation

Full documentation is available [here](docs/index.md).

## Citing Melting Pot

If you use Melting Pot in your work, please cite the accompanying article:

```bibtex
@inproceedings{leibo2021meltingpot,
    title={Scalable Evaluation of Multi-Agent Reinforcement Learning with
           Melting Pot},
    author={Joel Z. Leibo AND Edgar Du\'e\~nez-Guzm\'an AND Alexander Sasha
            Vezhnevets AND John P. Agapiou AND Peter Sunehag AND Raphael Koster
            AND Jayd Matyas AND Charles Beattie AND Igor Mordatch AND Thore
            Graepel},
    year={2021},
    journal={International conference on machine learning},
    organization={PMLR},
    url={https://doi.org/10.48550/arXiv.2107.06857},
    doi={10.48550/arXiv.2107.06857}
}
```

## Disclaimer

This is not an officially supported Google product.
