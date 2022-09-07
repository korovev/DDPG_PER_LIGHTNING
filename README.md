# Deep Deterministic Policy Gradient (DDPG) with Prioritized Experience Replay
[![](https://shields.io/badge/-nn--template-emerald?style=flat&logo=github&labelColor=gray)](https://github.com/lucmos/nn-template)

**Authors**: [Luca Iezzi](https://github.com/korovev) and [Giulia Ciabatti](https://github.com/Giulia1809).

This consists of a complete reimplementation of [DDPG with PrioritizedExperience Replay](https://arxiv.org/abs/1511.05952), and its adaptation on Pendulum-v1 and MountainCarContinuous-v0, one of the environments from [OpenAI Gym](https://www.gymlibrary.dev/).

|    **Agent playing**   |
|:-----------------------:|
| ![Controller run](readme_images/run.gif)           |

## Usage
This implementation is based on Python 3.8. 
Some of the dependencies include:
	- Pytorch
	- Lightning
To install all the requirements:
```bash
$ pip install -r requirements.txt
```

## 


## Hyperparameters
<table>
<tr><th>VAE </th><th>MDN-RNN</th><th>Controller</th></tr>
<tr><td>

|    **hyperparameter**   |     **value**     |
|:-----------------------:|:-----------------:|
| Num. rollouts           | 1000              |
| Setting                 | easy              |
| Num. levels             | 20                |
| Image size              | (64, 64, 3)       |
| Batch size              | 32                |
| Optimizer               | Adam              |
| Learning rate           | 0.001             |
| Learning rate scheduler | ReduceLROnPlateau |
| Latent size             | 64                |
| Epochs                  | 153               |

</td><td>

|    **hyperparameter**   |     **value**     |
|:-----------------------:|:-----------------:|
| Num. rollouts           | 1000              |
| Setting                 | easy              |
| Num. levels             | 20                |
| Image size              | (64, 64, 3)       |
| Batch size              | 32                |
| Optimizer               | Adam              |
| Learning rate           | 0.001             |
| Learning rate scheduler | ReduceLROnPlateau |
| Latent size             | 64                |
| LSTM hidden units       | 256               |
| Sequence length         | 32                |
| Epochs                  | 147               |

</td><td>

|    **hyperparameter**   |     **value**     |
|:-----------------------:|:-----------------:|
| Image size              | (64, 64, 3)       |
| Setting                 | easy              |
| Num. levels             | 20                |
| Evolution algorithm     | CMA-ES            |
| Learning rate scheduler | ReduceLROnPlateau |
| Latent size             | 64                |
| LSTM hidden units       | 256               |
| Population size         | 64                |
| Num. samples            | 16                |
| Target return           | 20                |
| Evaluation frequency    | 2                 |
| Epochs                  | 300               |

</td>


</tr> </table>


## Running
The complete pipeline to train the 3 model components:

### 1. Generate dataset
First, we generate 1000 rollouts from a random policy:
```bash
$ PYTHONPATH=. python3 src/generate_data.py --rollouts 1000
```
We then split the data in train, validation, and test:
```bash
$ PYTHONPATH=. python3 src/split_data.py
```
Finally, we reorganize the data in order to train the MDN-RNN:
```bash
$ PYTHONPATH=. python3 src/pl_data/new_dataset.py
```

### 2. Train Vision
To train the Vision, set `config_name="vae"` in line 149 of the [run.py](src/run.py) script, then run the following:
```bash
$ PYTHONPATH=. python3 src/run.py
```

### 3. Train Memory
To train the Memory, set `config_name="mdrnn"` in line 149 of the [run.py](src/run.py) script, then run the following:
```bash
$ PYTHONPATH=. python3 src/run.py
```

### 4. Train Controller
To train the Controller, run the following:
```bash
$ PYTHONPATH=. python3 src/controller.py
```

## Credits
We took inspiration from the implementations of [Corentin Tallec](https://github.com/ctallec/world-models) and [Sebastian Risi](https://github.com/sebastianrisi/ga-world-models).
