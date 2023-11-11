# <div align="center"> -- Reinforcement Learning Project -- <br/> Article : Human-level control through deep reinforcement learning</div>

## Introduction

This repository contains the code for the article "Human-level control through deep reinforcement learning" by Volodymyr Mnih et al. (2015). The code is based on the [original implementation](content/article.pdf)
The goal of this project is to reproduce the implementation of the article and to train the model on the Atari game Breakout.

## Installation

The code is written in Python 3.11.5 and uses PyTorch 2.1.0 avec CUDA 12.1. The dependencies are listed in the file `requirements.txt`. To install them, run the following command:

```bash
pip install -r requirements.txt
```

If torch did not recognize your CUDA version, you can force the installation of the correct version with the following command:

```bash
pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

For more information, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

## Usage

To train the model on the Atari game Breakout, run the following command:

```bash
python main.py play Breakout-v4
```

For more information on the arguments, run the following command:

```bash
python main.py --help
```

Output:

```bash
usage: main.py [-h] [--mode MODE] [--difficulty DIFFICULTY] [--repeat REPEAT] [--name NAME] {train,retrain,play} game_name

Game Launcher

positional arguments:
  {train,retrain,play}  Specify 'train', 'retrain' or 'play'
  game_name             Name of the game

options:
  -h, --help            show this help message and exit
  --mode MODE           Game mode
  --difficulty DIFFICULTY
                        Difficulty level
  --repeat REPEAT       Number of hours to train or retrain the model
  --name NAME           Name of the model to retrain or play
```

Warning: The name of the model must not end with `_{NUMBER}`, `_best` or `_last` as these names are reserved for the training process.

## Explanation of the implementation

The pseudo-code of the algorithm is given in the article, but some details are implemented differently in the code. The main difference is the replay memory size is 600000 instead of 1000000. My computer does not have enough memory to store 1000000 transitions in the replay memory.

The training process about the number of steps, episodes or hours is not implemented. In my case, I trained the model with hours because it is easier to estimate the time of training and I needed my computer for other tasks. The training process is stopped when the model is trained for the number of hours specified in the arguments.

The model is saved every hour, it contains the weights of the model and the target model, the optimizer state, the number of steps, episodes and hours of training.<br>
It is saved in the folder `models/game_name` with the name `{NAME}_{SUFFIX}.pth` where `NAME` is the name given in the arguments and `SUFFIX` is the number of hours of training or best or last. Last is the same as the last number of hours of training.<br>
However, the replay memory is not saved because it is too big, but it is initialized at the beginning of the training with the transitions of the last model until the replay memory size reaches the starting size to train the model. The main inconvenient is the transitions are not the same because the transitions are sampled with the fixed model and not the continuous training model.

## Results

TODO