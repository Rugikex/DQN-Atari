import argparse
import os
import sys

import gymnasium as gym

sys.path.append(os.getcwd())

from classes.agent import AtariAgent


def main():
    parser = argparse.ArgumentParser(description="Game Launcher")

    parser.add_argument(
        "action",
        choices=["train", "retrain", "play"],
        help="Specify 'train', 'retrain' or 'play'",
    )
    parser.add_argument("game_name", type=str, help="Name of the game")

    parser.add_argument("--mode", type=int, default=0, help="Game mode")
    parser.add_argument("--difficulty", type=int, default=0, help="Difficulty level")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of hours to train or retrain the model",
    )

    parser.add_argument("--name", type=str, default=None, help="Name of the model")

    args = parser.parse_args()

    if args.repeat < 1:
        raise Exception("The number of repeats must be greater than 0")
    
    render_mode = "human" if args.action == "play" else "rgb_array"
    
    env = gym.make(
        args.game_name,
        mode=args.mode,
        difficulty=args.difficulty,
        frameskip=1,
        repeat_action_probability=0.0,
        obs_type="rgb",
        full_action_space=False,
        render_mode=render_mode,
    )

    agent = AtariAgent(args.game_name, env, play=args.action == "play")

    if args.action == "train":
        agent.train(args.repeat, args.name)

    elif args.action == "retrain":
        agent.load_model(args.name)
        agent.train(args.repeat, args.name)

    elif args.action == "play":
        agent.load_model(args.name, play=True)
        agent.play()


if __name__ == "__main__":
    main()
