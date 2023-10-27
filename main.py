import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Game Launcher')
    
    parser.add_argument('action', choices=['train', 'play'], help="Specify 'train' or 'play'")
    parser.add_argument('game_name', type=str, help="Name of the game")

    parser.add_argument('--mode', type=int, default=0, help="Game mode")
    parser.add_argument('--difficulty', type=int, default=0, help="Difficulty level")

    parser.add_argument('--ep', type=int, help="Episode number to load for playing")

    args = parser.parse_args()

    if args.action == 'train':
        script_path = os.path.join('scripts', 'train.py')
        os.system(f"python {script_path} {args.game_name} {args.mode} {args.difficulty}")

    elif args.action == 'play':
        script_path = os.path.join('scripts', 'play.py')
        os.system(f"python {script_path} {args.game_name} {args.mode} {args.difficulty} {args.ep}")


if __name__ == "__main__":
    main()
