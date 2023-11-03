import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Game Launcher')
    
    parser.add_argument('action', choices=['train', 'retrain', 'play'], help="Specify 'train', 'retrain or 'play'")
    parser.add_argument('game_name', type=str, help="Name of the game")

    parser.add_argument('--mode', type=int, default=0, help="Game mode")
    parser.add_argument('--difficulty', type=int, default=0, help="Difficulty level")
    parser.add_argument('--repeat', type=int, default=1, help="Number of times to repeat the training (episodes * repeat)")

    parser.add_argument('--name', type=str, default=None, help="Name of the model to retrain or play")

    args = parser.parse_args()

    if args.repeat < 1:
        raise Exception('The number of repeats must be greater than 0')

    if args.action == 'train':
        script_path = os.path.join('scripts', 'train.py')
        os.system(f"python {script_path} {args.game_name} {args.mode} {args.difficulty} {args.repeat} {args.name}")
        
    elif args.action == 'retrain':
        script_path = os.path.join('scripts', 'retrain.py')
        os.system(f"python {script_path} {args.game_name} {args.mode} {args.difficulty} {args.repeat} {args.name}")

    elif args.action == 'play':
        script_path = os.path.join('scripts', 'play.py')
        os.system(f"python {script_path} {args.game_name} {args.mode} {args.difficulty} {args.name}")


if __name__ == "__main__":
    main()
