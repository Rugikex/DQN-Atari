import re
import os


pattern = r"episode_(\d+).keras"

def get_model_path(game_name: str, episode: str) -> tuple[str, str]:
    model_path = None
    max_number: int
    try:
        episode = int(episode)
    except ValueError:
        episode = None

    if episode is not None:
        model_path = f'episode_{episode}.keras'
        if not os.path.exists(os.path.join('models', game_name, f'{model_path}')):
            raise Exception('No model found')
        max_number = episode
        
    else:
        max_number = 0
        for filename in os.listdir(os.path.join('models', game_name)):
            match = re.match(pattern, filename)
            if match:
                # Extract the number of episode from the filename
                number = int(match.group(1))

                # Check if this number is greater than the current max_number
                if number > max_number:
                    max_number = number
                    model_path = filename

    if not model_path:
        raise Exception('No model found')
    
    # Check if the replay memory associated with the model exists
    replay_memory_path = f'replay_memory_{max_number}.pkl'
    if not os.path.exists(os.path.join('models', game_name, replay_memory_path)):
        raise Exception('No replay memory found')

    return model_path, replay_memory_path
