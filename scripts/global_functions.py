import os


def get_model_path(game_name: str, episode: int | None = None) -> tuple[str, str]:
    model_path = None
    max_number: int

    if episode:
        model_path = f'ep_{episode}'
        if not os.path.isfile(os.path.join('models', game_name, model_path)):
            raise Exception('No model found')
        max_number = episode
        
    else:
        max_number = 0
        for filename in os.listdir(os.path.join('models', game_name)):
            if filename.startswith('ep_'):
                # Extract the number of episode from the filename
                number = int(filename.split('_')[1])

                # Check if this number is greater than the current max_number
                if number > max_number:
                    max_number = number
                    model_path = filename

    if not model_path:
        raise Exception('No model found')
    
    # Check if the replay memory associated with the model exists
    replay_memory_path = os.path.join('models', game_name, f'replay_memory_{max_number}.pkl')
    if not os.path.exists(os.path.join('models', game_name, replay_memory_path)):
        raise Exception('No replay memory found')
    
    return model_path, replay_memory_path
