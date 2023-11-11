import os


def clear_models():
    """
    Clears all models
    Except for the README.md file
    """
    for filename in os.listdir(os.path.join("models")):
        if filename != "README.md":
            model_path = os.path.join("models", filename)
            for model_file in os.listdir(model_path):
                os.remove(os.path.join(model_path, model_file))
            os.rmdir(model_path)

    print("Models cleared")


if __name__ == '__main__':
    clear_models()
