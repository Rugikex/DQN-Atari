import os
import shutil


def clear_videos():
    """
    Clears all videos
    """
    if os.path.exists("videos"):
        shutil.rmtree("videos")
    print("Videos cleared")


if __name__ == "__main__":
    clear_videos()
