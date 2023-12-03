import os

from moviepy.editor import VideoFileClip


path = os.path.join("content", "game.mp4")
if not os.path.exists(path):
    raise FileNotFoundError(f"File does not exist at path {path}")

def convert_mp4_to_gif(mp4_file: str, gif_file: str, fps: int = 30):
    """
    Convert mp4 file to gif file
    
    Parameters
    ----------
    mp4_file : str
        Path to mp4 file
    gif_file : str
        Path to gif file
    fps : int
        Frames per second, by default 30
    """
    video = VideoFileClip(mp4_file)
    video.write_gif(gif_file, fps=fps, program='ffmpeg')

convert_mp4_to_gif(path, os.path.join("content", "game.gif"))
