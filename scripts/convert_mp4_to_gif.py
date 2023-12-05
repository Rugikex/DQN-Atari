import os

from moviepy.editor import VideoFileClip


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
    if not os.path.exists(mp4_file):
        raise FileNotFoundError(f"File does not exist at path {mp4_file}")

    video = VideoFileClip(mp4_file)

    if not os.path.exists(os.path.dirname(gif_file)):
        os.makedirs(os.path.dirname(gif_file))

    video.write_gif(gif_file, fps=fps, program="ffmpeg")


if __name__ == "__main__":
    video_name = "game"
    path = os.path.join("content", f"{video_name}.mp4")
    convert_mp4_to_gif(path, os.path.join("content", f"{video_name}.gif"))
