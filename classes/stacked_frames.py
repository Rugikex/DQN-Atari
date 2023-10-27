from collections import deque

import cv2
import numpy as np


class StackedFrames():
    def __init__(self, stack_size: int):
        self.frames: deque = deque(maxlen=stack_size)
        self.previous_frame: np.ndarray = np.zeros((210, 160, 3))

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        # Take maximum of current and previous frame rgb values
        max_frame = np.maximum(frame, self.previous_frame)
        self.previous_frame = frame

        # Extract luminance
        luminance_frame = np.dot(max_frame, [0.299, 0.587, 0.114])

        # Rescale to 84x84
        resized_frame = cv2.resize(luminance_frame, (84, 84), interpolation=cv2.INTER_AREA)

        return resized_frame

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(self._preprocess_frame(frame))
    
    def get_frames(self) -> np.ndarray:
        # Return a numpy array of shape (84, 84, 4)
        return np.stack(self.frames, axis=-1)
    
    def reset(self, frame: np.ndarray) -> None:
        self.previous_frame = np.zeros((210, 160, 3))
        initial_frame = self._preprocess_frame(frame)
        for _ in range(self.frames.maxlen):
            self.frames.append(initial_frame)
