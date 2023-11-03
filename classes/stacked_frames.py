from collections import deque

import cv2
import numpy as np


class StackedFrames():
    def __init__(self, stack_size: int) -> None:
        self._frames: deque = deque(maxlen=stack_size)

    def _preprocess_frame(self, frame: np.ndarray, previous_frame: np.ndarray) -> np.ndarray:
        # Take maximum of current and previous frame rgb values
        max_frame = np.maximum(frame, previous_frame)

        # Extract luminance
        luminance_frame = np.dot(max_frame, [0.299, 0.587, 0.114])

        # Rescale to 84x84
        resized_frame = cv2.resize(luminance_frame, (84, 84), interpolation=cv2.INTER_AREA)

        # Convert to uint8 to save memory
        return resized_frame.astype(np.uint8)

    def append(self, frame: np.ndarray, previous_frame: np.ndarray) -> None:
        self._frames.append(self._preprocess_frame(frame, previous_frame))
    
    def get_frames(self) -> np.ndarray:
        # Return a numpy array of shape (4, 84, 84)
        return np.array(self._frames)
    
    def reset(self, frame: np.ndarray) -> None:
        previous_frame = np.zeros(frame.shape, dtype=frame.dtype)
        initial_frame = self._preprocess_frame(frame, previous_frame)
        for _ in range(self._frames.maxlen):
            self._frames.append(initial_frame)
