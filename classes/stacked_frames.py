from collections import deque

import cv2
import numpy as np


class StackedFrames:
    """
    Stack frames to feed them to the agent

    Parameters
    ----------
    stack_size : int
        Number of frames to stack
    """

    def __init__(self, stack_size: int) -> None:
        self._frames: deque = deque(maxlen=stack_size)

    def _preprocess_frame(
        self, frame: np.ndarray, previous_frame: np.ndarray
    ) -> np.ndarray:
        """
        Preprocess the frame before appending it to the stack

        Parameters
        ----------
        frame : np.ndarray
            Current frame
        previous_frame : np.ndarray
            Previous frame

        Returns
        -------
        resized_frame : np.ndarray
            Resized frame
        """
        # Take maximum of current and previous frame rgb values
        max_frame = np.maximum(frame, previous_frame)

        # Extract luminance from the frame
        luminance_frame = cv2.cvtColor(max_frame, cv2.COLOR_BGR2YUV)[:, :, 0]

        # Resize the Y component to 84x84
        resized_frame = cv2.resize(luminance_frame, (84, 84), interpolation=cv2.INTER_AREA)

        # Convert to uint8 to save memory
        return resized_frame.astype(np.uint8)

    def append(self, frame: np.ndarray, previous_frame: np.ndarray) -> None:
        """
        Append a frame to the stack

        Parameters
        ----------
        frame : np.ndarray
            Current frame
        previous_frame : np.ndarray
            Previous frame
        """
        self._frames.append(self._preprocess_frame(frame, previous_frame))

    def get_frames(self) -> np.ndarray:
        """
        Get the stacked frames

        Returns
        -------
        frames : np.ndarray
            Stacked frames
        """
        return np.array(self._frames)

    def reset(self, frame: np.ndarray) -> None:
        """
        Reset the stack with the first frame

        Parameters
        ----------
        frame : np.ndarray
            First frame
        """
        previous_frame = np.zeros(frame.shape, dtype=frame.dtype)
        initial_frame = self._preprocess_frame(frame, previous_frame)
        for _ in range(self._frames.maxlen):
            self._frames.append(initial_frame)
