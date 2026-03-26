from collections import deque
from typing import Deque, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class StackedFrames:
    """
    Stack frames to feed them to the agent

    Parameters
    ----------
    stack_size : int
        Number of frames to stack
    """

    def __init__(self, stack_size: int, resolution: Tuple[int, int] = (84, 84)) -> None:
        self._frames: Deque[NDArray[np.uint8]] = deque(maxlen=stack_size)
        self.resolution: Tuple[int, int] = resolution

    def _preprocess_frame(
        self, frame: NDArray[np.uint8], previous_frame: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """
        Preprocess the frame before appending it to the stack

        Parameters
        ----------
        frame : NDArray[np.uint8]
            Current frame
        previous_frame : NDArray[np.uint8]
            Previous frame

        Returns
        -------
        resized_frame : NDArray[np.uint8]
            Resized frame
        """
        # Take maximum of current and previous frame rgb values
        max_frame: NDArray[np.uint8] = np.maximum(frame, previous_frame)

        # Extract luminance from the frame
        luminance_frame: NDArray[np.uint8] = cv2.cvtColor(max_frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to the desired resolution
        resized_frame: NDArray[np.uint8] = cv2.resize(
            luminance_frame, self.resolution, interpolation=cv2.INTER_AREA
        )

        return resized_frame

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
