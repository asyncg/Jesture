import numpy
import mediapipe

from typing import NamedTuple
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmarkList,
    NormalizedLandmark,
)
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer


class LandmarkDetector:
    """
    A utility class for detecting hand landmarks.

    This class provides methods for detecting landmarks (in frames)
    and retrieving the position of specific landmarks.

    Follow this link for more details regarding hand landmarks
    <https://developers.google.com/mediapipe/solutions/vision/hand_landmarker>

    Attributes:
        hand_processor (mediapipe.solutions.hands.Hands): A MediaPipe Hands
            instance used for detecting hand landmarks.
    """

    def __init__(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        """
        Initialize the LandmarkDetector.

        Args:
            max_num_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence value
                ([0.0, 1.0]) for hand detection.
            min_tracking_confidence (float): Minimum confidence value
                ([0.0, 1.0]) for hand tracking.
        """

        self.hand_processor: mediapipe.solutions.hands.Hands = (
            mediapipe.solutions.hands.Hands(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

    def fetch_landmarks(
        self, rgb_frame: numpy.ndarray
    ) -> dict[int, tuple[float, float]]:
        """
        Fetches the landmarks from the input frame.

        Args:
            rgb_frame (numpy.ndarray): Input RGB frame.

        Returns:
            dict[int, tuple[float, float]]: Dict containing landmark IDs
                as keys and corresponding (x, y) coordinates as values.
        """

        process_results: NamedTuple = self.hand_processor.process(rgb_frame)
        multi_hand_landmarks: list[NormalizedLandmarkList] = (
            process_results.multi_hand_landmarks
        )
        landmark_map: dict[int, tuple[float, float]] = {}

        if not multi_hand_landmarks:
            return landmark_map

        hand_landmarks: NormalizedLandmarkList
        for hand_landmarks in multi_hand_landmarks:
            landmark_container: RepeatedCompositeFieldContainer = (
                hand_landmarks.landmark
            )

            landmark_id: int
            landmark: NormalizedLandmark
            for landmark_id, landmark in enumerate(landmark_container):
                # `landmark.x` and `landmark.y` refer to the
                # abscissa and ordinate of the landmark
                landmark_map[landmark_id] = (landmark.x, landmark.y)

        return landmark_map

    def fetch_landmark_position(
        self, rgb_frame: numpy.ndarray, landmark_id: int
    ) -> tuple[float, float] | None:
        """
        Fetches the position of a specific landmark from the input RGB frame.

        Args:
            rgb_frame (numpy.ndarray): Input RGB frame.
            landmark_id (int): ID of the landmark to fetch.

        Returns:
            tuple[float, float] | None: Tuple containing (x, y) coordinates
                of the landmark if found, else None.
        """

        landmark_position_map: dict[int, tuple[float, float]] | None = (
            self.fetch_landmarks(rgb_frame)
        )
        return landmark_position_map.get(landmark_id, None)
