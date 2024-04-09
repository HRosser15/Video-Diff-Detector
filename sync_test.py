from operator import call
import sys
import unittest
import cv2 as cv
import numpy as np
from unittest import mock
from unittest.mock import Mock, patch

from sync import find_frame_difference, find_matching_frame, find_sync_frame_number, get_frame_at_number, find_matching_frame_number, main

class TestFindFrameDifference(unittest.TestCase):

    def test_empty_frame(self):
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertFalse(find_frame_difference(gray1, frame2))

    def test_identical_frames(self):
        gray1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        frame2 = cv.cvtColor(gray1, cv.COLOR_GRAY2BGR)
        self.assertFalse(find_frame_difference(gray1, frame2))
    
    def test_frames_with_high_difference(self):
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.assertTrue(find_frame_difference(gray1, frame2))

    def test_frames_with_threshold_difference(self):
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[0:50, 0:50, :] = 255  # Set a quarter of the frame to white
        self.assertTrue(find_frame_difference(gray1, frame2))

    def test_large_input(self):
        gray1 = np.random.randint(0, 256, (10000, 10000), dtype=np.uint8)
        frame2 = cv.cvtColor(gray1, cv.COLOR_GRAY2BGR)
        frame2[5000:6000, 5000:6000, :] = 255  # Create a noticeable difference
        self.assertTrue(find_frame_difference(gray1, frame2))

    def test_single_element_no_difference(self):
        gray1 = np.array([[0]], dtype=np.uint8)
        frame2 = np.array([[[0, 0, 0]]], dtype=np.uint8)
        self.assertFalse(find_frame_difference(gray1, frame2))

    def find_matching_frame(gray1, frame2):
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  
        diff = cv.absdiff(gray1, gray2)
        _, thresholded_diff = cv.threshold(diff, 10, 255, cv.THRESH_BINARY)
        non_zero_count = np.count_nonzero(thresholded_diff)
        threshold = 20
        return non_zero_count < threshold

class TestFindMatchingFrame(unittest.TestCase):

    def test_empty_frames(self):
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertTrue(find_matching_frame(gray1, frame2))

    def test_invalid_input_shape(self):
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((50, 50, 3), dtype=np.uint8)
        with self.assertRaises(cv.error):
            find_matching_frame(gray1, frame2)

    def test_high_threshold_no_match(self):
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.assertFalse(find_matching_frame(gray1, frame2))

    def test_identical_frames(self):
        gray1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        frame2 = cv.cvtColor(cv.cvtColor(gray1, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2GRAY)
        self.assertTrue(find_matching_frame(gray1, cv.cvtColor(gray1, cv.COLOR_GRAY2BGR)))

    def test_threshold_edge_case(self):
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[0:20, 0:20, :] = 10 
        self.assertTrue(find_matching_frame(gray1, frame2))

    def test_large_input_frames(self):
        gray1 = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        frame2 = cv.cvtColor(cv.cvtColor(gray1, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2GRAY)
        self.assertTrue(find_matching_frame(gray1, cv.cvtColor(gray1, cv.COLOR_GRAY2BGR)))

    def test_small_input_frames(self):
        gray1 = np.array([[1]], dtype=np.uint8)
        frame2 = np.array([[[1, 1, 1]]], dtype=np.uint8)
        self.assertTrue(find_matching_frame(gray1, frame2))

class TestFindSyncFrameNumber(unittest.TestCase):
    def setUp(self):
        # Create a VideoCapture object for testing. This uses a small video or a generated video.
        self.base_vid_path = "test_video.mp4"
        self.base_vid = cv.VideoCapture(self.base_vid_path)

        # Generating a simple video for testing purposes, if not available.
        if not self.base_vid.isOpened():
            height, width = 480, 640
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(self.base_vid_path, fourcc, 20.0, (width, height))

            for i in range(50):  # Let's say a video of 50 frames.
                blank_image = np.zeros((height, width, 3), np.uint8)
                if i == 25:  # Insert a distinct frame in the middle
                    blank_image[:] = (255, 0, 0)  # Blue frame
                out.write(blank_image)

            out.release()  # Finalizing the video file.
            self.base_vid = cv.VideoCapture(self.base_vid_path)  # Re-open the newly created video

        # Read the first frame to use as the first_frame input parameter.
        ret, self.first_frame = self.base_vid.read()
        self.first_frame = cv.cvtColor(self.first_frame, cv.COLOR_BGR2GRAY)

    def test_no_sync_frame(self):
        # Assuming a scenario where no sync (blue frame in this case) should be found.
        # We purposefully pass a frame that should not sync with any frame in the video.
        non_matching_first_frame = np.zeros((480, 640), np.uint8)  # A black frame, no sync should be found.
        result = find_sync_frame_number(non_matching_first_frame, self.base_vid)
        self.assertIsNone(result, "Expected None since no frame should match.")

    def test_base_vid_is_none(self):
        # Testing the scenario when base_vid is None
        with self.assertRaises(AttributeError):  # Assuming the function will raise AttributeError or adjust accordingly.
            find_sync_frame_number(self.first_frame, None)

    def test_base_vid_with_no_frames(self):
        # Create an "empty" video capture object
        base_vid = cv.VideoCapture("")  # An intentionally invalid path to create an empty video object.
        result = find_sync_frame_number(self.first_frame, base_vid)
        self.assertIsNone(result, "Expected None since the video has no frames.")

    def tearDown(self):
        # Release the VideoCapture object
        self.base_vid.release()

class TestGetFrameAtNumber(unittest.TestCase):
    def mock_read_success(*args, **kwargs):
    # Mock success response for cv.VideoCapture.read
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def mock_read_fail(*args, **kwargs):
    # Mock failure response for cv.VideoCapture.read
        return False, None

    @mock.patch('cv2.VideoCapture.read', side_effect=mock_read_success)
    def test_get_frame_at_valid_number(self, mock_read):
        vid = cv.VideoCapture()
        frame = get_frame_at_number(10, vid)
        self.assertIsNotNone(frame)
        self.assertTrue(isinstance(frame, np.ndarray))

    @mock.patch('cv2.VideoCapture.read', side_effect=mock_read_fail)
    def test_get_frame_at_invalid_number(self, mock_read):
        vid = cv.VideoCapture()
        frame = get_frame_at_number(999999, vid)
        self.assertIsNone(frame)

    @mock.patch('cv2.VideoCapture.set')
    @mock.patch('cv2.VideoCapture.read', side_effect=mock_read_success)
    def test_frame_number_set_correctly(self, mock_read, mock_set):
        vid = cv.VideoCapture()
        frame_number = 5
        get_frame_at_number(frame_number, vid)
        mock_set.assert_called_with(cv.CAP_PROP_POS_FRAMES, frame_number)

    @mock.patch('cv2.VideoCapture.read', side_effect=mock_read_fail)
    def test_get_frame_at_negative_number(self, mock_read):
        vid = cv.VideoCapture()
        frame = get_frame_at_number(-1, vid)
        self.assertIsNone(frame)

    @mock.patch('cv2.VideoCapture.read', side_effect=mock_read_success)
    def test_get_frame_at_zero(self, mock_read):
        vid = cv.VideoCapture()
        frame = get_frame_at_number(0, vid)
        self.assertIsNotNone(frame)
        self.assertTrue(isinstance(frame, np.ndarray))

    def test_get_frame_with_null_video(self):
        with self.assertRaises(AttributeError):  
            frame = get_frame_at_number(10, None)

class TestFindMatchingFrameNumber(unittest.TestCase):
  
    # Create a mock video capture object for alternative video
    def setUp(self):
        self.alt_vid = mock.MagicMock()
        self.alt_vid.read.side_effect = [
            (True, np.random.randint(0, 255, (10,10,3), dtype=np.uint8)),  # Frame 1
            (True, np.random.randint(0, 255, (10,10,3), dtype=np.uint8)),  # Frame 2
            (True, np.random.randint(0, 255, (10,10,3), dtype=np.uint8)),  # Matching Frame at 3
            (False, None)  # Simulate end of video
        ]
        
        # Create a sync_frame that should match with frame at position 3 in alt_vid
        self.sync_frame = self.alt_vid.read()[1]
        self.sync_frame = cv.cvtColor(self.sync_frame, cv.COLOR_BGR2GRAY)

    def test_no_matching_frame(self):
        # Adjust the side_effect to simulate no matching frame
        self.alt_vid.read.side_effect = [
            (True, np.random.randint(0, 255, (10,10,3), dtype=np.uint8)) for _ in range(5)
        ] + [(False, None)]  # No matching frame

        frame_number = find_matching_frame_number(self.sync_frame, self.alt_vid)
        
        self.assertIsNone(frame_number, "Should return None when no matching frame is found.")

    def test_empty_video(self):
        # Simulate empty video
        self.alt_vid.read.return_value = (False, None)

        frame_number = find_matching_frame_number(self.sync_frame, self.alt_vid)
        self.assertIsNone(frame_number, "Should handle empty video.")

    def test_invalid_sync_frame(self):
        # Simulate invalid sync_frame (empty array)
        frame_number = find_matching_frame_number(np.array([]), self.alt_vid)
        self.assertIsNone(frame_number, "Should handle invalid sync frame.")

if __name__ == '__main__':
    unittest.main()