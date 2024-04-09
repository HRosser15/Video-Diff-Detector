import unittest
import numpy as np
import cv2 as cv

from sync import find_frame_difference

class TestFindFrameDifference(unittest.TestCase):

    # test for identical frames
    def test_identical_frames(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8) 
        self.assertFalse(find_frame_difference(frame1, frame2))

    # test for different frames
    def test_completely_different_frames(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)   # Completely white BGR frame
        self.assertTrue(find_frame_difference(frame1, frame2))

    # test for difference being right at the threshhold
    def test_threshold_difference_below_edge(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[:50, :50, :] = 255  # Set a quadrant to white
        self.assertTrue(find_frame_difference(frame1, frame2), 
        "Edge case failed: The difference is right at the threshold but not exceeding.")

    # test the difference that just goes over the threshold    
    def test_threshold_difference_just_over_threshold(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[:50, :50, :] = 255
        frame2[50, 50, :] = 255
        self.assertTrue(find_frame_difference(frame1, frame2), 
        "Edge case failed: The difference slightly exceeds the threshold.")

    # test for no frame input 
    def test_no_frame2_input(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(TypeError):
            find_frame_difference(frame1)

    # tests invalid data type on frame 1
    def test_invalid_type_frame1(self):
        frame1 = "[Invalid Data Type]"
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(Exception):
            find_frame_difference(frame1, frame2)
    
    # tests invalid data type on frame 2
    def test_invalid_type_frame2(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = "[Invalid Data Type]"
        with self.assertRaises(Exception):
            find_frame_difference(frame1, frame2)

    # test for large frame differences
    def test_large_frame_difference(self):
        frame1 = np.zeros((10000, 10000), dtype=np.uint8)
        frame2 = np.full((10000, 10000, 3), 255, dtype=np.uint8)
        self.assertTrue(find_frame_difference(frame1, frame2), "Failed for large frames.")

if __name__ == '__main__':
    unittest.main()