import unittest
import cv2 as cv
import numpy as np
from unittest import mock
from unittest.mock import patch


from diff import rescaleFrame, handle_key_events, process_contours, split_contours, remove_subset, frame_difference, visualize_difference, combine_rectangles, set_display_properties, detected_contour_list, frame_count_list

# Global variables to store detected contours and their corresponding frame counts
detected_contour_list = []
frame_count_list = []

class TestRescaleFrameFunction(unittest.TestCase):

    def test_with_empty_input(self):
        with self.assertRaises(Exception):
            rescaleFrame(np.array([]), 0.5)

    def test_with_scale_one(self):
        frame = np.random.randint(255, size=(100, 100, 3), dtype=np.uint8)
        scaled_frame = rescaleFrame(frame, 1)
        self.assertEqual(scaled_frame.shape, frame.shape)

    def test_with_scale_greater_than_one(self):
        frame = np.random.randint(255, size=(100, 100, 3), dtype=np.uint8)
        scale = 2
        expected_dimensions = (frame.shape[1] * scale, frame.shape[0] * scale)
        scaled_frame = rescaleFrame(frame, scale)
        self.assertEqual(scaled_frame.shape[:2], expected_dimensions)

    def test_with_scale_less_than_one(self):
        frame = np.random.randint(255, size=(100, 100, 3), dtype=np.uint8)
        scale = 0.5
        expected_dimensions = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        scaled_frame = rescaleFrame(frame, scale)
        self.assertEqual(scaled_frame.shape[:2], expected_dimensions)

    def test_with_large_input_frame(self):
        frame = np.random.randint(255, size=(10000, 10000, 3), dtype=np.uint8)
        scale = 0.01
        expected_dimensions = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        scaled_frame = rescaleFrame(frame, scale)
        self.assertEqual(scaled_frame.shape[:2], expected_dimensions)

class TestHandleKeyEvents(unittest.TestCase):
    def test_pause_unpause(self):
        """Test if pressing 'p' toggles the pause state."""
        self.assertEqual(handle_key_events(ord('p'), False), (False, True))  # Unpaused to paused
        self.assertEqual(handle_key_events(ord('p'), True), (False, False))  # Paused to unpaused

    def test_quit(self):
        """Test if pressing 'q' sets the quit flag."""
        self.assertEqual(handle_key_events(ord('q'), False), (True, False))
        self.assertEqual(handle_key_events(ord('q'), True), (True, True))

    def test_unrecognized_key(self):
        """Test how function reacts to unrecognized key inputs. Should not alter the state."""
        initial_states = [False, True]
        for initial_state in initial_states:
            with self.subTest(initial_state=initial_state):
                self.assertEqual(handle_key_events(ord('a'), initial_state), (False, initial_state))

    def test_no_key_pressed(self):
        """Test behavior when no key is pressed (invalid input)."""
        self.assertEqual(handle_key_events(None, False), (False, False))  # Assuming 'None' simulates no key press
        self.assertEqual(handle_key_events(None, True), (False, True))

    def test_large_key_value(self):
        """Test with a large key value beyond the ASCII range, expecting unchanged state."""
        self.assertEqual(handle_key_events(10000, False), (False, False))
        self.assertEqual(handle_key_events(10000, True), (False, True))

class TestProcessContours(unittest.TestCase):
    def setUp(self) -> None:
        # Reset global variables before each test
        global detected_contour_list, frame_count_list
        detected_contour_list.clear()
        frame_count_list.clear()

    def test_empty_contours_input(self):
        process_contours([], 10, 100)
        self.assertEqual(len(detected_contour_list), 0, "detected_contour_list should remain empty for empty contours input")
        self.assertEqual(len(frame_count_list), 0, "frame_count_list should remain empty for empty contours input")

    def test_null_input(self):
        with self.assertRaises(TypeError):
            process_contours(None, None, None)

    def test_invalid_contours_input(self):
        with self.assertRaises(cv.error):
            process_contours(["Invalid contour"], 10, 100)

    def test_difference_report_trigger(self):
        global printed_messages
        printed_messages = []

        def mock_print(msg):
            printed_messages.append(msg)

        contour_initial = np.array([[[0, 0]], [[1, 1]], [[2, 2]]])
        with patch('builtins.print', mock_print):
            process_contours([contour_initial], 0, 0)
            # Simulating 31 frames later without matching contours to trigger the difference report
            process_contours([], 31, 31)
            self.assertTrue(len(printed_messages) > 0, "Difference report should have been triggered")
            self.assertIn("Difference found at", printed_messages[0])


class TestSplitContours(unittest.TestCase):
    def setUp(self):
        # Creating simple contours for testing
        self.larger_contour = np.array([[5, 5], [10, 5], [10, 10], [5, 10]], dtype=np.int32).reshape((-1, 1, 2))
        self.smaller_contour_inside = np.array([[6, 6], [9, 6], [9, 9], [6, 9]], dtype=np.int32).reshape((-1, 1, 2))
        self.smaller_contour_outside = np.array([[1, 1], [4, 1], [4, 4], [1, 4]], dtype=np.int32).reshape((-1, 1, 2))

    def test_split_contours_with_valid_input_inside(self):
        larger_contour_img = np.zeros((15, 15), dtype=np.uint8)
        cv.drawContours(larger_contour_img, [self.larger_contour], -1, (255), thickness=cv.FILLED)

        smaller_contour_img = np.zeros((15, 15), dtype=np.uint8)
        cv.drawContours(smaller_contour_img, [self.smaller_contour_inside], -1, (255), thickness=cv.FILLED)

        new_half, orig_half = split_contours(larger_contour_img, self.smaller_contour_inside)
        
        # Assert that the new half is not empty and the original half contains the larger minus smaller contour
        self.assertTrue(np.any(new_half))
        self.assertTrue(np.any(orig_half))
        self.assertFalse(np.array_equal(new_half, orig_half))

    def test_split_contours_with_valid_input_outside(self):
        larger_contour_img = np.zeros((15, 15), dtype=np.uint8)
        cv.drawContours(larger_contour_img, [self.larger_contour], -1, (255), thickness=cv.FILLED)

        outside_contour_img = np.zeros((15, 15), dtype=np.uint8)
        cv.drawContours(outside_contour_img, [self.smaller_contour_outside], -1, (255), thickness=cv.FILLED)

        new_half, orig_half = split_contours(larger_contour_img, self.smaller_contour_outside)
        
        # Assert that the new half is empty (outside contour) and the original half remains unchanged
        self.assertFalse(np.any(new_half))
        self.assertTrue(np.array_equal(orig_half, larger_contour_img))

    def test_split_contours_with_empty_smaller_contour(self):
        larger_contour_img = np.zeros((15, 15), dtype=np.uint8)
        cv.drawContours(larger_contour_img, [self.larger_contour], -1, (255), thickness=cv.FILLED)

        empty_contour = np.array([], dtype=np.int32).reshape((-1, 1, 2))
        
        with self.assertRaises(Exception):
            split_contours(larger_contour_img, empty_contour)

    def test_split_contours_with_non_contour_input(self):
        with self.assertRaises(cv.error):
            split_contours("This isnot a contour", "This is also not a contour")

class TestRemoveSubset(unittest.TestCase):
    def test_empty_inputs(self):
        self.assertRaises(cv.error, remove_subset, np.array([]), np.array([]))

    def test_invalid_inputs(self):
        self.assertRaises(cv.error, remove_subset, "invalid_input", "another_invalid_input")

if __name__ == '__main__':
    unittest.main()