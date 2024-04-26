import unittest
import cv2 as cv
import numpy as np
from unittest.mock import MagicMock, patch


from diff import rescaleFrame, handle_key_events, process_contours, split_contours, remove_subset, frame_difference, visualize_difference, combine_rectangles, set_display_properties, detected_contour_list, frame_count_list, write_output_video

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

class TestFrameDifferenceFunction(unittest.TestCase):

    def setUp(self):
        # Create sample frames for testing (100x100 pixels, 3 channel RGB)
        self.blank_frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.blank_frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create a frame with a simple difference
        self.frame_with_difference = np.zeros((100, 100, 3), dtype=np.uint8)
        self.frame_with_difference[50:60, 50:60] = [255, 255, 255]  # A white square in the center

    def test_identical_frames(self):
        # Case where the two frames are identical
        diff_image, contours = frame_difference(self.blank_frame1, self.blank_frame2)
        self.assertEqual(len(contours), 0)

    def test_frame_with_difference(self):
        # Case where there's a clear difference
        diff_image, contours = frame_difference(self.blank_frame1, self.frame_with_difference)
        self.assertTrue(len(contours) > 0)

    def test_threshold_impact_high(self):
        # Case where the threshold is set high enough that differences are ignored
        diff_image, contours = frame_difference(self.blank_frame1, self.frame_with_difference, threshold=255)
        self.assertEqual(len(contours), 0)
    
    def test_threshold_impact_low(self):
        # Case where the threshold is low, so even minimal differences are caught
        diff_image, contours = frame_difference(self.blank_frame1, self.frame_with_difference, threshold=1)
        self.assertTrue(len(contours) > 0)

    def test_invalid_input_types(self):
        # Case where input frames are not of the correct type (e.g., None)
        with self.assertRaises(cv.error):
            frame_difference(None, self.blank_frame2)

        with self.assertRaises(cv.error):
            frame_difference(self.blank_frame1, None)

    def test_different_dimensions(self):
        # Case where the frames have different sizes
        different_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with self.assertRaises(cv.error):
            frame_difference(self.blank_frame1, different_frame)

    def test_single_channel_input(self):
        # Case where the frames are single channel
        single_channel_frame1 = np.zeros((100, 100), dtype=np.uint8)
        single_channel_frame2 = single_channel_frame1.copy()
        single_channel_frame2[50:60, 50:60] = 255  # Make a difference in the second frame
        with self.assertRaises(cv.error):
            frame_difference(single_channel_frame1, single_channel_frame2)

class TestVisualizeDifference(unittest.TestCase):

    def test_empty_input(self):
        with self.assertRaises(Exception):
            visualize_difference(None, None, None, [], None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def test_no_contours(self):
        frame1 = np.zeros((100, 100, 3), np.uint8)
        frame2 = np.zeros((100, 100, 3), np.uint8)
        diff_image = np.zeros((100, 100), np.uint8)
        contours = []
        fps = 30
        frame_scale = 1
        border_size = 10
        bottom_border_size = 20
        text_x_pos = 10
        text_y_pos = 90
        text_size = 1
        text_weight = 2
        output_width = 200
        output_height = 100
        frame_width = 100
        frame_height = 100
        concatenated_frame = visualize_difference(frame1, frame2, diff_image, contours, None, fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height)
        
        # Check that output is not None
        self.assertIsNotNone(concatenated_frame)

    def test_with_valid_contours(self):
        frame1 = np.zeros((100, 100, 3), np.uint8)
        frame2 = np.zeros((100, 100, 3), np.uint8)
        diff_image = np.zeros((100, 100), np.uint8)
        contour = np.array([[25, 25], [25, 75], [75, 75], [75, 25]])
        contours = [contour]
        fps = 30
        frame_scale = 1
        border_size = 10
        bottom_border_size = 20
        text_x_pos = 10
        text_y_pos = 90
        text_size = 1
        text_weight = 2
        output_width = 200
        output_height = 100
        frame_width = 100
        frame_height = 100
        concatenated_frame = visualize_difference(frame1, frame2, diff_image, contours, None, fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height)
        
        # Check that output is not None
        self.assertIsNotNone(concatenated_frame)

    def test_invalid_frame_scale(self):
        frame1 = np.zeros((100, 100, 3), np.uint8)
        frame2 = np.zeros((100, 100, 3), np.uint8)
        diff_image = np.zeros((100, 100), np.uint8)
        contours = []
        fps = 30
        frame_scale = 0  # Invalid scale
        border_size = 10
        bottom_border_size = 20
        text_x_pos = 10
        text_y_pos = 90
        text_size = 1
        text_weight = 2
        output_width = 200
        output_height = 100
        frame_width = 100
        frame_height = 100
        with self.assertRaises(Exception):
            visualize_difference(frame1, frame2, diff_image, contours, None, fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height)

    def test_text_outside_frame(self):
        frame1 = np.zeros((100, 100, 3), np.uint8)
        frame2 = np.zeros((100, 100, 3), np.uint8)
        diff_image = np.zeros((100, 100), np.uint8)
        contours = []
        fps = 30
        frame_scale = 1
        border_size = 10
        bottom_border_size = 20
        text_x_pos = 500  # Outside of frame dimensions
        text_y_pos = 500  # Outside of frame dimensions
        text_size = 1
        text_weight = 2
        output_width = 200
        output_height = 100
        frame_width = 100
        frame_height = 100
        concatenated_frame = visualize_difference(frame1, frame2, diff_image, contours, None, fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height)
        
        # Expect that text will not be visible but function should still succeed
        self.assertIsNotNone(concatenated_frame)

    def test_large_contours(self):
        frame1 = np.zeros((1000, 1000, 3), np.uint8)
        frame2 = np.zeros((1000, 1000, 3), np.uint8)
        diff_image = np.zeros((1000, 1000), np.uint8)
        contour = np.array([[250, 250], [250, 750], [750, 750], [750, 250]])
        contours = [contour]
        fps = 30
        frame_scale = 0.5
        border_size = 20
        bottom_border_size = 40
        text_x_pos = 100
        text_y_pos = 900
        text_size = 2
        text_weight = 4
        output_width = 500
        output_height = 500
        frame_width = 1000
        frame_height = 1000
        concatenated_frame = visualize_difference(frame1, frame2, diff_image, contours, None, fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height)
        
        self.assertIsNotNone(concatenated_frame)

class TestCombineRectangles(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(combine_rectangles([]), [])

    def test_single_rectangle(self):
        self.assertEqual(combine_rectangles([(10, 10, 100, 100)]), [(10, 10, 100, 100)])

    def test_two_non_overlapping_rectangles(self):
        rects = [(10, 10, 50, 50), (100, 100, 50, 50)]
        self.assertEqual(combine_rectangles(rects), [(10, 10, 50, 50), (100, 100, 50, 50)])
    
    def test_two_overlapping_rectangles(self):
        rects = [(10, 10, 100, 100), (50, 50, 100, 100)]
        self.assertEqual(combine_rectangles(rects), [(10, 10, 140, 140)])
    
    def test_multiple_separate_rectangles(self):
        rects = [(10, 10, 30, 30), (100, 100, 30, 30), (200, 200, 30, 30)]
        self.assertEqual(combine_rectangles(rects), [(10, 10, 30, 30), (100, 100, 30, 30), (200, 200, 30, 30)])

    def test_overlapping_on_corner(self):
        rects = [(10, 10, 30, 30), (35, 35, 30, 30)]
        self.assertEqual(combine_rectangles(rects), [(10, 10, 55, 55)])

class TestSetDisplayProperties(unittest.TestCase):

    def setUp(self):
        self.mock_cap = MagicMock()
        self.mock_cap.get.side_effect = [1920, 1080, 30]  # Width, Height, FPS

    def test_set_display_properties_with_standard_resolution(self):
        expected_fps = 30
        expected_frame_scale = 0.2  # This expected value may vary depending on screen resolution during the test
        expected_border_size = 64  # Depending on frame_scale
        expected_text_size = 4
        expected_text_weight = 3
        expected_output_width = 819  # Depending on frame_scale, could vary
        expected_output_height = 271  # Depending on frame_scale

        with patch('builtins.print', autospec=True):
            fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height = set_display_properties(self.mock_cap)
        
        self.assertEqual(fps, expected_fps)
        self.assertEqual(frame_scale, expected_frame_scale)
        self.assertEqual(border_size, expected_border_size)
        self.assertEqual(text_size, expected_text_size)
        self.assertEqual(text_weight, expected_text_weight)
        self.assertEqual(output_width, expected_output_width)
        self.assertEqual(output_height, expected_output_height)
    
    def test_set_display_properties_with_invalid_input(self):
        self.mock_cap.get.side_effect = [0, 0, 0]  # Invalid width, height, fps
        with self.assertRaises(ZeroDivisionError):
            set_display_properties(self.mock_cap)
    
    @patch('builtins.print', autospec=True)
    def test_set_display_properties_with_small_resolution(self, mock_print):
        self.mock_cap.get.side_effect = [640, 480, 25]  # Width, Height, FPS
        expected_fps = 25
        expected_border_size = 21  # Depending on frame_scale
        
        fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height = set_display_properties(self.mock_cap)

        self.assertEqual(fps, expected_fps)
        self.assertEqual(border_size, expected_border_size)

class TestWriteOutputVideo(unittest.TestCase):

    # Mock the frame and a closed VideoWriter object
    def test_write_output_video_with_closed_video_writer(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        output_video = cv.VideoWriter()
        output_video.release()  # Release the video writer to simulate a closed state

        # Expect no error but the write method should not be called since the VideoWriter is closed
        write_output_video(frame, output_video)
        self.assertFalse(output_video.isOpened())  # Ensure the VideoWriter is indeed closed


if __name__ == '__main__':
    unittest.main()