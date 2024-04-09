import unittest
from unittest.mock import patch
import sys
import subprocess

class TestVideoSync(unittest.TestCase):

    # test successful sync
    @patch('subprocess.check_output')
    @patch('subprocess.run')
    def test_successful_sync(self, mock_run, mock_check_output):
        mock_args = ["main.py", "video1.mp4", "video2.mp4"]
        mock_check_output.return_value = '100,200,false'
        with patch.object(sys, 'argv', mock_args):
            from main import main
            main()
        mock_check_output.assert_called_with(['python', 'sync.py', 'video1.mp4', 'video2.mp4'], text=True, stderr=subprocess.STDOUT)
        mock_run.assert_called_with(['python', 'diff.py', 'video1.mp4', 'video2.mp4', '100', '200', 'False'])

    # test video switch sync
    @patch('subprocess.check_output')
    @patch('subprocess.run')
    def test_videos_switched_sync(self, mock_run, mock_check_output):
        mock_args = ["main.py", "video1.mp4", "video2.mp4"]
        mock_check_output.return_value = '0,0,true'
        with patch.object(sys, 'argv', mock_args):
            from main import main
            main()
        mock_check_output.assert_called_with(['python', 'sync.py', 'video1.mp4', 'video2.mp4'], text=True, stderr=subprocess.STDOUT)
        mock_run.assert_called_with(['python', 'diff.py', 'video1.mp4', 'video2.mp4', '0', '0', 'True'])
    
    #test sync failure
    @patch('subprocess.check_output')
    def test_sync_failure(self, mock_check_output):
        mock_args = ["main.py", "video1.mp4", "video2.mp4"]
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'python sync.py')
        with patch.object(sys, 'argv', mock_args):
            from main import main
            with self.assertRaises(subprocess.CalledProcessError):
                main()

    #test invalid output during sync
    @patch('subprocess.check_output')
    def test_invalid_output_sync(self, mock_check_output):
        mock_args = ["main.py", "video1.mp4", "video2.mp4"]
        mock_check_output.return_value = 'not,integers,ortrue'
        with patch.object(sys, 'argv', mock_args):
            from main import main
            with self.assertRaises(ValueError):
                main()

    #test for empty argument during sync
    @patch('subprocess.check_output')
    def test_empty_argument_sync(self, mock_check_output):
        mock_args = ["main.py"]
        with patch.object(sys, 'argv', mock_args):
            from main import main
            with self.assertRaises(IndexError):
                main()

if __name__ == '__main__':
    unittest.main()