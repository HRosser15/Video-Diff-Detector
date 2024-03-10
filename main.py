import sys
import subprocess

def main():
    # Get video paths from command line arguments
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]

    # Call video synchronization script
    sync_output = subprocess.check_output(['python', 'video_sync.py', video1_path, video2_path], text=True, stderr=subprocess.STDOUT)

    base_start_frame, alt_start_frame = map(int, sync_output.split())

    # Call difference detection script with relevant information as arguments
    subprocess.run(['python', 'diff_5.py', video1_path, video2_path, str(base_start_frame), str(alt_start_frame)])

if __name__ == "__main__":
    main()
