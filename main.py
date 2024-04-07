import sys
import subprocess

def main():
    # Get video paths from command line arguments
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]

    # Call video synchronization script
    sync_output = subprocess.check_output(['python', 'sync.py', video1_path, video2_path], text=True, stderr=subprocess.STDOUT)

    # Parse the output string
    base_start_frame, alt_start_frame, videos_switched_str = sync_output.strip().split(',')
    base_start_frame = int(base_start_frame)
    alt_start_frame = int(alt_start_frame)
    videos_switched = videos_switched_str.lower() == 'true'
    print(f'videos_switched: {videos_switched}')

    # Call difference detection script with relevant information as arguments
    subprocess.run(['python', 'diff_find.py', video1_path, video2_path, str(base_start_frame), str(alt_start_frame), str(videos_switched)])

if __name__ == "__main__":
    main()