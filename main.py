import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Video Difference Detector')
    parser.add_argument('video1_path', help='Path to the first video file')
    parser.add_argument('video2_path', help='Path to the second video file')
    parser.add_argument('-v', '--video', default='false', help='Save the analysis process to video')
    parser.add_argument('-t', '--threshold', type=int, help='Set the threshold value for pixel-wise difference calculation (default: 25).')
    parser.add_argument('-c', '--contour-area', type=int, help='Set the minimum contour area threshold for contour filtering (default: 100).')
    parser.add_argument('-r', '--resolution', help='Set the resolution for the output video (e.g., 1280x720).')
    parser.add_argument('-l', '--log-file', help='Specify the path to a log file for writing log messages.')
    parser.add_argument('-d', '--duration', default=1, help='Specify the duration (in seconds) a difference must be present to be logged (default: 1 second).')
    parser.add_argument('--weight', type=int, help='Weight of the bounding boxes (default: (based on input video dimensions))')
    parser.add_argument('--margin', type=int, help='Margin around the bounding boxes (default: (based on input video dimensions))')
    parser.add_argument('-b', '--box-color', default='yellow', help='Set the color of the bounding boxes (default: yellow).')
    parser.add_argument('-nv', '--no-visualization', action='store_true', help='Disable visualization of the analysis process')
    parser.add_argument('-stf', '--sync-threshold-find', type=int, help='Set the threshold value finding a reference frame in the base video (default: based on video resolution).')
    parser.add_argument('-stm', '--sync-threshold-match', type=int, help='Set the threshold value finding a matching frame in the delta video (default: based on video resolution).')
    args = parser.parse_args()

    # Get video paths from command line arguments
    video1_path = args.video1_path
    video2_path = args.video2_path



    # Call video synchronization script, and catch any errors that occur
    try:
        sync_output = subprocess.check_output(['python', 'sync.py', video1_path, video2_path], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error: Unable to synchronize the videos.")
        print("Please ensure that both videos have the same resolution.")
        print("Error details:")
        print(e.output)
        return
    
    # Parse the output string from the video synchronizer
    base_start_frame, alt_start_frame, videos_switched_str = sync_output.strip().split(',')
    if base_start_frame == 'None' or alt_start_frame == 'None' or videos_switched_str == 'None':
        print("Error: Unable to find a sync frame. The videos could not be synchronized.")
        print("Please ensure that the videos have overlapping content and are of sufficient length.")
        return
    
    base_start_frame = int(base_start_frame)
    alt_start_frame = int(alt_start_frame)
    videos_switched = videos_switched_str.lower() == 'true'
    if videos_switched:
        print('===Videos switched===')
        print('No matching frame was found in the alternative video. The videos have been switched.')
        print('This if often the result of the videos being different lengths or having different content.')
        print('This will NOT affect how differences appear in the logs.')

    diff_command = ['python', 'diff.py', video1_path, video2_path, str(base_start_frame), str(alt_start_frame)]

    if args.threshold:
        diff_command.extend(['--threshold', str(args.threshold)])
    if args.contour_area:
        diff_command.extend(['--contour-area', str(args.contour_area)])
    if args.margin:
        diff_command.extend(['--margin', str(args.margin)])
    if args.resolution:
        diff_command.extend(['--resolution', args.resolution])
    if args.log_file:
        diff_command.extend(['--log-file', args.log_file])
    if args.weight is not None:
        diff_command.extend(['--weight', str(args.weight)])
    if args.video.lower() == 'true':
        diff_command.append('-v')
    if args.no_visualization:
        diff_command.append('-nv')
    diff_command.extend(['-d', str(args.duration)])

    subprocess.run(diff_command)

if __name__ == "__main__":
    main()