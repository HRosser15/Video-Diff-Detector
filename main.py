import sys
import subprocess

def main():
    if len(sys.argv) != 3:
        print("Error: Incorrect number of arguments.")
        print("Usage: python main.py <video1_file_name.ext> <video2_file_name.ext>")
        return
    
    # Get video paths from command line arguments
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]

    try:
        # Call video synchronization script
        sync_output = subprocess.check_output(['python', 'sync.py', video1_path, video2_path], text=True, stderr=subprocess.STDOUT)

        # Parse the output string from the video syncrhonizer
        # These values are passed into the diff.py subprocess to tell it which frames to start at
        base_start_frame, alt_start_frame, videos_switched_str = sync_output.strip().split(',')
        base_start_frame = int(base_start_frame)
        alt_start_frame = int(alt_start_frame)
        videos_switched = videos_switched_str.lower() == 'true'
        if videos_switched:
            print('===Videos switched===')
            print('No matching frame was found in the alternative video. The videos have been switched.')
            print('This if often the result of the videos being different lengths or having different content.')
            print('This will NOT affect how differences appear in the logs.')

        # Call difference detection script with relevant information as arguments
        subprocess.run(['python', 'diff.py', video1_path, video2_path, str(base_start_frame), str(alt_start_frame), str(videos_switched)])

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output.decode('utf-8').strip()}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

'''
python main.py scenario_base.mp4 scenario_alt2.mp4
python main.py Gauge_base.mp4 Gauge_diff2.mp4   
python main.py 3D_Cockpit_4.mp4 3D_Cockpit_3.mp4  
'''   