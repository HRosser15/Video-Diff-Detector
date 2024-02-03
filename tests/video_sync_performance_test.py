import os
import time
import subprocess
from logic_only.test_avg_hash_video_sync import main as avg_hash_simplified_main
from logic_only.test_pixel_wise_video_sync import main as pixelwise_simplified_main

# TO RUN:
# open terminal with "CTRL + `"
# enter 'python tests\\video_sync_performance_test.py' and press enter
# press 'q' to exit the video window

def measure_execution_time(program_path):
    execution_times = []

    for _ in range(200):
        start_time = time.time()
        subprocess.run(['python', program_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

    # Create 'logs' directory if it doesn't exist
    logs_directory = 'tests/logs'
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    # Save execution times to logs
    log_filename = os.path.join(logs_directory, os.path.basename(program_path).replace('.py', '_execution_time_log.txt'))
    with open(log_filename, 'w') as log_file:
        for time_val in execution_times:
            log_file.write(f'{time_val}\n')

        avg_execution_time = sum(execution_times) / len(execution_times)
        log_file.write(f'average execution time: {avg_execution_time}\n')

def main():
    script_path = os.path.abspath(__file__)

    # Measure execution time for avg_hash_video_sync.py
    avg_hash_program_path = 'logic_only/test_avg_hash_video_sync.py'
    measure_execution_time(avg_hash_program_path)

    # Measure execution time for pixelwise_video_sync.py
    pixelwise_program_path = 'logic_only/test_pixelwise_video_sync.py'
    measure_execution_time(pixelwise_program_path)

    # Measure execution time for pixelwise_video_sync.py
    dct_phash_program_path = 'logic_only/test_dct_phash_video_sync.py'
    measure_execution_time(dct_phash_program_path)

if __name__ == '__main__':
    main()
