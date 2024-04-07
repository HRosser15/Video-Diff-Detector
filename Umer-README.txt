# TO RUN
- navigate to \Video-Diff-Detector

#### main.py: 
- run ```python main.py scenario_base.mp4 scenario_alt2.mp4```
- The video names can be changed to any videos in the Videos folder.
- The program will fail if the two videos cannot be successfully synced, so refer to the list of videos below to see which videos can be analyzed together.

#### sync.py: 
- run ```python sync.py scenario_base.mp4 scenario_alt2.mp4```
- The video names can be changed to any videos in the Videos folder.
- The program will fail if the two videos cannot be successfully synced, so refer to the list of videos below to see which videos can be analyzed together.

#### diff.py: 
- run ```python diff.py scenario_base.mp4 scenario_alt2.mp4 0 4``` where the two digits at the end represent the starting frames
- The video names can be changed to any videos in the Videos folder.
- Since this runs without the synchronizer, false positives will be found if incorrect starting frame numbers are input. If it would be helpful, I can add a list of starting frames for each pair of videos to this file.

# How the three files work.
main.py will run the video synchronier (sync.py) and use its outputs to run the difference detector (diff.py) from the correct starting frames.
sync.py takes in two inputs: the name of the base video, and the name of the alt video.
diff.py takes in 4 arguments: both video names, and both starting frames for their respctive videos.

# Output_Video folder
This contains the output video. 
- We will need to include the output name in the command line so they don't overwrite, or we can implement logic to add an incremental number to the output video name.

# Video folder
The Videos folder has different videos for testing.
#### The main videos that need to work are:
- (scenario_base.mp4 && scenario_alt1.mp4)
- (scenario_base.mp4 && scenario_alt2.mp4)
- (Gauge_base.mp4 && Gauge_diff1.mp4)
- (Gauge_base.mp4 && Gauge_diff2.mp4)

#### For 3D videos, the following would be great if they work, but aren't required:
- (3D_cat1.mov && 3D_cat2.mov)
- (3D_Cockpit_3.mov && 3D_Cockpit_4.mov)

## Visualiations and Demos
- These do NOT need to be tested. 
- These are folders and files I made to visualize different pieces of the program. 
- While the application doesn't rely on them, they do contain some helpful visuals and they show thresholds that work for specific use cases.
    -- For instance, in Visualizations/Video_Sync, the 3d_cat, 3d_cockpit, and Gauge files all contain thresholds that works best for those specific videos.
    -- We can either use these values to find a value that works for all use cases, or we can use the values in a conditional flow where if the application fails, the thresholds are raised or lowered a specific amount and then tried again until the application works.
    -- We could also make options in the command line such as ```--threshold high``` for complex videos and ```--threshold low``` for simple videos.