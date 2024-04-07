# VideoDiffDetector

Automated video comparison tool using Python and OpenCV. Syncs two video feeds, detects and annotates differences, and logs these differences. Developed for the 402 Software Engineering Group at Robins AFB.

## Description

[empty]

## Technologies Used

- Python
- OpenCV
- VSCode
- Git
- more TBD

## Prerequisites for contributing to or running this application:

Python 3.12.1:

- To check if you have **Python 3.12.1**, run
  <p>**python -V**</p>
  in a command prompt.

#### If you don't have Python, please follow these steps:

- Download the installer from [Python's official website](https://www.python.org/downloads/windows/). Get the installer for your OS, not the embeddable package.
- When you run the installer, ensure you check the boxes `add python.exe to PATH` and `use admin privileges when installing py.exe`. If you don't, python will not work in VSCode.
- Select `Customize installation` and ensure `pip` is checked.
- Click "Next" and ensure that `Add Python to environment variables` is checked.
- Click "Install"
<p color="red"></p>

OpenCV:

- in your command prompt, terminal, or VSCode terminal, run
`pip install opencv-contrib-python`
<!--- once it finishes, run **pip install caer** -->

VSCode:

- Download and install from the official VSCode website if you don't already have it: https://code.visualstudio.com/download

Git:

- Check if you have git installed by using
  `git --version`
- If you don't have it or don't have the most recent version, download and install from the offical Git website: https://git-scm.com/downloads

VSCode Python Extension:

- In VSCode, click the extensions button on the left, search for `Python`, and install it.

Potentially More TBD:

- This will be updated as the application is updated.

## Cloning the Repository

- In VSCode, press `` CTRL + ` (control + backtick) `` to open the terminal.
- Navigate to the directory you want to clone this repo using
  `cd path/to/your/directory`
- Clone the repo using
  `git clone https://github.com/HRosser15/Video-Diff-Detector.git`
- Navigate into the cloned repo using
  `cd Video-Diff-Detector`

## Where videos should be placed for analysis

- Videos to be analyzed MUST be in the "Videos" folder.
- By default, multiple videos are available to test the application on.

## Running the program

- In a terminal, navigate to your cloned repo, where your current directory should be `Video-Diff-Detector`
- Enter in `python main.py <video1_file_name.ext> <video2_file_name.ext>`
  -- Replace `<video1_file_name.ext>` and `<video2_file_name.ext>` with your desired file names.
  -- By default, the application will add the necessary filepath to the beginning of these inputs so they are retrieved from `Video-Diff-Detector/Videos/`

## To Begin Contributing

- Create your own branch with
  `git checkout -b your-branch-name`

## Pushing Your Local Changes to the GitHub Repo

- If you haven't already, set your global identity with
  `git config --global user.name "Your Name"`
  and
  `git config --global user.email "your.email@example.com"`
- To stage your local changes, use
  `git add file1_name.ext file2_name.ext** (if you're staging specific files in the branch)`
  <p>- This is useful for committing only files you have completed when some files are a WIP.</p>
  <p>- It is also useful when you have completed a lot of work, and want to separate your commits to make the repo history easier to read</p>
  or
  `git add .` (with the period if you're staging all files in the branch)</p>
- To commit the staged changes:
  `git commit`
  This will pull up a COMMIT file. Whatever you type in here will save as the commit message. Please be descriptive here. a general commit message should look like:

```bash
[short summary of what you did]

[detailed summary of what you did]
```

so a commit might look like:

```bash
Optimized thresholds for abs_diff.py

Changed threshold in frame_difference() to 20.
Created new function to filter out contours that were too small.
...
```

Save this file with "CTRL + S" and close it with "CTRL + W".

- To push your local changes to the repo, use
  `git push origin your-branch-name`
- If you forgot your branch name or want to confirm you are on the right branch, use
  `git branch`
  to view the list of branches. Your current branch should be marked with an asterisk (\*) and be a different color.
- To merge your branch with the main branch, **Submit a pull request on GitHub** and wait for someone to review it.
