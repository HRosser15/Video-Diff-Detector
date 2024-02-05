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
- If you don't have it, you can download it from Python's official website: https://www.python.org/downloads/windows/
- Ensure you customize your installation to add Python to your PATH environment variable.

OpenCV: 
- in your command prompt, run
      <p>**pip install opencv-contrib-python**</p>
<!--- once it finishes, run **pip install caer** -->

VSCode:
- Download and install from the official VSCode website if you don't already have it: https://code.visualstudio.com/download

Git: 
- Check if you have git installed by using
          <p>**git --version**</p>
- If you don't have it or don't have the most recent version, download and install from the offical Git website: https://git-scm.com/downloads
- Open VSCode, click the extensions button on the left, search for '**Git**' and install it.

Python VSCode Extension:
- In VSCode, click the extensions button on the left, search for '**Python**', and install it.

Potentially More TBD:
- This will be updated as the application is updated.


## Cloning the Repository
- In VSCode, press **"CTRL + `"** to open the terminal.
- Navigate to the directory you want to clone this repo using
          <p>**cd path/to/your/directory**</p>
- Clone the repo using
      <p>**git clone https://github.com/HRosser15/Video-Diff-Detector.git**</p>
- Navigate into the cloned repo using
      <p>**cd Video-Diff-Detector'**</p>

## To Begin Contributing
- Create your own branch with
      <p>**git checkout -b your-branch-name**<p>

## Pushing Your Local Changes to the GitHub Repo
- If you haven't already, set your global identity with
      <p>**git config --global user.name "Your Name"**</p>
  and
      <p>**git config --global user.email "your.email@example.com"**</p>
- To stage your local changes, use
      <p>**git add file1_name.ext file2_name.ext** (if you're staging specific files in the branch)</p> 
        <p>- This is useful for committing only files you have completed when some files are a WIP.</p>
        <p>- It is also useful when you have completed a lot of work, and want to separate your commits to make the repo history easier to read</p>
  or
        <p>**git add .** (with the period if you're staging all files in the branch)</p>
- To commit the staged changes:
      <p>**git commit -m "Your message here"**</p>
- To push your local changes to the repo, use
      <p>**git push origin your-branch-name**</p>
- If you forgot your branch name or want to confirm you are on the right branch, use
      <p>**git branch**</p>
  to view the list of branches.
- To merge your branch with the main branch, **Submit a pull request on GitHub** and wait for someone to review it.
