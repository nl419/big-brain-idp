# big-brain-idp
BIG BRAINS ONLY

Please use the NumPy format for Docstrings.
https://numpydoc.readthedocs.io/en/latest/format.html

## Installation instructions
- `pip uninstall opencv-python`
- `pip install opencv-contrib-python imutils`
- Open a VS Code Window
- Open the command palette: View -> Command Palette... (Ctrl+Shift+P)
- Start typing "git clone" and click on "Git: Clone"
- Paste in: https://github.com/nl419/big-brain-idp
- Press enter, and select the folder location for the repository on your computer

## Syncing changes from VS Code to GitHub
- Try making a trivial change (e.g. edit the README.md file), then save the file.
- Click the "Source Control" icon (Ctrl+Shift+G) on the left
- Hover your cursor over the "Changes" heading, then click the "+" icon to stage all changes
- Type a message in the text box at the top describing your changes
- Click on the tick icon to commit your changes to your local repo
- If you get an error saying something about needing a username and email, open a terminal window (Ctrl+') and use these commands (edit as appropriate)
```
git config --global user.name "YOUR USERNAME HERE"
git config --global user.email "YOUR EMAIL HERE"
```
- Press the "Sync Changes" button - you may get a prompt to log in to GitHub.

The changes are now synced with the online repo! See https://github.com/nl419/big-brain-idp to check it's updated.

## TODO list

Software:

- [x] Redo checkerboard calibration
- [x] Track orientation
- [x] Make tracking robust vs lighting changes
- [x] Track blocks
- [x] Track target locations
- [x] Navigation code
- [x] Arduino WiFi code
- [x] Laptop WiFi code

Electrical:

- [x] LED display
- [x] Colour detector

Mechanical:

- [x] Final chassis
- [x] Block shover