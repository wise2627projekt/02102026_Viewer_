# PhotoVec Viewer

Plays a photo show (`.photovecshow` file) made with the PhotoVec Show Maker: a clean grid of black-and-white thumbnails; a click brings
a photo forward in colour, with two sliders (names chosen by the maker of the show) and a black arrow back to the grid.

* **Windows program:** download `PhotoVecViewer-Windows.zip` from the *Releases* page (right side of this page), unzip it completely,
  double-click `PhotoVecViewer.exe`. It asks which show to play. Windows may say "Windows protected your PC" (the program is not
  signed): click *More info*, then *Run anyway*.
* **Mac (Apple Silicon):** download `PhotoVecViewer-macOS.zip`, double-click it, open `PhotoVecViewer.app`. First time: right-click ->
  *Open* (macOS 15+: *System Settings -> Privacy & Security -> Open Anyway*). A show can be double-clicked or dropped on the app.
* **Keys:** Esc / Backspace / the arrow = back, left/right = first slider, up/down = second slider, F11 or Cmd+F (Ctrl+F) = full screen.
* **From source:** `run_from_source.bat` (Windows) or `run_from_source.command` (Mac), Python 3.9 or newer. Only three libraries are needed
  (numpy, OpenCV, PySide6).

Licences of the libraries: PySide6 / Qt for Python (LGPL-3.0), OpenCV (Apache-2.0), numpy (BSD). Check them if you distribute the program
yourself - this is not legal advice.
