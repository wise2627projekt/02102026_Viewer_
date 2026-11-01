# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: the PhotoVec Viewer only.
#   Windows: a folder with PhotoVecViewer.exe (the libraries are packed next to it)
#   macOS:   PhotoVecViewer.app (Apple Silicon), which also registers itself as the program for *.photovecshow files
import sys

a = Analysis(["photovec_viewer.py"], pathex=["."], binaries=[], datas=[],
             hiddenimports=["PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore"],
             hookspath=[], runtime_hooks=[], noarchive=False,
             excludes=["tkinter", "matplotlib", "scipy", "skimage", "PIL", "pytest", "IPython", "PySide6.QtNetwork", "PySide6.QtQml",
                       "PySide6.QtQuick", "PySide6.QtPdf", "PySide6.QtSvg", "PySide6.QtOpenGL", "PySide6.QtWebEngineCore"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="PhotoVecViewer", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PhotoVecViewer")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="PhotoVecViewer.app", bundle_identifier="app.photovec.viewer",
                 info_plist={"CFBundleName": "PhotoVec Viewer", "CFBundleDisplayName": "PhotoVec Viewer", "CFBundleShortVersionString": "1.0",
                             "NSHighResolutionCapable": True, "LSMinimumSystemVersion": "11.0",
                             "CFBundleDocumentTypes": [{"CFBundleTypeName": "PhotoVec show", "CFBundleTypeExtensions": ["photovecshow"],
                                                        "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Owner"}]})
