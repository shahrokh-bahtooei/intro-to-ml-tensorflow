#!/bin/sh
pyinstaller \
 --add-binary 'src/flower_classifier_SavedModel:flower_classifier_SavedModel' \
 --add-data 'src/label_map.json:.' \
 --name flower_recognizer \
 cli.py

# valid values for --target-arch flag on mac: 'x86_64', 'arm64', 'universal2'