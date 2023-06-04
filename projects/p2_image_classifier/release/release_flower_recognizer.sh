#!/bin/sh
pyinstaller \
 --add-binary 'resources/flower_classifier_SavedModel:Resources/model' \
 --add-data 'resources/label_map.json:Resources' \
 --add-data 'resources/test_images:Resources/test_images' \
 --name flower_recognizer \
 cli.py

# valid values for --target-arch flag on mac: 'x86_64', 'arm64', 'universal2'