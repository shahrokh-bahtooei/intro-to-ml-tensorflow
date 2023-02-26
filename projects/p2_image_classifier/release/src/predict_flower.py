import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

# These two will hide the tip about performance improvement on TensorFlow and the warning about Autograph.
# They must be placed before tensorflow import.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import PIL
import numpy as np
import pandas as pd
import tensorflow as tf

parent = Path(__file__).resolve().parent
default_model_path = str(Path.joinpath(parent, 'flower_classifier_SavedModel'))
default_labels_path = str(Path.joinpath(parent, 'label_map.json'))

DEFAULT_TOP_K = 5

MODEL_IMAGE_SIZE = 224


def receive_arguments(testing_args=None):
    parser = argparse.ArgumentParser(description='Recognize a flower')

    parser.add_argument('image', type=str, metavar='IMAGE_PATH',
                        help='Path to the flower image')
    parser.add_argument('-m', '--model', type=str, metavar='MODEL_PATH',
                        default=default_model_path,
                        help='Path to the classifier model')
    parser.add_argument('-t', '--top_k', type=int, metavar='TOP_K',
                        default=DEFAULT_TOP_K,
                        help='Return the top K most likely classes')
    parser.add_argument('-c', '--category_names', type=str, metavar='CAT_NAMES_PATH',
                        default=default_labels_path,
                        help='Path to a JSON file mapping labels to flower names')

    return parser.parse_args(testing_args)


def load_model(model_path: str) -> tf.keras.Model:
    try:
        model = tf.keras.models.load_model(model_path)

        return model

    except (IOError, ImportError, OSError, TypeError) as e:
        try:
            print(e.msg)
        except AttributeError:
            print(e)
        finally:
            print('Model could not be loaded!')
            exit()


def load_image(image_path: str) -> np.ndarray:
    image = PIL.Image.open(image_path)
    return np.asarray(image)


def process_image(image: np.ndarray) -> np.ndarray:
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    image /= 255
    image = np.expand_dims(image, axis=0)
    return image


def predict_all_probs(processed_image, model) -> np.ndarray:
    return model.predict(processed_image)[0]


def get_top_k_with_labels(all_probs, top_k):
    top_k_indices = np.argpartition(all_probs, -top_k)[-top_k:]
    desc_top_k_indices = top_k_indices[np.argsort(-all_probs[top_k_indices])]

    top_k_probs = all_probs[desc_top_k_indices].tolist()
    top_k_labels = (desc_top_k_indices + 1).astype(str).tolist()

    return top_k_probs, top_k_labels


def predict(model_path: str, image_path: str, top_k: int) -> (List[float], List[str]):
    model = load_model(model_path)

    image = load_image(image_path)
    processed_image = process_image(image)

    all_probs = predict_all_probs(processed_image, model)

    top_k_probs, top_k_labels = get_top_k_with_labels(all_probs, top_k)

    return top_k_probs, top_k_labels


def load_category_names(category_names_path: str) -> dict:
    with open(category_names_path, 'r') as f:
        category_names = json.load(f)

    return category_names


def map_labels_to_categories(labels: list, category_names: dict):
    return [category_names[label] for label in labels]


def print_prediction(top_k_probs: List[float],
                     top_k_labels: List[str],
                     category_names_path: str) -> None:
    if category_names_path == '':
        table = pd.DataFrame(data={'Label': top_k_labels, 'Probability': top_k_probs})
        print(table.to_string())
    else:
        category_names = load_category_names(category_names_path)
        top_k_flower_names = map_labels_to_categories(top_k_labels, category_names)
        table = pd.DataFrame(data={'Flower Name': top_k_flower_names, 'Probability': top_k_probs})
        print(table.to_string())


def main():
    args = receive_arguments()
    probs, labels = predict(args.model, args.image, args.top_k)
    print_prediction(probs, labels, args.category_names)


if __name__ == '__main__':
    main()
