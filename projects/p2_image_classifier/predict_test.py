import pytest
import pandas as pd
from collections import namedtuple
from pathlib import Path

from predict import (receive_arguments, predict, map_labels_to_categories,
                     print_prediction,
                     DEFAULT_TOP_K, load_category_names)

SampleImage = namedtuple('SampleImage', ['label', 'path'])

images = [
    SampleImage(label='61', path='test_images/cautleya_spicata.jpg'),
    SampleImage(label='2', path='test_images/hard-leaved_pocket_orchid.jpg'),
    SampleImage(label='59', path='test_images/orange_dahlia.jpg'),
    SampleImage(label='52', path='test_images/wild_pansy.jpg')
]

top_ks = [
    '1',
    '3',
    '102'
]

model_path = 'flower_classifier.h5'

cat_names_path = 'label_map.json'


class TestPredict:

    @pytest.fixture(params=images, ids=[
        '61',
        '2',
        '59',
        '52'
    ])
    def image(self, request):
        return request.param

    @pytest.fixture(params=top_ks)
    def top_k(self, request):
        return request.param

    @pytest.fixture
    def ness_args(self, image):
        return receive_arguments([image.path,
                                  model_path])

    @pytest.fixture
    def all_args(self, image, top_k):
        return receive_arguments([image.path,
                                  model_path,
                                  '--top_k', top_k,
                                  '--category_names', cat_names_path])

    def test_receive_arguments_ness(self, ness_args, image):
        assert ness_args.image == image.path
        assert ness_args.model == model_path
        assert ness_args.top_k == DEFAULT_TOP_K
        assert ness_args.category_names == ''

    def test_receive_arguments_all(self, all_args, image, top_k):
        assert all_args.image == image.path
        assert all_args.model == model_path
        assert all_args.top_k == int(top_k)
        assert all_args.category_names == cat_names_path

    def test_load_category_names(self):
        category_names = load_category_names(cat_names_path)

        assert category_names['1'] == 'pink primrose'
        assert category_names['102'] == 'blackberry lily'

    def test_map_labels_to_categories(self):
        labels = ['1', '0', '101']
        category_names = {'0': 'pink primrose',
                          '1': 'hard-leaved pocket orchid',
                          '101': 'blackberry lily'}

        act_categories = map_labels_to_categories(labels, category_names)
        exp_categories = ['hard-leaved pocket orchid',
                          'pink primrose',
                          'blackberry lily']

        assert act_categories == exp_categories

    def test_print_prediction_with_ness_args(self, capfd):
        probs = [.8, .1, .03, .02, .01]
        labels = ['0', '1', '2', '3', '101']

        print_prediction(probs, labels, '')
        act_out, acc_err = capfd.readouterr()

        df = pd.DataFrame(data={'Label': labels, 'Probability': probs})
        exp_out = f'{df.to_string()}\n'

        assert act_out == exp_out

    def test_print_prediction_with_all_args(self, capfd):
        probs = [0.950380, 0.008613, 0.004906, 0.003384]
        labels = ['60', '23', '45', '10']
        categories = ['pink-yellow dahlia', 'fritillary', 'bolero deep blue', 'globe thistle']

        print_prediction(probs, labels, cat_names_path)
        act_out, acc_err = capfd.readouterr()

        df = pd.DataFrame(data={'Flower Name': categories, 'Probability': probs})
        exp_out = f'{df.to_string()}\n'

        assert act_out == exp_out

    def test_predict_with_ness_args(self, ness_args, image):
        probs, labels = predict(ness_args.model, ness_args.image, ness_args.top_k)

        expected_label = image.label

        assert labels[0] == expected_label
        assert probs[0] > .3
        assert len(probs) == len(labels) == ness_args.top_k

    def test_predict_with_all_args(self, all_args):
        probs, labels = predict(all_args.model, all_args.image, all_args.top_k)
        category_names = load_category_names(all_args.category_names)
        detected_flower_category = map_labels_to_categories(labels, category_names)[0]

        image_name = Path(all_args.image).stem
        expected_flower_category = image_name.replace('_', ' ')

        assert detected_flower_category == expected_flower_category
        assert probs[0] > .3
        assert len(probs) == len(labels) == all_args.top_k
