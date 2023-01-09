# ND230 Introduction to Machine Learning with TensorFlow Nanodegree Program

Projects and exercises for the Udacity Intro to Machine Learning with TensorFlow course.

## Getting Started

The major problems I've solved during this nanodegree in the format of Jupyter notebooks for Supervised Learning project
and Neural Networks exercises and project can be both viewed and re-run through the following links on Google
Colaboratory and Kaggle platforms. The notebook for Unsupervised Learning project can only be viewed statically—for it's
been supposed not to include the licensed datasets.

### Supervised Learning

👉 Project: **Finding Donors for CharityML**

[![Open in Colab][colab-svg]][colab-supervised-proj]
[![Open in Kaggle][kaggle-svg]][kaggle-supervised-proj]

### Introduction to Neural Networks with TensorFlow

| #   | Lesson                                                                               | Link                                        | 
|-----|--------------------------------------------------------------------------------------|---------------------------------------------|
| 1   | Introduction to TensorFlow and using tensors                                         | [![Open in Colab][colab-svg]][colab-nn-ex1] | 
| 2   | Building fully-connected neural networks with TensorFlow                             | [![Open in Colab][colab-svg]][colab-nn-ex2] | 
| 3   | How to train a fully-connected network with backpropagation on MNIST                 | [![Open in Colab][colab-svg]][colab-nn-ex3] | 
| 4   | Exercise–train a neural network on Fashion-MNIST                                     | [![Open in Colab][colab-svg]][colab-nn-ex4] | 
| 5   | Using a trained network for making predictions and validating networks               | [![Open in Colab][colab-svg]][colab-nn-ex5] | 
| 6   | How to save and load trained models                                                  | [![Open in Colab][colab-svg]][colab-nn-ex6] | 
| 7   | Load image data with ImageDataGenerator, and also data augmentation                  | [![Open in Colab][colab-svg]][colab-nn-ex7] | 
| 8   | Use transfer learning to train a state-of-the-art image classifier for dogs and cats | [![Open in Colab][colab-svg]][colab-nn-ex8] |

👉 Project: **Create Your Own Image Classifier—TensorFlow**

[![Open in Colab][colab-svg]][colab-nn-proj]
[![Open in Kaggle][kaggle-svg]][kaggle-nn-proj]

👨🏻‍💻👩🏻‍💻 The resulted app is named as `flower_recognizer`. To run it locally on your computer, download the built binaries for your OS from the release section:

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/shahrokh-bahtooei/intro-to-ml-tensorflow)](https://github.com/shahrokh-bahtooei/intro-to-ml-tensorflow/releases)

### Unsupervised Learning

👉 Project: **Creating Customer Segments with Arvato**

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/projects/p3_identify_customer_segments/identify_customer_segments.ipynb)
[![Udacity Review](https://badgen.net/badge/view%20review/pdf/red?icon=pdf)](https://github.com/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/projects/p3_identify_customer_segments/Udacity_review.pdf)

## Installation

To run the notebooks and scripts in this repo locally on a computer with the _exact_ dependencies, just run the
following commands in a terminal, and open the link it offers at the end for a jupyter notebook:

```sh
python3 -m pip install jupyter-repo2docker
repo2docker https://github.com/shahrokh-bahtooei/intro-to-ml-tensorflow.git
```

Prerequisites:

[![Python 3.6](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Docker](https://badgen.net/badge/icon/Docker?icon=docker&label)](https://https://docker.com/)

Alternatively, this repo could be run over a conda environment:

```sh
git clone https://github.com/shahrokh-bahtooei/intro-to-ml-tensorflow.git
cd intro-to-ml-tensorflow
conda env create --prefix ./env -f environment_minimal.yml
conda activate env/
jupyter notebook
```

Prerequisites:

[![Miniforge](https://img.shields.io/badge/Miniforge-gray.svg)](https://github.com/conda-forge/miniforge)
/ [![Miniconda](https://img.shields.io/badge/Miniconda-gray.svg)](https://docs.conda.io/en/latest/miniconda.html)


[colab-svg]: <https://colab.research.google.com/assets/colab-badge.svg>

[kaggle-svg]: <https://kaggle.com/static/images/open-in-kaggle.svg>


[colab-supervised-proj]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/projects/p1_charityml/finding_donors.ipynb>

[kaggle-supervised-proj]: <https://www.kaggle.com/notebooks/welcome?src=https://github.com/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/projects/p1_charityml/finding_donors.ipynb>


[colab-nn-ex1]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/lessons/intro-to-tensorflow/Part_1_Introduction_to_Neural_Networks_with_TensorFlow_(Exercise).ipynb>

[colab-nn-ex2]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/lessons/intro-to-tensorflow/Part_2_Neural_networks_with_TensorFlow_and_Keras_(Exercise).ipynb>

[colab-nn-ex3]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/lessons/intro-to-tensorflow/Part_3_Training_Neural_Networks_(Exercise).ipynb>

[colab-nn-ex4]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/lessons/intro-to-tensorflow/Part_4_Fashion_MNIST_(Exercise).ipynb>

[colab-nn-ex5]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/lessons/intro-to-tensorflow/Part_5_Inference_and_Validation_(Exercise).ipynb>

[colab-nn-ex6]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/lessons/intro-to-tensorflow/Part_6_Saving_and_Loading_Models.ipynb>

[colab-nn-ex7]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/lessons/intro-to-tensorflow/Part_7_Loading_Image_Data_(Exercise).ipynb>

[colab-nn-ex8]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/lessons/intro-to-tensorflow/Part_8_Transfer_Learning_(Exercise).ipynb>


[colab-nn-proj]: <https://colab.research.google.com/github/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/projects/p2_image_classifier/build_flower_classifier.ipynb>

[kaggle-nn-proj]: <https://www.kaggle.com/notebooks/welcome?src=https://github.com/shahrokh-bahtooei/intro-to-ml-tensorflow/blob/master/projects/p2_image_classifier/build_flower_classifier.ipynb>

---
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer
to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
