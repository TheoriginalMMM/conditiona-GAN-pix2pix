# Pix2Pix Conditional GAN Project

## Overview
This repository is dedicated to the implementation of the Pix2Pix Conditional Generative Adversarial Network (GAN) for image-to-image translation tasks. The generator and discriminator were trained from scratch for two datasets: maps to satellite views (and vice versa) and faces to comics translation.

## Datasets
- **Maps Dataset**: Available on [Kaggle](https://www.kaggle.com/datasets/alincijov/pix2pix-maps), this dataset allows the model to translate between maps and satellite views.
- **Face to Comics Dataset**: Also available on [Kaggle](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic), this dataset is used for translating human faces to comic styles.

## Files in the Repository
- `conf.py`: Configuration parameters and settings.
- `datasets.py`: Dataset loading and preprocessing script.
- `discriminator.py`: Discriminator network architecture.
- `export tensorboard results.ipynb`: Jupyter notebook for exporting TensorBoard results.
- `generator.py`: Generator network architecture.
- `paper.pdf`: Original Pix2Pix paper.
- `train.py`: Training script for the Pix2Pix model.
- `utils.py`: Utility functions.
- A report written for a Master's degree course will also be added to this repository.

## Original Pix2Pix Paper
The Pix2Pix model is based on the paper titled "Image-to-Image Translation with Conditional Adversarial Networks" by Isola et al., 2017, which is included in this repository.

## Citing Resources
Please cite the original publishers of the datasets and the Pix2Pix paper as follows:

```bibtex
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```

- Maps Dataset citation: Alin Cijov, "Pix2Pix Maps Dataset", Kaggle, [https://www.kaggle.com/datasets/alincijov/pix2pix-maps](https://www.kaggle.com/datasets/alincijov/pix2pix-maps).
- Face to Comics Dataset citation: Defileroff, "Comic Faces Paired Synthetic Dataset", Kaggle, [https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic).

## Usage
To train the Pix2Pix model, execute the `train.py` script with the appropriate dataset and configuration settings in `conf.py`. Ensure the datasets are downloaded and structured as expected by `datasets.py`.

## Contributions
This project has been developed for educational purposes within a Master's degree program. Contributions to enhance the model and its applications are welcome.

## License
This code is distributed under the same terms as the datasets and the Pix2Pix paper. Adherence to the respective licenses is required when using and distributing this code.
