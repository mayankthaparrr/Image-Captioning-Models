# Image Captioning Models

An image-captioning project that generates a natural-language description for an image using an encoder-decoder architecture. The repository compares two CNN feature extractors—**ResNet-50** and **MobileNetV2**—paired with an **LSTM** caption decoder and trained on the Flickr30k dataset.

## How it works

1. A pretrained CNN converts each image into a fixed-length visual feature vector.
2. A learned encoder projects that vector into the initial hidden and cell states of an LSTM.
3. The LSTM predicts caption tokens one at a time, starting with `<SOS>` and stopping at `<EOS>`.

Image features are cached locally as pickle files so the CNN does not need to run again before each training session.

## Implementations

| Model | Feature extractor | Feature size | Source |
| --- | --- | ---: | --- |
| ResNet-50 + LSTM | Pretrained ResNet-50, classification head removed | 2,048 | `image_caption_resnet50_flickr30k (3).py` |
| MobileNetV2 + LSTM | Pretrained MobileNetV2, classifier replaced with identity | 1,280 | `image-captioning-mobilenet-flickr30k .ipynb` |

Both variants use a 512-dimensional embedding, a 512-dimensional LSTM hidden state, dropout of 0.3, Adam (`lr=0.001`), and 20 training epochs. Images are split 80% / 10% / 10% for training, validation, and testing with a fixed random seed.

## Kaggle notebooks

Run the implementations directly on Kaggle:

- [ResNet-50 + LSTM — Flickr30k](https://www.kaggle.com/code/mayankthaparr/image-caption-resnet50-flickr30k)
- [MobileNetV2 + LSTM — Flickr30k](https://www.kaggle.com/code/mayankthaparr/image-captioning-mobilenet-flickr30k)

## Requirements

- Python 3.9+
- PyTorch and Torchvision
- pandas
- spaCy and the English model
- Pillow
- matplotlib
- NLTK

Install the dependencies:

```bash
pip install torch torchvision pandas spacy pillow matplotlib nltk
python -m spacy download en_core_web_sm
```

CUDA is used automatically when it is available; otherwise the code runs on CPU.

## Dataset

The code expects the Flickr30k image dataset and its `results.csv` captions file. In the original Kaggle setup, the paths are:

```text
/kaggle/input/datasets/hsankesara/flickr-image-dataset/
└── flickr30k_images/
    ├── flickr30k_images/   # image files
    └── results.csv         # captions metadata
```

For a local setup, download Flickr30k through a source you are authorized to use, then update `ROOT_DIR` and `CAPS_FILE` near the top of the chosen implementation. The CSV must include `image_name` and ` comment` columns, as used by the source dataset.

## Run

### ResNet-50 version

The Python script is exported from a notebook and can be run cell-by-cell in VS Code, Jupyter, or another Python notebook environment. Update the dataset paths first, then run the cells in order:

```bash
jupyter notebook
```

Open `image_caption_resnet50_flickr30k (3).py` as an interactive notebook, or execute it after setting the dataset paths.

### MobileNetV2 version

Open and run the notebook sequentially:

```bash
jupyter notebook "image-captioning-mobilenet-flickr30k .ipynb"
```

The workflow performs feature extraction, creates the data split, trains the model, saves the best validation checkpoint as `best_model.pth`, plots `loss_curve.png`, generates sample captions, and evaluates the test set.

## Outputs

Running either implementation can create these local artifacts:

- `flickr30k_features.pkl` or `flickr30k_features_MNV2.pkl` — cached CNN features
- `best_model.pth` — the checkpoint with the lowest validation loss
- `loss_curve.png` — training and validation loss chart

These files can be large and are best kept out of version control.

## Evaluation

The implementations generate captions greedily and report:

- BLEU-1 and BLEU-4
- Token-level precision, recall, and F1 score

For the recorded results, generated captions, and training outputs, refer to the [Kaggle notebooks](#kaggle-notebooks).

## Repository structure

```text
.
├── image_caption_resnet50_flickr30k (3).py
└── image-captioning-mobilenet-flickr30k .ipynb
```

## Notes

- Captions are tokenized with spaCy and a vocabulary is built from words observed at least five times.
- The current decoder uses greedy decoding with a maximum caption length of 20 tokens.
- The model checkpoints are saved as state dictionaries; retain the corresponding vocabulary and architecture settings when loading them for inference.

## License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for details.
