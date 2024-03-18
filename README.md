# TTS project 

## Project description

In this project FastSpeech2 ([paper](https://arxiv.org/pdf/2006.04558.pdf)) model was implemented for TTS task.

## Project structure
- **/hw_asr** - project scripts
- **/generated** - generated samples for evaluation
- **/waveglow** - converter from MEL to audio with example notebook
- _install_dependencies.sh_ - script for dependencies installation
- _requirements.txt_ - Python requirements list
- _train.py_ - script to run train
- _test.py_ - script to run test
- _features_makers.py_ - Pitch and Energy extractors
- _synthesizer.py_ - class to synthesize speech from text
- _preprocess_data.ipynb_ - saves features for Dataset

## Installation guide

It is strongly recommended to use new virtual environment for this project.

To install all required dependencies and final model run:
```shell
./install_dependencies.sh
```

## Reproduce results
To run train with _LJSpeech_ datasets:
```shell
python -m train -c hw_tts/configs/train_config.json
```

To run test inference (texts are hard-coded in test.py):
```shell
python test.py -c hw_tts/configs/test_config.json -r best_model/best_model.pth
```

Current results are stored in **/generated** directory.

## Author
Dmitrii Uspenskii HSE AMI 4th year.
