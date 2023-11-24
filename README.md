# ASR project 

## Project description
This is HW1 in the course Deep Learning for Sound Processing.

In this project DeepSpeech2 model was implemented for ASR task.

## Project structure
- **/hw_asr** - project scripts
- **/requirements_loaders** - shell scripts to load requirements
- **test_data** - additional small test dataset
- _install_dependencies.sh_ - script for dependencies installation
- _requirements.txt_ - Python requirements list
- _train.py_ - script to run train
- _test.py_ - script to run test

## Installation guide

It is strongly recommended to use new virtual environment for this project.

To install all required dependencies and final model run:
```shell
./install_dependencies.sh
```

## Reproduce results
To run train with _LibriSpeech_ _train-100_ and _train-360_ datasets:
```shell
python -m train -c hw_asr/configs/train_config.json
```

To run test inference with _LibriSpeech_ _test-clean_ and _test-other_ datasets:
```shell
python test.py \
   -c hw_asr/configs/test-clean_config.json \
   -r best_model/model_best.pth \
   -o test-clean_result.json \
   -b 50

python test.py \
   -c hw_asr/configs/test-other_config.json \
   -r best_model/model_best.pth \
   -o test-other_result.json \
   -b 50
```

There is additional Jupyter Notebook _hw_asr/tests/calc_stats.ipynb_ to calculate CER and WER metrics from .json file created by _test.py_.

## Author
Dmitrii Uspenskii HSE AMI 4th year.
