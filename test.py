import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_tts.model as module_model
from hw_tts.trainer import Trainer
from hw_tts.utils import ROOT_PATH
from hw_tts.utils.object_loading import get_dataloaders
from hw_tts.utils.parse_config import ConfigParser
from synthesizer import Synthesizer
from scipy.io.wavfile import write


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    synthesizer = Synthesizer(model)

    sentences = [
        'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
        'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
        'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space'
    ]

    param_mapping = {
        'alpha': 'duration',
        'beta': 'pitch',
        'gamma': 'energy',
    }
    
    for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
        dir = f'generated/sentence_{idx}'
        os.makedirs(dir, exist_ok=True)
        dir_regular = os.path.join(dir, 'regular')
        os.makedirs(dir_regular, exist_ok=True)
        regular_params = {
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 1.0
        }
        regular_audio = synthesizer.synthesize(sentence, regular_params)
        write(os.path.join(dir_regular, 'regular.wav'), 22050, regular_audio)

        dir_plus_20 = os.path.join(dir, 'plus_twenty_percent')
        os.makedirs(dir_plus_20, exist_ok=True)
        for param in regular_params.keys():
            param_name = param_mapping[param]
            params = regular_params.copy()
            params[param] = 1.2
            audio = synthesizer.synthesize(sentence, params)
            write(os.path.join(dir_plus_20, f'{param_name}.wav'), 22050, audio)
        plus_twenty_params = {
            'alpha': 1.2,
            'beta': 1.2,
            'gamma': 1.2
        }
        audio = synthesizer.synthesize(sentence, plus_twenty_params)
        write(os.path.join(dir_plus_20, 'all_params.wav'), 22050, audio)

        dir_minus_20 = os.path.join(dir, 'minus_twenty_percent')
        os.makedirs(dir_minus_20, exist_ok=True)
        for param in regular_params.keys():
            param_name = param_mapping[param]
            params = regular_params.copy()
            params[param] = 0.8
            audio = synthesizer.synthesize(sentence, params)
            write(os.path.join(dir_minus_20, f'{param_name}.wav'), 22050, audio)
        plus_twenty_params = {
            'alpha': 0.8,
            'beta': 0.8,
            'gamma': 0.8
        }
        audio = synthesizer.synthesize(sentence, plus_twenty_params)
        write(os.path.join(dir_minus_20, 'all_params.wav'), 22050, audio)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output)
