import sys
import wandb
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn as nn
import torch
from pathlib import Path
import argparse
import os
import numpy as np
from tqdm import tqdm
import hadamard_utils
from welford_torch import OnlineCovariance

from dataloader import R1Dataset

sys.path.append('../fake_quant')

from rotation_utils import get_orthogonal_matrix
from utils import supported_models, supported_datasets
from model_utils import get_model

os.environ['WANDB_API_KEY'] = "XXXXXXXXXXXXXXX"
os.environ['WANDB_MODE'] = "offline"

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


default_path = '../data/'

def train_R1(dataset, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("---> start calculating R1 analytically")

    hidden_size = args.hidden_size
    
    # Initialize Welford object
    w = OnlineCovariance(dtype=torch.float64)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    for batch in tqdm(dataloader, desc="Calculating Covariance"):
        # Reshape the batch
        x = batch.to(device).view(-1, hidden_size)
        
        w.add_all(x.to(dtype=torch.float64))

    # 4. Get the eigenvectors directly from the Welford object!
    # w.eig_vec contains the eigenvectors (U) of the covariance matrix
    # We cast to float32 to match the rest of the network operations
    U = w.eig_vec.to(device=device, dtype=torch.float64)

    # Generate normalized Hadamard matrix H
    H = hadamard_utils.random_hadamard_matrix(args.hidden_size, device=device).to(torch.float64)

    # Calculate trained R1 = U @ H
    R1 = U @ H
    print(f"R1 shape: {R1.shape}")

    print("---> R1 calculation done ")
    return R1.cpu(), U.cpu()


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, choices=supported_models,
                        help='model name.')
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        choices=supported_datasets,
                        help='Dataset for Evaluation (default: wikitext2)',)
    parser.add_argument('--nsamples', type=int, default=128, help='Number of train sample.')
    parser.add_argument('--train_subset_size', type=float, default=0.1, help='')
    parser.add_argument('--calib_sample', type=int, default=128,
                        help='Number of sample.')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--bsz', type=int, default=64,
                        help='Batch-size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to use wandb for logging (default: False)')
    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether save learned r1 (default: True).')

    args = parser.parse_args()

    now = datetime.now()
    run_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    setattr(args, 'run_time', run_time)

    save_folder = f'corrmax.{args.bsz}'
    setattr(args, 'save_folder', save_folder)

    data_path = os.path.join(
        default_path,
        f"train_data/{args.calib_dataset}_{args.calib_sample}samples/{args.model.split('/')[-1]}/r1_train")
    setattr(args, 'data_path', data_path)
    print(f'---> data path: {data_path}')
    save_path = os.path.join(
        default_path,
        f"trained_rotation/{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r1")
    setattr(args, 'save_path', save_path)

    return args, now


if __name__ == "__main__":
    args, run_time = parser_gen()

    model = get_model(args.model, args.hf_token)
    setattr(args, 'hidden_size', model.config.hidden_size)
    del model

    dataset = R1Dataset(args.data_path, nsamples=args.nsamples)
    print(f'---> start time: {run_time}')
    r1, U = train_R1(dataset, args)
    finish_time = datetime.now()
    train_time = finish_time - run_time
    print(f"---> Training time: {train_time}")

    os.makedirs(f'./logs/', exist_ok=True)
    with open(f'./logs/time_log.txt', 'a') as f:
        f.write(
            f'{args.model.split("/")[-1]} train r1 time: {train_time} {args.nsamples}sample {args.train_subset_size}% \n')

    if args.save_model: 
        save_path = f"{args.save_path}/{args.save_folder}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_path)
        save_name = f"{save_path}/{args.save_folder}.pt"
        torch.save({'R1': r1}, save_name)
        torch.save({'U': U}, save_name+'.U.pt')

        print(f"---> R1 has been saved in: {save_path}")
