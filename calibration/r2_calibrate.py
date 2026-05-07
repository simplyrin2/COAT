import sys

from zmq import device
import wandb
from torch.utils.data import DataLoader
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch
from pathlib import Path
import argparse
import os
import transformers
import torch.nn.functional as F
import numpy as np
from scipy.linalg import hadamard
from tqdm import tqdm
from welford_torch import OnlineCovariance
import hadamard_utils

from dataloader import R2Dataset

sys.path.append('../fake_quant')

from rotation_utils import get_orthogonal_matrix
from utils import supported_models, supported_datasets
from model_utils import get_model


os.environ['WANDB_API_KEY'] = "XXXXXXXXXXXXXXXXX"
os.environ['WANDB_MODE'] = "offline"

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


default_path = '../data/'


def train_r2(dataloader,
             layer_id,
             device,
             args):

    print(f"---> start calculating R2 analytically for layer {layer_id}")

    head_num = args.head_num
    kv_head = args.kv_head
    head_dim = args.hidden_size // head_num
    heads_per_kv = head_num // kv_head

    # Initialize Welford objects
    w_list = [OnlineCovariance() for _ in range(kv_head)]

    for batch in tqdm(dataloader, desc=f"Calculating Covariance Layer {layer_id}"):
        # batch: [B, seq_len, hidden_size]
        x = batch.to(device)
        
        # Reshape to separate heads: [B * seq_len, head_num, head_dim]
        x = x.view(-1, head_num, head_dim)
        
        # Group by KV heads: [B * seq_len, kv_head, heads_per_kv, head_dim]
        x = x.view(-1, kv_head, heads_per_kv, head_dim)
        
        # Permute to [kv_head, B * seq_len * heads_per_kv, head_dim]
        x = x.permute(1, 0, 2, 3).reshape(kv_head, -1, head_dim)
        
        for kv_idx in range(kv_head):
            w_list[kv_idx].add_all(x[kv_idx])

    R2_rotate = torch.zeros((kv_head, head_dim, head_dim), device=device, dtype=torch.float32)
    
    H = hadamard_utils.random_hadamard_matrix(head_dim, device=device).to(torch.float32)

    # R2 = U @ H
    for kv_idx in range(kv_head):
        # w.eig_vec contains the eigenvectors (U) of the covariance matrix
        U = w_list[kv_idx].eig_vec.to(device=device, dtype=torch.float32)
        
        R2_rotate[kv_idx] = U @ H

    print(f"---> R2 of layer {layer_id} calculation done ")
    return R2_rotate.cpu()


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, choices=supported_models,
                        help='model name.')
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        choices=supported_datasets,
                        help='Dataset for Evaluation (default: wikitext2)',)
    parser.add_argument('--nsamples', type=int, default=128, help='Number of train sample.')
    parser.add_argument('--calib_sample', type=int, default=128,
                        help='Number of sample.')
    parser.add_argument('--hf_token', type=str, default=None)

    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether save learned r1 (default: True).')
    parser.add_argument('--save_file_name', type=str, default='',
                        help="Saving folder name for trained model (default: '')")
    parser.add_argument('--bsz', type=int, default=128,
                        help='Batch-size for training (default: 128)')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to use wandb for logging (default: False)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed for HuggingFace and PyTorch')

    args = parser.parse_args()

    now = datetime.now()
    run_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    setattr(args, 'run_target_time', run_time)
    args.save_file_name += f'corrmax.{args.bsz}'

    data_path = os.path.join(
        default_path,
        f"train_data/{args.calib_dataset}_{args.calib_sample}samples/{args.model.split('/')[-1]}/r2_train")
    setattr(args, 'data_path', data_path)
    print(f'---> data path: {data_path}')

    save_path = os.path.join(
        default_path,
        f"trained_rotation/{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r2/{args.save_file_name}")
    setattr(args, 'save_path', save_path)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    return args, now


def main():
    args, run_time = parser_gen()
    print(f'---> start time: {run_time}')

    transformers.set_seed(args.seed) 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(args.model, args.hf_token)
    setattr(args, 'hidden_size', model.config.hidden_size)
    setattr(args, 'kv_head', model.config.num_key_value_heads)
    setattr(args, 'head_num', model.config.num_attention_heads)
    setattr(args, 'hidden_layers', model.config.num_hidden_layers)
    del model

    save_dict = {}

    for layer_id in range(args.hidden_layers):
        data_dire = os.path.join(args.data_path,
                                 f'layer_{layer_id}_self_attn_o_proj.pt')
        r2_dataset = R2Dataset(data_dire, args.nsamples, device)

        dataloader = DataLoader(r2_dataset,
                                batch_size=args.bsz,
                                shuffle=True, num_workers=0)

        r2 = train_r2(dataloader=dataloader,
                      layer_id=layer_id,
                      device=device,
                      args=args)
        save_dict[f"model.layers.{layer_id}.self_attn.R2"] = r2.detach()
        print(f"---> layer {layer_id} 's r2 has been trained.")
        if device.type == 'cuda':
            print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    if args.save_model:
        save_name = os.path.join(args.save_path, f"{args.save_file_name}.pt")
        torch.save(save_dict, save_name)

    finish_time = datetime.now()
    train_time = finish_time - run_time
    print(f"---> Training time: {train_time}")
    print(f'---> Save in: {save_name}')

    os.makedirs(f'./logs/', exist_ok=True)
    with open(f'./logs/r2_time_log.txt', 'a') as f:
        f.write(f'{args.model.split("/")[-1]} train r2 time: {train_time} {args.nsamples}sample \n')


if __name__ == "__main__":
    main()