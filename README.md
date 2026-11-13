# COrrelation-Aware Orthogonal Transform (COAT) for LLM Quantization

This repository is the official implementation of the COAT paper available on [OpenReview](https://openreview.net/forum?id=6JGbm48VoG&referrer=%5Bthe%20profile%20of%20Indranil%20Patra%5D%28%2Fprofile%3Fid%3D~Indranil_Patra1%29).

## Citation

If you use this repository, please cite:

- [COAT (OpenReview)](https://openreview.net/forum?id=6JGbm48VoG&referrer=%5Bthe%20profile%20of%20Indranil%20Patra%5D%28%2Fprofile%3Fid%3D~Indranil_Patra1%29)
- [DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](https://arxiv.org/abs/2511.04063)
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
- [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)

## 1. Requirements:
- python 3.10, pytorch >= 2.0
- install pytorch with cuda from https://pytorch.org/get-started/locally/, it is prerequisite for fast-hadamard-transform package.
- pip install -r requirement.txt
  
  install fast-hadamard-transform
  
  ```python
  cd third-part
  git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
  cd fast-hadamard-transform
  pip install .
  ```

  install lm-eval
  
  ```python
  git clone https://github.com/EleutherAI/lm-evaluation-harness.git
  cd lm-evaluation-harness
  pip install -e .
  ```

## Guidelines for running the code

- The *./quant* folder contains the code for fusing the calibrated rotation matrix and performing the quantization test. 
- The *./calibration* folder contains the code for obtaining the calibration set and the calibration rotation matrix. 


# Covariance calibration for $R_1$ and $R_2$

Go to *./calibration* folder

Currently, we only support **LLaMa** models. You can simply run the `get_r1_r2_train_data.py` to collect the calibration data. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--calib_dataset`: dataset used to obtain the calibration set
- `--r_path`: where to save the r1&2 calibration data
- `--r_list`: which rotation matrix calibration data to obtain
- `--nsamples`: number of samples
- `--seqlen`: sequence length
  
For example, to get the calibration data for Llama 2-7b, you can run the following command:

```bash
python get_r1_r2_train_data.py --model meta-llama/Llama-2-7b-hf --calib_dataset wikitext2 --nsamples 128 --seqlen 2048
```

## Calibration of $R_1$
Currently, we only support **LLaMa** models. You can simply run the `r1_calibrate.py` to calibrate R1. The most important arguments are:


- `--model`: the model name (or path to the weights)
- `--ep`: number of epochs for training
- `--bsz`: batch-size for training
- `--lr`: learning rate for training
  
For example, to calibrate R1 for Llama 2-7b, you can run the following command:

```bash
python r1_calibrate.py --model meta-llama/Llama-2-7b-hf --bsz 64
```

## Calibration of $R_2$
You can simply run the `r2_calibrate.py` to calibrate R2. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--ep`: number of epochs for training
- `--bsz`: batch-size for training
- `--lr`: learning rate for training

For example, to calibrate R1 for Llama 2-7b, you can run the following command:

```bash
python r2_calibrate_qr.py --model meta-llama/Llama-2-7b-hf --bsz 64
```

# Fake Quantization in CorrQuant

Under *./quant* directory, we provide the torch script for the experiments aka evaluation in CorrQuant. 


## Language Generation and Zero-Shot Evaluations

Currently, we only support **LLaMa** models. You can simply run the `main.py` to reproduce the results in the paper. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--bsz`: the batch size for PPL evaluation
- `--fuse_norm`: whether we want to fuse the layer norm
- `--use_r1`: whether we want to use R1 for rotate attention, up-projection and gate-projection inputs.
- `--r1_path`: we are going to use the calibrated R1 path
- `--use_r2`: whether we want to use R2 for rotate out-projection inputs.
- `--r2_path`: we are going to use the calibrated R2 path
- `--use_r3`: whether we want to use R1
- `--use_r4`: whether we want to use R1
- `--ppl_eval`: whether we want to run PPL test
- `--ppl_eval_dataset`: the tasks for PPL
- `--lm_eval`: whether we want to run LM-Eval for Zero-Shot tasks
- `--tasks`: the tasks for LM-Eval
- `--cal_dataset`: the calibration dataset for GPTQ quantization
- `--a_bits`: the number of bits for activation quantization
- `--w_bits`: the number of bits for weight quantization
- `--v_bits`: the number of bits for value quantization
- `--k_bits`: the number of bits for key quantization
- `--w_clip`: Whether we want to clip the weights
- `--a_clip_ratio`: The ratio of clipping for activation
- `--k_clip_ratio`: The ratio of clipping for key
- `--v_clip_ratio`: The ratio of clipping for value
- `--w_asym`: Whether we want to use asymmetric quantization for weights
- `--a_asym`: Whether we want to use asymmetric quantization for activation
- `--v_asym`: Whether we want to use asymmetric quantization for value
- `--k_asym`: Whether we want to use asymmetric quantization for key
- `--a_groupsize`: The group size for activation quantization
- `--w_groupsize`: The group size for weight quantization
- `--v_groupsize`: The group size for value quantization
- `--k_groupsize`: The group size for key quantization
  
For example, to run a model with quantizing all weights and activations, you can run the following command:

```bash
bash eval.sh <GPU_ID> <MODEL> <W_BITS> <A_BITS> <KV_BITS> <R2_PATH> <R1_PATH>
```
