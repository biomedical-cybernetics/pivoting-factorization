# Pivoting Factorization

Reference implementation (example) of the model proposed in the paper:

**[Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models](https://icml.cc/virtual/2025/poster/46433)** (Published at ICML 2025)  
Jialin Zhao<sup>1,2</sup>, Yingtao Zhang<sup>1,2</sup>, Carlo Vittorio Cannistraci<sup>1,2,3</sup>  
<sup>1</sup> Center for Complex Network Intelligence (CCNI), Tsinghua Laboratory of Brain and Intelligence (THBI), Department of Psychological and Cognitive Sciences, <sup>2</sup> Department of Computer Science, <sup>3</sup> Department of Biomedical Engineering, Tsinghua University, China  
Correspondence to: Jialin Zhao <jialin.zhao97@gmail.com>, Carlo Vittorio Cannistraci <kalokagathos.agon@gmail.com>

## Installation

Ensure you have the necessary dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Apply M Reconstruction

```
python PIFA.py --model ../model/llama-2-7b-hf --mode m --overall_ratio 0.5 \
--attn_ratio 0.5 --pruning_nsamples 256 --dataset wikitext2 --seed 3 \
--model_seq_len 2048 --save_path ./results --old_output_factor 0.25 \
--reconstruct_step 2 --reconstruct_nsamples 128 --use_pifa
```

### 2.PIFA Compression

```
python PIFA.py --mode pifa --model_path <path_to_model>
```

### 3.Evaluate Perplexity (PPL)

```
python PIFA.py --mode ppl --model_path <path_to_model>
```

## Contact

Please contact jialin.zhao97@gmail.com in case you have any questions.

## Acknowledgment

This repository is built upon the [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) repository.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{
zhao2025pivoting,
title={Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models},
author={Zhao, Jialin and Zhang, Yingtao and Cannistraci, Carlo Vittorio},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=5OLRHkzTYk}
}
```