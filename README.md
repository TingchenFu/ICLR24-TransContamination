<p align="center" width="100%">
</p>

<div id="top" align="center">

THE REASONABLENESS BEHIND UNREASONABLE TRANSLATION CAPABILITY OF LARGE LANGUAGE MODEL
-----------------------------

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC%20By%20NC%204.0-yellow)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/WEIGHT_DIFF_LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- <h4> |<a href="https://arxiv.org/abs/2310.09168"> 📑 Paper </a> |
<a href="https://huggingface.co/datasets?sort=trending&search=Explore_Instruct"> 🤗 Data </a> |  
<a href="https://huggingface.co/models?sort=trending&search=Explore-LM"> 🤗 Model </a> |
<a href="https://github.com/fanqiwan/Explore-Instruct"> 🐱 Github Repo </a> |
</h4> -->

<!-- **Authors:** -->

_**Tingchen Fu<sup>‡†</sup>, Lemao Liu<sup>‡</sup>, Deng Cai<sup>‡</sup>, Guoping Huang<sup>‡</sup>, Shuming Shi<sup>‡</sup>, Rui Yan<sup>†</sup>**_


<!-- **Affiliations:** -->


_<sup>†</sup> Gaoling School of Artificial Intelligence, Renmin University of China_
<sup>‡</sup> Tencent AI Lab

</div>


<!-- ## News
- **Oct 16, 2023:** 🔥 We're excited to announce that the Explore-Instruct datasets in brainstorming, rewriting, and math domains are now available on 🤗 [Huggingface Datasets](https://huggingface.co/datasets?sort=trending&search=Explore_Instruct)! Additionally, we've released Explore-LM models that have been initialized with LLaMA-7B and fine-tuned with the Explore-Instruct data in each domain. You can find these models on 🤗 [Huggingface Models](https://huggingface.co/models?sort=trending&search=Explore-LM). Happy exploring and instructing! -->

## Contents

- [THE REASONABLENESS BEHIND UNREASONABLE TRANSLATION CAPABILITY OF LARGE LANGUAGE MODEL](#the-reasonableness-behind-unreasonable-translation-capability-of-large-language-model)
- [Contents](#contents)
- [Overview](#overview)
- [Data Release](#data-release)
- [Post-train/Pre-train](#post-trainpre-train)
- [Evaluation](#evaluation)
- [License](#license)

## Overview

<!-- We propose Explore-Instruct, a novel approach to enhancing domain-specific instruction coverage. We posit that the domain space is inherently structured akin to a tree, reminiscent of cognitive science ontologies. Drawing from the essence of classical search algorithms and incorporating the power of LLMs, Explore-Instruct is conceived to actively traverse the domain space and generate instruction-tuning data, **not** necessitating a predefined tree structure. Specifically, Explore-Instruct employs two strategic operations: lookahead and backtracking exploration: -->

Large language models (LLMs) exhibit non-trivial or even state-or-the-art capacity in neural machine transtion, violating the conventional wisdom that translation ability highly relies on large-scale parallel corpus. To understand the mechanism behind the unreasonable translation ability, we propose that three types of unintentional bilingual data make crucial contribution to the translation ability of LLM. Specifically, three common types of unintentional bilingual data includes:


- **sentence alignment** The co-occurence of a sentence and its translation within close proximity in a document.

- **word alignment** The co-occurrence of one or more words (though not an entire sentence) and their translations within close proximity in a single document.

- **code-switching** The co-occurrence of two languages within close proximity in a document, where the content in the two languages is semantically related rather than bearing a direct translation relationship.




<p align="center">
    <img src="./assets/case_v1.pdf" width="95%"> <br>
</p>

## Data Release

We release the excavated unintentional bilingual data from mC4.en and mC4.zh. The data is available on 🤗 [Huggingface Datasets](https://huggingface.co/datasets?sort=trending&search=Explore_Instruct). Each sample is a structured data file in the JSON format. It consists of a list of dictionaries, with each dictionary containing the following fields: 

<!-- We release the Explore-Instruct data in brainstorming, rewriting, and math domains on 🤗 [Huggingface Datasets](https://huggingface.co/datasets?sort=trending&search=Explore_Instruct). Each domain includes two versions of datasets: the basic and extended version. The base version contains 10k instruction-tuning data and the extended version contains 16k, 32k, and 64k instruction-tuning data for each domain respectively. Each dataset is a structured data file in the JSON format. It consists of a list of dictionaries, with each dictionary containing the following fields: -->

- `file_no`: `str`, the name of file in mC4 from which the case is found. 
- `doc_no`: `str`, the document number in the current file.
- `text`: `str`, the unintentional bilingual text.

The statistics of the

<p align="left">
    <img src="./assets/ratio.pdf" width="50%"> <br>
</p>

|                    |            | en      | zh      |
|--------------------|------------|---------|---------|
| sentence alignment | # Document | 210,931 | 2462    |
|                    | # Sequence | 355,320 | 432     |
|   word alignment   | # Document | 658643  | 1972764 |
|                    | # Sequence | 500,550 | 659,456 |
|     code switch    | # Document | 2021502 | 5086373 |
|                    | # Sequence | 903,810 | 997,376 |



## Post-train/Pre-train

To post-train BLOOM-560m on unintentional bililingual data, you can use the following command:

```
export NCCL_IB_GID_INDEX=3
accelerate launch  \
--machine_rank 0  \
--num_machines 1  \
--num_processes 8  \
--config_file  accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_ubd.py  \
--debug False \
--streaming True   \
--train_file   PATH_TO_UNINTENTIONAL_BILINGUAL_DATA   \
--from_scratch False \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path  bigscience/bloom-560m    \
--per_device_train_batch_size  2  \
--gradient_accumulation_steps  32  \
--learning_rate 3e-4    \
--num_warmup_steps  500   \
--window_size 1024          \
--special_name   posttrain_wordalign     \
--num_train_epochs  1      \
--with_tracking False      \
--max_train_steps  -1    \
--checkpoint_step 4000   \
--seed 0  \
```


To post-train BLOOM-7b1 with PEFT technique on unintentional bililingual data, you can use the following command:

```
accelerate launch  \
--machine_rank 0 \
--num_machines  1 \
--num_processes  8 \
--config_file  accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_ubd_peft.py  \
--debug False \
--streaming True   \
--from_scratch False \
--train_file  file_sent.txt   \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-7b1  \
--per_device_train_batch_size  1  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-4    \
--num_warmup_steps  0   \
--special_name   peft_sentalign     \
--num_train_epochs  1      \
--with_tracking False      \
--lora_target_module query_key_value,dense   \
--max_train_steps -1  \
--checkpoint_step 2000   \
```


## Evaluation

For post-training expeirments, we measure the translation performance with BLEURT and COMET. To generate translation hypothesis, we may use the following command:

```
python3 -u llm_generate.py  
--model_name_or_path   bigscience/bloom-560m    
--ckpt_path  PATH_TO_THE_POST-TRAINED_MODEL.   
--n_example 3  
--source_language en 
--target_language zh   
--dataset WMTnews21
```



For pre-training experiment, since BLEURT and COMET is issensitive to minor improvement when the translation performance is poor, we measure the perplexity using the following command. 

```
python3  -u plm_ppl.py  
--model_name_or_path bigscience/bloom-560m
--ckpt_path PATH_TO_THE_MODEL_TRAINED_FROM_SCRATCH
--n_example  3  
--source_language en 
--target_language zh 
--architecture target-only  
--use_prompt True
```



<!-- ## Limitations

Explore-Instruct is still under development and needs a lot of improvements. We acknowledge that our work focuses on the enhancement of domain-specific instruction coverage and does not address other aspects of instruction-tuning, such as the generation of complex and challenging instructions or the mitigation of toxic and harmful instructions. Future work is needed to explore the potential of our approach in these areas. -->

## License

The work is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. 
<!-- 
## Citation

If you find this work is relevant with your research or applications, please feel free to cite our work!
```
@misc{wan2023explore,
   title={Explore-Instruct: Enhancing Domain-Specific Instruction Coverage through Active Exploration},
   author={Fanqi, Wan and Xinting, Huang and Tao, Yang and Xiaojun, Quan and Wei, Bi and Shuming, Shi},
   year={2023},
   eprint={2310.09168},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
``` -->

<!-- ## Acknowledgments

This repo benefits from [Stanford-Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [Vicuna](https://github.com/lm-sys/FastChat). Thanks for their wonderful works! -->