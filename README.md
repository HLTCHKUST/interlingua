# High-Dimensional Interlingual Representations of Large Language Models
<img align="right" src="img/caire.png" width="20%"> <img align="right" src="img/HKUST.jpeg" width="12%">
[Bryan Wilie](https://bryanwilie.github.io/), Samuel Cahyawijaya, Junxian He, and Pascale Fung.<br>

This is the official repository for the [paper: "High-Dimensional Interlingual Representations of Large Language Model"](https://aclanthology.org/2025.sigtyp-1.14.pdf), orally presented and published in the [SIGTYP at ACL 2025](https://sigtyp.github.io/ws2025-sigtyp-schedule.html).

## Overview

Large language models (LLMs) trained on massive multilingual datasets hint at the formation of interlingual constructs--a shared region in the representation space. However, evidence regarding this phenomenon is mixed, leaving it unclear whether these models truly develop unified interlingual representations, or present a partially aligned constructs. We explore 31 diverse languages varying on their resource-levels, typologies, and geographical regions; and find that multilingual LLMs exhibit inconsistent cross-lingual alignments. 

To address this, we propose an interlingual representation framework identifying both the shared interlingual semantic region and fragmented components, existed due to representational limitations. We introduce <b>Interlingual Local Overlap (ILO)</b> score to quantify interlingual alignment by comparing the local neighborhood structures of high-dimensional representations.

We utilize the ILO score to investigate the impact of single-language fine-tuning on the interlingual alignment in multilingual LLMs. Our results indicate that training exclusively on a single language disrupts the alignment in early layers, while doing <b>selective freezing</b> on these layers preserves alignment of the interlingual representations, leading to improved cross-lingual generalization. 

These results validate our framework and metric for evaluating interlingual representation, and further underscore that interlingual alignment is crucial for scalable multilingual learning.

## Usage

To derive the ILO scores of a language model, run `bash run_get_ilo.sh`.<br>
To selectively freeze model's parameters, use `from src.param_freeze import selective_grad_freeze` or see `Selective freezing.ipynb` for a simple demonstration.<br>
On both deriving the ILO scores and selectively freezing model's parameters, we have set the default to follow the best settings as reported in the paper.

## Citation

If you find the research paper or the code useful, please cite:
```
@inproceedings{wilie2025interlingua,
    title={High-dimensional interlingual representations of large language models},
    author={Wilie, Bryan and Cahyawijaya, Samuel and He, Junxian and Fung, Pascale},
    booktitle = "Proceedings of the 7th Workshop on Research in Computational Linguistic Typology and Multilingual NLP",
    year = {2025},
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    paper = {https://aclanthology.org/2025.sigtyp-1.14/},
    }
```