# LLM-Powered Domain-Robust Anchors via Logits Distillation for Unsupervised Domain Adaptation



------

## Highlights

![Architecture](https://github.com/el2k/DRA/blob/main/Architecture.png)
> Abstract: The advent of recent large vision-language models(VLMs) has significantly advanced unsupervised domain adaptation (UDA) for generating cross-domain representations, yet it remains fundamentally challenged by three critical bottlenecks: insufficient semantic robustness of a simple prompt template, poor generalization of parameter-efficient adaptation under severe domain shifts, and unstable target-domain adaptation caused
by noisy pseudo-labels. To address these challenges, we propose LLM-Powered Domain-Robust Anchors via Logits Distillation for
UDA (termed DRA). 
(1) We introduce a Domain-Agnostic Prompt Generator (DAPG) that leverages a large language model (LLM)
to construct semantically rich, domain-invariant class anchors.By providing robust and transferable semantic anchors for cross-
modal alignment, DAPG effectively alleviates the semantic bias of vanilla single-class-name prompts and enhances the semantic
robustness under domain shifts. 
(2) To enable efficient and generalizable adaptation of VLMs, we propose Geometry-Aware
LoRA (GAL), which guides the low-rank adaptation optimization toward flatter minima in the loss landscape. This design significantly improves the cross-domain generalization ability of parameter-efficient adaptation, striking a favorable balance between parameter efficiency and representational transferability under large domain gaps. 
(3) We develop a stable student adaptation strategy in an end-to-end teacher-student distillation paradigm, which effectively mitigates the negative impact of noisy pseudo-labels and stabilizes the self-training process, while preserving the intrinsic visual discriminability of the pre-trained VLM. We conduct extensive cross-domain experiments on four widely-used UDA benchmarks, including Office-Home, Office-31,
VisDA-2017, and DomainNet. Experimental results verify that DRA consistently achieves state-of-the-art performance across diverse cross-domain scenarios compared with existing CNN-, Transformer-, and VLMs-based solutions. Notably, DRA demonstrates strong robustness on highly challenging large-domain-gap settings while requiring ultra-low learnable parameters and inference overhead, validating the effectiveness and scalability
of the proposed lightweight adaptation paradigm. The code is available at https://github.com/el2k/DRA.

## Main Contributions

- **New perspective:** To the best of our knowledge, we are the first to come up with LLM-powered prompts toconstruct domain-robust class anchors for UDA. Each anchor can comprehensively describe the class information it corresponds to, without introducing particular bias toward any specific domain, thereby yielding robust cross-domain visual representations.
- **Novel Paradigm:** Deviating from the latest two-stage or teacher-trainable approaches, we introduce an end-to-end
teacher-student distillation paradigm (DRA) that transfers knowledge from an entirely frozen teacher to a student
model updated solely via geometry-aware LoRA finetuning, i.e., a lightweight adaptation strategy designed to explicitly navigate optimization toward flatter solution landscapes, yielding improved stability and domain transferability.
- **High Performance:** We conduct extensive experiments on four widely-used UDA benchmarks, including Office-Home, Office-31, VisDA-2017 and Do-
mainNet. Experimental results demonstrate that our proposed DRA consistently outperforms existing state-of-the-art (SOTA) UDA methods under large domain shifts.Moreover, DRA achieves superior robustness with significantly lower computational complexity, validating the effectiveness and scalability of the proposed lightweight cross-domain paradigm.

------




## Installation



For installation and other package requirements, please follow the instructions as follows. This codebase is tested on Ubuntu 18.04 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

- Setup conda environment.

```
# Create a conda environment
conda create -y -n dra python=3.8

# Activate the environment
conda activate dra

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```



- Install dassl library.

```
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch

# Install dependencies
pip install -r requirements.txt

```



- Clone DRA code repository and install requirements.

```
# Clone DRA code base
git clone https://github.com/el2k/DRA.git
cd DRA

# Install requirements
pip install -r requirements.txt
```



## Data preparation



Please follow the instructions as follows to prepare all datasets. Datasets list:

- [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?pli=1&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)
- [VisDA-2017](http://ai.bu.edu/visda-2017/#download)
- [DomainNet](http://ai.bu.edu/M3SDA/)
------

## Training and Evaluation

Please follow the instructions for training, evaluating and reproducing the results. Firstly, you need to **modify the directory of data by yourself**.

### Training



```
# Example: trains on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/DRA/main_DRA.sh officehome b32_ep10_officehome DRA ViT-B/16 a-c  0
```



### Evaluation



```
# evaluates on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/DRA/eval_DRA.sh officehome b32_ep10_officehome DRA ViT-B/16 a-c 0
```



The details are at each method folder in [scripts folder]([DRA/scripts at main ¡¤ el2k/DRA (github.com)](https://github.com/el2k/DRA/tree/main/scripts)).



## Acknowledgements



Our style of reademe refers to [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment). And our code is based on [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp), [DAPL](https://github.com/LeapLabTHU/DAPrompt/tree/main) , [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)  , [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) , [EKDA]([https://github.com/1d1x1w/CDU], [PMCC](https://github.com/246dxw/PMCC) and  etc. repository. We thank the authors for releasing their code. If you use their model and code, please consider citing these works as well. Supported methods are as follows:

| Method       | Paper                                          | Code                                                         |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| CoOp         | [IJCV 2022](https://arxiv.org/abs/2109.01134)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| CoCoOp       | [CVPR 2022](https://arxiv.org/abs/2203.05557)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| VPT          | [ECCV 2022](https://arxiv.org/abs/2203.17274)  | [link](https://github.com/KMnP/vpt)                          |
| IVLP & MaPLe | [CVPR 2023](https://arxiv.org/abs/2210.03117)  | [link](https://github.com/muzairkhattak/multimodal-prompt-learning) |
| DAPL         | [TNNLS 2023](https://arxiv.org/abs/2202.06687) | [link](https://github.com/LeapLabTHU/DAPrompt)               |
| PDA          | [AAAI 2024](https://arxiv.org/abs/2312.09553)  | [link](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) |
| EKDA          | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/39871)  | [link](https://anonymous.4open.science/r/EKDA) |
| PMCC         | [PR 2026](https://www.sciencedirect.com/science/article/abs/pii/S0031320325007551?via%3Dihub)  | [link](https://github.com/246dxw/PMCC)                       |
