import os.path as osp
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import time
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms as T
from PIL import Image

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param, mkdir_if_missing, read_image
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import INTERPOLATION_MODES

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.baseda import *
from utils.MK_MMD import MK_MMD
from utils.clip_part import *
from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    lora_state_dict,
    save_lora,
    load_lora,
    apply_lora_rn,
    apply_tlora_rank_mask,
)
from loralib.layers import LoRALayer
from itertools import chain
_tokenizer = _Tokenizer()


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find("BatchNorm") != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model, clip_model_teacher):
        super().__init__(cfg, classnames, clip_model)

        self.text_encoder = Simple_TextEncoder(clip_model)
        self.cfg = cfg
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)
             
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.class_texts = [c.replace("_", " ") for c in classnames]
        self.use_gpt_prompts = bool(getattr(cfg.TRAINER.DRA, "USE_GPT_PROMPTS", False))
        self.gpt_prompt_json = str(getattr(cfg.TRAINER.DRA, "GPT_PROMPT_JSON", ""))
        self.class_prompt_slices = []

        if self.use_gpt_prompts:
            prompt_json_path = self._resolve_prompt_json_path(self.gpt_prompt_json)
            prompt_dict = {}
            if prompt_json_path and osp.exists(prompt_json_path):
                try:
                    with open(prompt_json_path, "r", encoding="utf-8") as f:
                        prompt_dict = json.load(f)
                    if not isinstance(prompt_dict, dict):
                        print(f"[GPT Prompts] invalid json root type, fallback to default class names: {prompt_json_path}")
                        prompt_dict = {}
                except Exception as e:
                    print(f"[GPT Prompts] failed to load {prompt_json_path}, fallback to default class names: {e}")
                    prompt_dict = {}
                    exit()
            else:
                print(f"[GPT Prompts] file not found: {self.gpt_prompt_json}, fallback to default class names")

            flat_prompts = []
            start = 0
            for class_name in self.class_texts:
                prompts = prompt_dict.get(class_name, None)
                if not isinstance(prompts, list):
                    prompts = [class_name]
                clean_prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
                if len(clean_prompts) == 0:
                    clean_prompts = [class_name]

                flat_prompts.extend(clean_prompts)
                end = start + len(clean_prompts)
                self.class_prompt_slices.append((start, end))
                start = end
            self.tokenized_prompts = clip.tokenize(flat_prompts)
            self.tokenized_prompts_u = self.tokenized_prompts
            print(f"[GPT Prompts] enabled, loaded {len(flat_prompts)} prompts for {len(self.class_texts)} classes")
        else:
            self.tokenized_prompts = clip.tokenize(self.class_texts)
            self.tokenized_prompts_u = self.tokenized_prompts

        self.clip_model_teacher = clip_model_teacher.to(self.logit_scale.device)
        self.confi = cfg.CONFI
        self.aux_cls_weight = float(getattr(cfg.TRAINER.DRA, "AUX_CLS_WEIGHT", 1.0))
        self.dim = self.image_encoder.dim
        self.n_cls = len(classnames)
        self.epoch = cfg.OPTIM.MAX_EPOCH
        self.print_beats = True

        self.ema_m = float(getattr(cfg.TRAINER.DRA, "FREEMATCH_EMA_M", 0.9))
        self.confi_start = float(getattr(cfg.TRAINER.DRA, "CONFI_START", max(self.confi, 0.8)))
        self.confi_end = float(getattr(cfg.TRAINER.DRA, "CONFI_END", self.confi))
        self.beta_min = float(getattr(cfg.TRAINER.DRA, "BETA_MIN", 0.0))
        self.beta_max = float(getattr(cfg.TRAINER.DRA, "BETA_MAX", 0.95))
        self.beta_warmup_epochs = int(getattr(cfg.TRAINER.DRA, "BETA_WARMUP_EPOCHS", 2))
        self.min_keep_rate = float(getattr(cfg.TRAINER.DRA, "MIN_KEEP_RATE", 0.2))
        self.dynamic_thresh_floor = float(getattr(cfg.TRAINER.DRA, "DYNAMIC_THRESH_FLOOR", 0.0))

        min_prob = 1.0 / self.n_cls
        self.confi_start = max(min_prob, min(self.confi_start, 0.999))
        self.confi_end = max(min_prob, min(self.confi_end, self.confi_start))
        self.beta_min = min(max(self.beta_min, 0.0), 1.0)
        self.beta_max = min(max(self.beta_max, self.beta_min), 1.0)
        self.min_keep_rate = min(max(self.min_keep_rate, 0.0), 1.0)
        self.dynamic_thresh_floor = max(min_prob, min(self.dynamic_thresh_floor, self.confi_start))
        self._last_beta = None

        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.dim),
            nn.LayerNorm(self.dim, eps=1e-6),
            nn.Linear(self.dim, self.n_cls, bias=False),
        )
        self.classifier_layer.apply(weights_init_classifier)

        self.register_buffer("class_probs_ema", torch.full((self.n_cls,), min_prob))
        self.register_buffer("global_prob_ema", torch.tensor([self.confi_start]))

    def _resolve_prompt_json_path(self, prompt_json_path):
        if not prompt_json_path:
            return ""
        if osp.isabs(prompt_json_path):
            return prompt_json_path

        project_root = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
        candidate_paths = [
            osp.abspath(prompt_json_path),
            osp.join(project_root, prompt_json_path),
        ]
        for path in candidate_paths:
            if osp.exists(path):
                return path
        return candidate_paths[-1]

    def _encode_text_features(self):
        text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if not self.use_gpt_prompts:
            return text_features

        class_features = []
        for start, end in self.class_prompt_slices:
            cls_feat = text_features[start:end].mean(dim=0, keepdim=True)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            class_features.append(cls_feat)
        return torch.cat(class_features, dim=0)

    def _aggregate_logits_per_class(self, logits):
        if not self.use_gpt_prompts:
            return logits

        class_logits = []
        for start, end in self.class_prompt_slices:
            class_logits.append(logits[:, start:end].mean(dim=1, keepdim=True))
        return torch.cat(class_logits, dim=1)

    def _epoch_progress(self, epoch):
        if epoch is None:
            return 0.0
        max_epoch = max(int(self.epoch) - 1, 1)
        return float(min(max(int(epoch), 0), max_epoch) / max_epoch)

    def forward(self, image, label=None, epoch=None, source=False, train=False):
        self._last_beta = None
        text_features = self._encode_text_features()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if self.cfg.MODEL.BACKBONE.NAME.split('-')[0] != 'ViT':
            compute_dtype = self.logit_scale.dtype
            text_features = text_features.to(compute_dtype)
            image_features = image_features.to(compute_dtype)
            logit_scale = self.logit_scale.to(compute_dtype).exp()
            logits = logit_scale * image_features @ text_features.t()
        else:
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

        logits_a = self.classifier_layer(image_features.float())
        logits_mix = logits + self.aux_cls_weight * logits_a.to(logits.dtype)

        if train:
            if source:
                loss = F.cross_entropy(logits, label)
                loss += self.aux_cls_weight * F.cross_entropy(logits_a, label)
                return loss, logits_mix, image_features
            else:
                logits_u, _ = self.clip_model_teacher(
                    image.type(self.dtype),
                    self.tokenized_prompts_u.to(self.logit_scale.device),
                )
                logits_u = self._aggregate_logits_per_class(logits_u)
                
                with torch.no_grad():
                    prob_stu_pure = torch.softmax(logits, dim=-1)
                    prob_tea_pure = torch.softmax(logits_u, dim=-1) 
                    
                    student_conf = prob_stu_pure.max(dim=-1)[0].mean()
                    
                    # ==================== 方案优化：安全稳健的 Hybrid Beta ====================
                    # 1. 计算基于 Epoch 的余弦进度 [0 -> 1] (比纯线性更平滑，前期稳，后期快)
                    import math
                    if epoch is not None and self.epoch > 0:
                        progress = epoch / self.epoch
                        time_beta = 0.5 * (1.0 - math.cos(math.pi * progress))
                    else:
                        time_beta = 0.0
                        
                    # 2. 结合学生置信度与时间进度
                    # 学生不能仅凭自信就夺权，它的权重被 time_beta 严格限制，防止在 VisDA 上盲目自信
                    beta = time_beta * (student_conf.item() ** 2)
                    
                    if self.beta_warmup_epochs > 0 and epoch is not None:
                        warmup_progress = min(max(int(epoch), 0) / max(self.beta_warmup_epochs, 1), 1.0)
                        beta_cap = getattr(self, 'beta_min', 0.0) + (getattr(self, 'beta_max', 0.95) - getattr(self, 'beta_min', 0.0)) * warmup_progress
                        beta = min(beta, beta_cap)
                    
                    beta = min(max(beta, getattr(self, 'beta_min', 0.0)), getattr(self, 'beta_max', 0.95))
                    self._last_beta = float(beta)
                    # =========================================================================

                probs_fusion = beta * prob_stu_pure + (1 - beta) * prob_tea_pure
                max_probs, label_p = torch.max(probs_fusion, dim=-1)

                # ==================== FreeMatch 自适应阈值核心逻辑 ====================
                with torch.no_grad():
                    mean_max_prob = max_probs.mean()
                    current_batch_class_probs = probs_fusion.mean(dim=0)
                    
                    # 🚨 【致命 Bug 修复】：你之前漏掉了冷启动判断，导致前期阈值为0，全是噪声！
                    if self.global_prob_ema.sum() == 0:
                        self.global_prob_ema.copy_(mean_max_prob)
                        self.class_probs_ema.copy_(current_batch_class_probs)
                    else:
                        self.global_prob_ema.mul_(self.ema_m).add_(mean_max_prob, alpha=1 - self.ema_m)
                        self.class_probs_ema.mul_(self.ema_m).add_(current_batch_class_probs, alpha=1 - self.ema_m)

                    max_class_prob = self.class_probs_ema.max()
                    dynamic_thresholds = self.global_prob_ema * (self.class_probs_ema / (max_class_prob + 1e-8))

                    # 动态上限
                    progress_conf = epoch / self.epoch if epoch is not None else 0
                    conf_cap = getattr(self, 'confi_start', 0.8) + (getattr(self, 'confi_end', 0.95) - getattr(self, 'confi_start', 0.8)) * progress_conf
                    
                    # 安全下限：必须 >= 均匀分布概率，否则瞎猜也会被算作伪标签
                    min_prob_floor = max(1.0 / self.n_cls, getattr(self, 'dynamic_thresh_floor', 0.05))
                    
                    dynamic_thresholds = torch.clamp(
                        dynamic_thresholds,
                        min=min_prob_floor,
                        max=conf_cap,
                    )

                    sample_thresholds = dynamic_thresholds[label_p]

                mask = max_probs.ge(sample_thresholds).float()

                # 兜底机制
                min_keep = getattr(self, 'min_keep_rate', 0.0)
                if min_keep > 0:
                    keep_rate = float(mask.mean().item())
                    if keep_rate < min_keep:
                        k = max(1, int(min_keep * max_probs.numel()))
                        topk_idx = torch.topk(max_probs, k=k, largest=True).indices
                        fallback_mask = torch.zeros_like(mask)
                        fallback_mask[topk_idx] = 1.0
                        mask = torch.maximum(mask, fallback_mask)

                denom = mask.sum().clamp(min=1.0)
                loss = (F.cross_entropy(logits, label_p, reduction="none") * mask).sum() / denom
                loss += self.aux_cls_weight * (F.cross_entropy(logits_a, label_p, reduction="none") * mask).sum() / denom
                
                # ==================== 🌟 终极防坍塌：InfoMax 类别均匀正则化 ====================
                # 计算当前 batch 模型在目标域预测概率的均值
                mean_prob_target = prob_stu_pure.mean(dim=0)
                # 最小化负熵 -> 最大化熵。强迫模型在一个 Batch 里预测出各种不同的类别，防止全变 knife
                loss_div = torch.sum(mean_prob_target * torch.log(mean_prob_target + 1e-6))
                
                # Ablation: disable diversity regularization term.
                # 如需恢复，改回从 cfg 读取 DIV_WEIGHT 并加到总 loss。
                div_weight = float(getattr(self.cfg.TRAINER.DRA, "DIV_WEIGHT", 1.0))
                loss += div_weight * loss_div
                # =========================================================================

                return loss, logits_mix, image_features
        else:
            return logits_mix
class CLIPGradCAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        # 强制走推理模式，只获取 logits
        return self.model(image, train=False)
@TRAINER_REGISTRY.register()
class DRA(BaseDA):
    '''Multi-modal Prompt Learning (DRA)
    
    Adapt from DRA: Multi-modal Prompt Learning
    https://arxiv.org/abs/2210.03117
    '''
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        backbone_name = cfg.MODEL.BACKBONE.NAME
        is_vit_backbone = backbone_name.split('-')[0] == 'ViT'

        print(f"Loading CLIP (backbone: {backbone_name})")
        clip_model = load_clip_to_cpu(cfg)
        #clip_model_teacher = load_clip_to_cpu(cfg, teacher=True)
        #如果是RN50架构，使用更好的教师模型（RN101）来提供更强的监督信号，帮助学生模型更快地学习到有用的特征。
        if backbone_name == "RN50":
            print("Using stronger teacher model (ViT-L14) for RN50 student...")
            clip_model_teacher = load_clip_to_cpu(cfg, teacher=True)
        else:   
            print("Using same architecture for teacher(ViT-L14) and student...")
            clip_model_teacher = load_clip_to_cpu(cfg, teacher=True)

        if cfg.TRAINER.DRA.PREC == "fp32" or cfg.TRAINER.DRA.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16
            exit()

          # len(list_lora_layers): 12(文本) + 12（视觉）

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, clip_model_teacher)
        if is_vit_backbone:
            self.list_lora_layers = apply_lora(cfg, self.model)
        else:
            # +++ 调试代码 +++
            print("--- Inspecting Image Encoder Architecture ---")
            print(self.model.image_encoder)
            print("-------------------------------------------")
            # +++ 调试结束 +++
            self.list_lora_layers = apply_lora_rn(cfg, self.model)
        
        print("Turning off gradients in both the image and the text encoder...")
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)
            # LoRA-only mode: only LoRA adapter params are trainable.
            if "lora" in _:
                param.requires_grad_(True)
            if "classifier_layer" in _:
                param.requires_grad_(True)
            
        Total_Memory = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Model Total Memory : " + str(Total_Memory) + "MB")

        self.model.to(self.device)

        # T-LoRA schedule config: dynamic active rank over the full training process.
        self.tlora_enabled = bool(getattr(cfg.TRAINER.DRA, "USE_TLORA", True))
        self.tlora_max_rank = int(getattr(cfg.TRAINER.DRA, "RANK_RAMP", cfg.TRAINER.DRA.R))
        self.tlora_min_rank = int(getattr(cfg.TRAINER.DRA, "MIN_RANK", 1))
        self.tlora_alpha = float(getattr(cfg.TRAINER.DRA, "ALPHA_RANK_SCALE", 1.0))
        # Deterministic eval rank avoids stale mask state leaking from training batches.
        eval_rank_cfg = int(getattr(cfg.TRAINER.DRA, "EVAL_RANK", self.tlora_max_rank))
        self.tlora_eval_rank = max(self.tlora_min_rank, min(eval_rank_cfg, self.tlora_max_rank))

        # Warmup epochs to avoid early collapse from noisy pseudo-labels / alignment losses.
        # Office31 sets cfg.WARM_UP=1 in train.py; OfficeHome typically uses 0.
        self.unsup_warmup_epochs = int(getattr(cfg.TRAINER.DRA, "UNSUP_WARMUP_EPOCHS", 0))
        self.warmup_epochs = max(getattr(cfg, "WARM_UP", 0), self.unsup_warmup_epochs)
        #lora_params_test = [p for n, p in self.model.named_parameters() if "lora" in n]
        #self.optimizer_test = LoRARite(lora_params_test, lr=0.0005, betas=(0.9, 0.999))
        # model lora finetune
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        #self.sched = build_lr_scheduler(self.optimizer_test, cfg.OPTIM)
        self.register_model("LoRA", self.model, self.optim, self.sched)
        #self.register_model("LoRA", self.model, self.optimizer_test, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.DRA.PREC == "amp" else None
        self.t_sne_path = osp.join(self.output_dir, "tsne")
        # mkdir_if_missing(self.t_sne_path)
        # Epoch timing bookkeeping
        self.current_epoch = 0
        self._epoch_start_time = None
        # Flat-LoRA controls (disabled by default)
        self.use_flat_lora = bool(getattr(cfg.TRAINER.DRA, "USE_FLAT_LORA", False))
        self.flat_lora_rho = float(getattr(cfg.TRAINER.DRA, "FLAT_LORA_RHO", 0.05))
        self.flat_lora_interval = max(1, int(getattr(cfg.TRAINER.DRA, "FLAT_LORA_INTERVAL", 1)))
        self._flat_lora_noise_cache = []
        if (not is_vit_backbone) and self.use_flat_lora:
            print(f"[FlatLoRA] enabled for backbone={backbone_name}")
            self.flat_lora_rho = 0.2
        if self.use_flat_lora and self.flat_lora_rho <= 0:
            print("[FlatLoRA] FLAT_LORA_RHO <= 0, disabling flat-lora perturbation")
            self.use_flat_lora = False

    def _flat_lora_factor(self):
        total_steps = max(int(self.cfg.OPTIM.MAX_EPOCH) * int(self.num_batches), 1)
        global_step = int(getattr(self, "epoch", 0)) * int(self.num_batches) + int(getattr(self, "batch_idx", 0))
        global_step = min(max(global_step, 0), total_steps)
        # cosine scheduler, increase from 0 to 1
        return 0.5 * (1 - np.cos(global_step / total_steps * np.pi)), global_step

    def _apply_flat_lora_noise(self):
        if not self.use_flat_lora:
            return False

        factor, global_step = self._flat_lora_factor()
        if global_step % self.flat_lora_interval != 0:
            return False

        self._flat_lora_noise_cache = []
        for module in self.model.modules():
            if not isinstance(module, LoRALayer):
                continue
            if not hasattr(module, "params_with_lora") or module.r <= 0:
                continue

            for param_name in module.params_with_lora.keys():
                if not hasattr(module, param_name):
                    continue

                base_param = getattr(module, param_name)
                if not isinstance(base_param, torch.Tensor) or base_param.ndim < 2:
                    continue

                with torch.no_grad():
                    merged_delta = module.merge_BA(param_name)
                    effective = base_param.data + module.scaling * merged_delta
                    effective_2d = effective.view(effective.shape[0], -1)
                    row_norm = torch.norm(effective_2d, dim=1, keepdim=True)
                    std_row = factor * (self.flat_lora_rho + 1e-16) / np.sqrt(effective_2d.shape[1]) * row_norm
                    view_shape = [effective.shape[0]] + [1] * (effective.ndim - 1)
                    std = std_row.view(*view_shape).expand_as(base_param.data)
                    noise = torch.normal(mean=torch.zeros_like(base_param.data), std=std)
                    base_param.data.add_(noise)
                    self._flat_lora_noise_cache.append((base_param, noise))

        return len(self._flat_lora_noise_cache) > 0

    def _remove_flat_lora_noise(self):
        if not self._flat_lora_noise_cache:
            return

        with torch.no_grad():
            for base_param, noise in self._flat_lora_noise_cache:
                base_param.data.sub_(noise)
        self._flat_lora_noise_cache = []


    def forward_backward(self, batch_x, batch_u):
        image_x, label, image_u,label_u = self.parse_batch_train(batch_x, batch_u)
        flat_noise_applied = False

        # Apply T-LoRA dynamic rank mask once per batch.
        if getattr(self, "tlora_enabled", False):
            current_step = int(getattr(self, "epoch", 0)) * int(self.num_batches) + int(getattr(self, "batch_idx", 0))
            total_steps = int(self.cfg.OPTIM.MAX_EPOCH) * int(self.num_batches)
            self.current_tlora_rank = apply_tlora_rank_mask(
                self.list_lora_layers,
                current_step=current_step,
                total_steps=total_steps,
                max_rank=self.tlora_max_rank,
                min_rank=self.tlora_min_rank,
                alpha=self.tlora_alpha,
            )
        else:
            self.current_tlora_rank = None

        # 确保标签与对应 logits 在同一设备上（防止 CPU/GPU 混用）
        label = label.to(image_x.device)
        label_u = label_u.to(image_u.device)

        # Epoch timing: mark start time on first batch of epoch
        if getattr(self, 'batch_idx', 0) == 0:
            self._epoch_start_time = time.time()
            # Reset beta accumulators for this epoch
            self.beta_sum = 0.0
            self.beta_count = 0

        if self.use_flat_lora:
            flat_noise_applied = self._apply_flat_lora_noise()

        loss_x, logits_x, source_features = self.model(image_x, label, epoch=self.epoch, source=True, train=True)

        if self.epoch < self.warmup_epochs:
            # During warmup, train only on labeled source to preserve zeroshot performance.
            with torch.no_grad():
                logits_u = self.model(image_u, train=False)
            loss_u = torch.zeros((), device=loss_x.device, dtype=loss_x.dtype)
            loss_mmd = torch.zeros((), device=loss_x.device, dtype=loss_x.dtype)
            loss = loss_x 
        else:
            loss_u, logits_u, target_features = self.model(image_u, epoch=self.epoch, source=False, train=True)
            # 如果 model 在 forward 中保存了最近一批次计算的 beta，则累加用于计算 epoch 平均值
            try:
                last_beta = getattr(self.model, '_last_beta', None)
                if last_beta is not None:
                    self.beta_sum += float(last_beta)
                    self.beta_count += 1
            except Exception:
                pass
            # Ablation: disable MMD term while keeping the metric key in logs.
            loss_mmd = MK_MMD(source_features, target_features)
            # loss_mmd = torch.zeros((), device=loss_x.device, dtype=loss_x.dtype)
            loss = loss_x + loss_mmd + loss_u
            # loss = loss_x + loss_u

        if flat_noise_applied:
            self.model_zero_grad()
            self.model_backward(loss)
            self._remove_flat_lora_noise()
            self.model_update_with_gradient_monitoring(
                max_norm=20.0,
                monitor_interval=10,
                clip_on_explosion=True,
            )
        else:
            self.model_backward_and_update_with_gradient_monitoring(
            loss,
            max_norm=20.0,
            monitor_interval=10,
            clip_on_explosion=True,
        )
        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "loss_mmd": loss_mmd.item(),
            "acc_source": compute_accuracy(logits_x, label)[0].item(),
            "acc_target": compute_accuracy(logits_u, label_u)[0].item(),
        }
        if self.current_tlora_rank is not None:
            loss_summary["tlora_rank"] = float(self.current_tlora_rank)

        # Step LR scheduler once per epoch (after the last batch).
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        if flat_noise_applied and self._flat_lora_noise_cache:
            # Safety fallback: ensure temporary perturbations never leak to next step.
            self._remove_flat_lora_noise()

        return loss_summary

    

    @torch.no_grad()
    def test(self, split=None):
        # Force a fixed rank for evaluation, so metrics are comparable across epochs
        # and after loading the best checkpoint.
        if getattr(self, "tlora_enabled", False):
            if bool(getattr(self.cfg.TRAINER.DRA, "PRINT_EVAL_RANK", True)):
                print(f"[Eval] fixed tlora rank = {self.tlora_eval_rank}")
            apply_tlora_rank_mask(
                self.list_lora_layers,
                current_step=0,
                total_steps=1,
                max_rank=self.tlora_eval_rank,
                min_rank=self.tlora_eval_rank,
                alpha=1.0,
            )
        return super().test(split=split)
    
    def after_train(self):
        super().after_train()
        if self.save:
            try:
                # Reload the best-performing weights to draw T-SNE on the optimal model
                self.load_model(self.output_dir)
            except FileNotFoundError as exc:
                print(f"[T-SNE] Skip reloading best weights: {exc}")


    def _collect_bn_buffers(self):
        # Save running stats from BatchNorm so best/last checkpoints can be faithfully replayed.
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        bn_buffers = {}

        for module_name, module in model_ref.named_modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            for buffer_name, buffer_value in module.named_buffers(recurse=False):
                full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                bn_buffers[full_name] = buffer_value.detach().cpu().clone()

        return bn_buffers

    def _load_bn_buffers(self, bn_path):
        if not osp.exists(bn_path):
            print(f"Warning: BN buffer checkpoint not found at {bn_path}, keep current BN buffers")
            return

        bn_ckpt = torch.load(bn_path, map_location="cpu")
        bn_state = bn_ckpt.get("bn_buffers", bn_ckpt)
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        model_buffers = dict(model_ref.named_buffers())

        loaded = 0
        for name, value in bn_state.items():
            if name not in model_buffers:
                continue

            target = model_buffers[name]
            src = value.to(target.device)

            if src.shape != target.shape:
                continue

            target.copy_(src)
            loaded += 1

        print(f"BN buffers loaded from {bn_path} (loaded={loaded})")

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()
        for name in names:
            save_dir = osp.join(directory, name)
            mkdir_if_missing(save_dir)
            bn_state = {"bn_buffers": self._collect_bn_buffers()}
            if model_name != "":
                save_lora(self.cfg, self.list_lora_layers, save_dir, filename='LoRA-best')
                # Keep classifier head in sync with the LoRA snapshot.
                classifier_state = {
                    "classifier_layer": {
                        k: v.detach().cpu() for k, v in self.model.classifier_layer.state_dict().items()
                    }
                }
                torch.save(classifier_state, osp.join(save_dir, "classifier-best.pt"))
                torch.save(bn_state, osp.join(save_dir, "bn-best.pt"))
            else:
                save_lora(self.cfg, self.list_lora_layers, save_dir, filename='LoRA-last')
                classifier_state = {
                    "classifier_layer": {
                        k: v.detach().cpu() for k, v in self.model.classifier_layer.state_dict().items()
                    }
                }
                torch.save(classifier_state, osp.join(save_dir, "classifier-last.pt"))
                torch.save(bn_state, osp.join(save_dir, "bn-last.pt"))

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        for name in names:
            if epoch is not None:
                load_lora(self.cfg, self.list_lora_layers, osp.join(directory, name), filename='LoRA-last')
                classifier_path = osp.join(directory, name, "classifier-last.pt")
                bn_path = osp.join(directory, name, "bn-last.pt")
            else:
                load_lora(self.cfg, self.list_lora_layers, osp.join(directory, name), filename='LoRA-best')
                classifier_path = osp.join(directory, name, "classifier-best.pt")
                bn_path = osp.join(directory, name, "bn-best.pt")

            if osp.exists(classifier_path):
                classifier_ckpt = torch.load(classifier_path, map_location=self.device)
                cls_state = classifier_ckpt.get("classifier_layer", classifier_ckpt)
                self.model.classifier_layer.load_state_dict(cls_state, strict=True)
                print(f"Classifier weights loaded from {classifier_path}")
            else:
                print(f"Warning: classifier checkpoint not found at {classifier_path}, keep current classifier weights")

            self._load_bn_buffers(bn_path)