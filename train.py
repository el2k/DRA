import argparse

import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data.datasets import OfficeHome, VisDA17, Office31

# custom
from trainers import *


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir
        
    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if hasattr(cfg.TRAINER, "DRA"):
        if args.use_gpt_prompts:
            cfg.TRAINER.DRA.USE_GPT_PROMPTS = True
        if args.gpt_prompts_json:
            cfg.TRAINER.DRA.GPT_PROMPT_JSON = args.gpt_prompts_json
        elif bool(getattr(cfg.TRAINER.DRA, "USE_GPT_PROMPTS", False)):
            dataset_prompt_map = {
                "Office31": "gpt_file/office31_prompt.json",
                "OfficeHome": "gpt_file/officeHome_prompt.json",
                "VisDA17": "gpt_file/visda_prompt.json",
                "DomainNet": "gpt_file/domainnet_prompt.json",
                "miniDomainNet": "gpt_file/miniDomainNet_prompt.json",
            }
            auto_prompt = dataset_prompt_map.get(cfg.DATASET.NAME, "")
            if auto_prompt:
                cfg.TRAINER.DRA.GPT_PROMPT_JSON = auto_prompt
    
    if args.gpu:   
        cfg.GPU = args.gpu
    
    if args.save:
        cfg.SAVE_MODLE = args.save
        
    if args.domains:
        cfg.DOMAINS = args.domains
        if cfg.DATASET.NAME == "OfficeHome":
            DOMAINS = {'a': "art", 'c':"clipart", 'p':"product", 'r':"real_world"}
            cfg.CONFI = 0.7
            cfg.WARM_UP = 0
            cfg.EPOCH = 10
            if hasattr(cfg.TRAINER, "DRA"):
                cfg.TRAINER.DRA.FREEMATCH_EMA_M = 0.90
                cfg.TRAINER.DRA.CONFI_START = 0.85
                cfg.TRAINER.DRA.CONFI_END = 0.70
               
        elif cfg.DATASET.NAME == "VisDA17":
            DOMAINS = {'s': "synthetic", 'r':"real"}
            cfg.CONFI = 0.6
            cfg.WARM_UP = 0
            cfg.EPOCH = 10
            if hasattr(cfg.TRAINER, "DRA"):
                cfg.TRAINER.DRA.FREEMATCH_EMA_M = 0.905
                cfg.TRAINER.DRA.CONFI_START = 0.85
                cfg.TRAINER.DRA.CONFI_END = 0.70
                
        elif cfg.DATASET.NAME == "Office31":
            DOMAINS = {'a': "amazon", 'w': "webcam", 'd': "dslr"}
            # Office31 is sensitive to over-strict pseudo-label filtering.
            cfg.CONFI = 0.9
            cfg.WARM_UP = 1
            cfg.EPOCH = 0
            if hasattr(cfg.TRAINER, "DRA"):
                cfg.TRAINER.DRA.FREEMATCH_EMA_M = 0.999
                cfg.TRAINER.DRA.CONFI_START = 0.90
                cfg.TRAINER.DRA.CONFI_END = 0.80
                 
        elif cfg.DATASET.NAME == "DomainNet":
            DOMAINS = {'c': "clipart", 'i': "infograph", 'p': "painting", 'q': "quickdraw", 'r': "real", 's': "sketch"}
            cfg.CONFI = 0.7
            cfg.WARM_UP = 0
            if hasattr(cfg.TRAINER, "DRA"):
                cfg.TRAINER.DRA.FREEMATCH_EMA_M = 0.95
                cfg.TRAINER.DRA.CONFI_START = 0.85
                cfg.TRAINER.DRA.CONFI_END = 0.70
              
        elif cfg.DATASET.NAME == "miniDomainNet":
            DOMAINS = {'c': "clipart", 'p': "painting", 'r': "real", 's': "sketch"}
            cfg.CONFI = 0.7
            cfg.WARM_UP = 0
            if hasattr(cfg.TRAINER, "DRA"):
                cfg.TRAINER.DRA.FREEMATCH_EMA_M = 0.92
                cfg.TRAINER.DRA.CONFI_START = 0.82
                cfg.TRAINER.DRA.CONFI_END = 0.70
                

        source_domain, target_domain = args.domains.split('-')[0], args.domains.split('-')[1]
        cfg.DATASET.SOURCE_DOMAINS = [DOMAINS[source_domain]]
        cfg.DATASET.TARGET_DOMAINS = [DOMAINS[target_domain]]


def extend_cfg(cfg, args):
 
    from yacs.config import CfgNode as CN

    cfg.MODEL.BACKBONE.PATH = "./assets"    # path of pretrained model
    cfg.MODEL.PATCH_SIZE = 16
    cfg.MODEL.HIDDEN_SIZE = 768     # as model change, this param need to be changed
    cfg.MODEL.NUM_LAYER = 12        # as model change, this param need to be changed
    cfg.DATASET.NUM_SHOTS = None    # optional
    cfg.SAVE_MODEL = True
    cfg.TEST.FINAL_MODEL = "best_val"
        
    cfg.TRAINER.DRA = CN()
    cfg.TRAINER.DRA.PREC = "fp16"
    cfg.TRAINER.DRA.DROPOUT = 0.0
    cfg.TRAINER.DRA.DEEP_LAYERS = None 
    cfg.TRAINER.DRA.SHARE_LAYER = cfg.TRAINER.DRA.DEEP_LAYERS
    
    cfg.TRAINER.DRA.TP = False
    cfg.TRAINER.DRA.T_DEEP = False
    cfg.TRAINER.DRA.CSC = False  
    cfg.TRAINER.DRA.N_CTX = 2     # number of text context vectors
    cfg.TRAINER.DRA.CTX_INIT = "a photo of a"
    cfg.TRAINER.DRA.CLASS_TOKEN_POSITION = "end"  
    
    cfg.TRAINER.DRA.VP = False
    cfg.TRAINER.DRA.V_DEEP = cfg.TRAINER.DRA.T_DEEP
    cfg.TRAINER.DRA.NUM_TOKENS = cfg.TRAINER.DRA.N_CTX    # number of visual context vectors
    cfg.TRAINER.DRA.LOCATION = "middle"

    cfg.TRAINER.DRA.POSITION = 'all'
    cfg.TRAINER.DRA.PARAMS = ['q', 'k', 'v']
    cfg.TRAINER.DRA.R = 4
 
    cfg.TRAINER.DRA.USE_TLORA = False
    cfg.TRAINER.DRA.MIN_RANK = 8
    cfg.TRAINER.DRA.ALPHA_RANK_SCALE = 1.0
    cfg.TRAINER.DRA.EVAL_RANK = cfg.TRAINER.DRA.R
    cfg.TRAINER.DRA.PRINT_EVAL_RANK = True
    cfg.TRAINER.DRA.USE_FLAT_LORA = True
    cfg.TRAINER.DRA.FLAT_LORA_RHO = 0.2
    cfg.TRAINER.DRA.FLAT_LORA_INTERVAL = 1 
    cfg.TRAINER.DRA.ALPHA = 1 
    cfg.TRAINER.DRA.DROPOUT_RATE = 0.25

    # Unified pseudo-label stabilization knobs (dataset-agnostic defaults).
    cfg.TRAINER.DRA.BETA_MIN = 0.0
    cfg.TRAINER.DRA.BETA_MAX = 0.95
    cfg.TRAINER.DRA.BETA_WARMUP_EPOCHS = 1
    cfg.TRAINER.DRA.MIN_KEEP_RATE = 0.2
    cfg.TRAINER.DRA.DYNAMIC_THRESH_FLOOR = 0.0


    cfg.TRAINER.DRA.ADAPTER_START = 4
    cfg.TRAINER.DRA.ADAPTER_END = 12
    cfg.TRAINER.DRA.ADAPTER_DIM = 32
    cfg.TRAINER.DRA.ADAPTER_SCALE = 0.1
    # Optional GPT prompt file for replacing class text templates.
    cfg.TRAINER.DRA.USE_GPT_PROMPTS = True

        
        
def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)
    print(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(cfg.MODEL_DIR, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="./results", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="",
                        help="path to config file for dataset setup")
    parser.add_argument("--model-dir", type=str, default="",
                        help="load model from this directory for eval-only mode")
    
    parser.add_argument("--domains", type=str, help="domains for DA/DG")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")

    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--use-gpt-prompts", action="store_true", help="use GPT prompt json as text templates")
    parser.add_argument("--gpt-prompts-json", type=str, default="", help="path to GPT prompt json")
    
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    
    parser.add_argument("--resume", type=str, default="",
                        help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--load-epoch", type=int,
                        help="load model weights at this epoch for evaluation")

    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")    # if you use this hyperpameter, you need modify the source code of dassl library.
                                                                                    # i.e., in dassl.engine.trainer line 314: self.device = torch.device("cuda:{}".format(cfg.GPU))
    parser.add_argument("--seed", type=int, default=2,
                        help="only positive value enables a fixed seed")
    parser.add_argument("--save", type=str, default=False, help="need to save model")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")

    args = parser.parse_args()
    
    main(args)
