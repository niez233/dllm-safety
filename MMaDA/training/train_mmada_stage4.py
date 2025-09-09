
# Copyright 2025 MMaDA Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import pandas
import logging
import math
import shutil
import time
import html
import random
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.data import Text2ImageDataset
from training.utils import get_config, flatten_omega_conf, image_transform, image_transform_squash
from training.imagenet_dataset import ImageNetDataset
from parquet import RefinedWebDataset, ChatDataset, VQADataset

from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import ImageReward as RM
try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = (config.training.batch_size_t2i
                                + config.training.batch_size_lm
                                + config.training.batch_size_mmu)
    total_batch_size = (
            (config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu)
            * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.tokenizer_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)
    
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16).to(accelerator.device)

    mask_id = model.config.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
            config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Data for generation
    if config.dataset.gen_type == "t2i":
        dataset = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            tokenizer=uni_prompting.text_tokenizer,  # we want to get raw texts, tokenizer is just for length counting
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_t2i,
            per_gpu_batch_size=config.training.batch_size_t2i,
            global_batch_size=total_batch_size_t2i_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
            external_text_to_image_2M_512_caption_path=dataset_config.external_text_to_image_2M_512_caption_path,
        )
        train_dataloader_t2i = dataset.train_dataloader
        num_update_steps_per_epoch = math.ceil(
            train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    elif config.dataset.gen_type == "t2i_parquet":
        # this part relies on the internal packages, which will not be released
        num_update_steps_per_epoch = math.ceil(config.experiment.max_train_examples_t2i / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

        train_dataloader_t2i = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            batch_size=config.training.batch_size_t2i,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size
        )

    elif config.dataset.gen_type == "imagenet1k":
        dataset_imagenet = ImageNetDataset(
            dataset_config.train_t2i_shards_path_or_url,
            image_size=preproc_config.resolution,
        )

        print('process index : ',
              accelerator.process_index, ', ', accelerator.num_processes,
              "Length: ", len(dataset_imagenet))

        if accelerator.num_processes > 1:
            sampler = DistributedSampler(dataset_imagenet,
                                         num_replicas=accelerator.num_processes,
                                         rank=accelerator.process_index,
                                         shuffle=True,
                                         )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_dataloader_t2i = DataLoader(dataset_imagenet, batch_size=config.training.batch_size_t2i,
                                          sampler=sampler, collate_fn=dataset_imagenet.collate_fn,
                                          shuffle=shuffle, num_workers=dataset_config.num_workers)
        num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    else:
        raise ValueError(f"Unsupported dataset type {config.dataset.type}")

    total_batch_size_mmu_without_accum = config.training.batch_size_mmu * accelerator.num_processes
    # Data for image captioning
    if config.dataset.und_type == "captioning":
        dataset_mmu = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            tokenizer=uni_prompting.text_tokenizer,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_mmu,
            per_gpu_batch_size=config.training.batch_size_mmu,
            global_batch_size=total_batch_size_mmu_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
            external_text_to_image_2M_512_caption_path=dataset_config.external_text_to_image_2M_512_caption_path,
            external_ai2d_caption_path=dataset_config.external_ai2d_caption_path,
            external_clevr_caption_path=dataset_config.external_clevr_caption_path,
            external_docvqa_caption_path=dataset_config.external_docvqa_caption_path,
            external_geo_caption_path=dataset_config.external_geo_caption_path,
            is_captioning=True,
            add_caption_prompt=dataset_config.add_caption_prompt,
        )
        train_dataloader_mmu = dataset_mmu.train_dataloader

    elif config.dataset.und_type == "captioning_parquet":
        train_dataloader_mmu = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            batch_size=config.training.batch_size_mmu,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            is_captioning=True
        )

    else:
        raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    # LLM pure text dataset: RefinedWeb
    dataset_lm = RefinedWebDataset(data_path=dataset_config.train_lm_shards_path_or_url,
                                   rank=accelerator.process_index,
                                   world_size=accelerator.num_processes)
    train_dataloader_lm = torch.utils.data.DataLoader(dataset_lm, batch_size=config.training.batch_size_lm,
                                                      sampler=None, collate_fn=dataset_lm.collate_fn,
                                                      num_workers=dataset_config.num_workers)

    dataset_instruct = ChatDataset(data_path=dataset_config.train_instruct_shards_path_or_url,
                                   rank=accelerator.process_index,
                                   world_size=accelerator.num_processes,
                                   max_length=preproc_config.max_lm_text_length,
                                   tokenizer=uni_prompting.text_tokenizer,
                                   )

    train_dataloader_instruct = torch.utils.data.DataLoader(dataset_instruct, batch_size=config.training.batch_size_lm,
                                                      sampler=None, collate_fn=dataset_instruct.collate_fn,
                                                      num_workers=dataset_config.num_workers)

    dataset_vqa = VQADataset(
        json_path=dataset_config.external_vqa_caption_path,
        image_root=dataset_config.vqa_images_path,
        tokenizer=uni_prompting.text_tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resolution=preproc_config.resolution,
        max_length=preproc_config.max_seq_length
    )
    train_dataloader_vqa = torch.utils.data.DataLoader(dataset_vqa, batch_size=config.training.batch_size_mmu,
                                                       sampler=None, collate_fn=dataset_vqa.collate_fn,
                                                       num_workers=dataset_config.num_workers)

    dataset_clevr2 = VQADataset(
        json_path=dataset_config.external_clevr2_caption_path,
        image_root=dataset_config.clevr2_images_path,
        tokenizer=uni_prompting.text_tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resolution=preproc_config.resolution,
        max_length=preproc_config.max_seq_length
    )
    train_dataloader_clevr2 = torch.utils.data.DataLoader(dataset_clevr2, batch_size=config.training.batch_size_mmu,
                                                       sampler=None, collate_fn=dataset_clevr2.collate_fn,
                                                       num_workers=dataset_config.num_workers)

    dataset_geo170k = VQADataset(
        json_path=dataset_config.external_geo170k_caption_path,
        image_root=dataset_config.geo170k_images_path,
        tokenizer=uni_prompting.text_tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resolution=preproc_config.resolution,
        max_length=preproc_config.max_seq_length,
        image_transform_method = "pad"
    )
    train_dataloader_geo170k = torch.utils.data.DataLoader(dataset_geo170k, batch_size=config.training.batch_size_mmu,
                                                       sampler=None, collate_fn=dataset_geo170k.collate_fn,
                                                       num_workers=dataset_config.num_workers)

    # Combine these dataloaders into a single iterable model
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "instruct_flow": train_dataloader_instruct,
        "mmu_flow": train_dataloader_mmu,
        "vqa_flow": train_dataloader_vqa,
        "clevr2_flow": train_dataloader_clevr2,
        "geo170k_flow": train_dataloader_geo170k,
    }

    # 
    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    start_step = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        logger.info(f"dirs: {dirs}")
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        logger.info(f"path: {path}")
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = start_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            if os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin'):
                state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            elif os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin.index.json'):
                from safetensors.torch import load_file
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, f'{path}/unwrapped_model/')
            # if safetensors sharded checkpoint exists
            elif os.path.exists(f'{path}/unwrapped_model/model.safetensors.index.json'):
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(
                    model, 
                    f'{path}/unwrapped_model/',
                )
            else:
                raise FileNotFoundError(f"Checkpoint {path}/unwrapped_model/pytorch_model.bin not found")
    else:
        logger.info("Not resuming from checkpoint")

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    mask_dtype = model.get_input_embeddings().weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
            texts: Union[str, list[str]],
            min_masking_rate: float = 0.0,
            is_train: bool = True,
            seed: int = None
    ):

        image_tokens = vq_model.get_code(pixel_values_or_image_ids)
        image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
            seed=seed
        )
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')
        return input_ids, labels, mask_prob, image_tokens, masks

    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        texts: Union[str, list[str]], max_seq_len, eps=1e-3
    ):
        # create MLM mask and labels
        
        input_ids_lm, attention_mask, labels_lm = uni_prompting((texts, max_seq_len), 'lm')
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id 
        answer_lengths_lm = torch.sum(attention_mask, dim=-1, keepdim=True)
        answer_lengths_lm = answer_lengths_lm.clamp(min=1)
        answer_lengths_lm = answer_lengths_lm.repeat(1, noisy_batch.shape[1])
        
        return noisy_batch, labels_lm, p_mask, answer_lengths_lm

    @torch.no_grad()
    def prepare_inputs_and_labels_for_chat_text(
        texts: Union[str, list[str]], max_seq_len, eps=1e-3
    ):
        # create MLM mask and labels
        
        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts, max_seq_len), 'lm_chat')
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id 
        noisy_batch[prompt_mask.bool()] = input_ids_lm[prompt_mask.bool()]
        masked_indices = noisy_batch == mask_id 
        answer_lengths_lm = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths_lm = answer_lengths_lm.clamp(min=1)
        answer_lengths_lm = answer_lengths_lm.repeat(1, noisy_batch.shape[1])
        
        return noisy_batch, labels_lm, p_mask, answer_lengths_lm

    @torch.no_grad()
    def prepare_inputs_and_labels_for_mmu(
        input_ids_mmu, prompt_masks, labels_mmu, eps=1e-3
    ):
        b, l = input_ids_mmu.shape
        t = torch.rand(b, device=input_ids_mmu.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_mmu.device) < p_mask
        # 126336 is used for [MASK] token 
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_mmu)
        masked_indices = noisy_batch == mask_id 
        noisy_batch[prompt_masks.bool()] = input_ids_mmu[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id 

        prompt_masks = prompt_masks.to(torch.int64)    
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])    

        return noisy_batch, labels_mmu, p_mask, answer_lengths



    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:

            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
            batch_size_lm = len(batch["lm_flow"]["input_ids"])
            batch_size_mmu = batch["mmu_flow"]["images"].shape[0]

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)

            # Encode images to image tokens, mask them and create input and labels
            (
                input_ids,
                labels,
                mask_prob,
                image_tokens_ori,
                t2i_masks
            ) = prepare_inputs_and_labels(pixel_values, texts, config.training.min_masking_rate)

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for language modeling
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            max_seq_len = input_ids.shape[-1]

            probs = [config.training.base_in_lm_coeff, config.training.instruct_in_lm_coeff]
            probs_total = sum(probs)
            probs = [p / probs_total for p in probs]
            cum_probs = [sum(probs[:i+1]) for i in range(len(probs))]
            rand_val = random.random()
            if rand_val < cum_probs[0]:
                texts_lm = batch["lm_flow"]["input_ids"]
                (
                    input_ids_lm,  
                    labels_lm,
                    p_mask_lm,
                    answer_lengths_lm
                ) = prepare_inputs_and_labels_for_text(texts_lm, max_seq_len)  
                input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
                labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)
            else:
                texts_lm = batch["instruct_flow"]["input_ids"]
                (
                    input_ids_lm,  
                    labels_lm,
                    p_mask_lm,
                    answer_lengths_lm
                ) = prepare_inputs_and_labels_for_chat_text(texts_lm, max_seq_len)  
                input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
                labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)

            

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for captioning/multimodal understanding
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            if "llava" in config.dataset.und_type:
                pixel_values_mmu, input_ids_mmu, labels_mmu = (batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"],batch["mmu_flow"]["labels"])
                pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

                input_ids_mmu = torch.cat([
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(
                        accelerator.device),
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(
                        accelerator.device),
                    image_tokens_mmu,
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(
                        accelerator.device),
                    input_ids_mmu,
                ], dim=1).long()

                labels_mmu = torch.cat([
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                    torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,
                (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                    labels_mmu.to(accelerator.device)
                ], dim=1).long()

            else:
                probs = [config.training.cot_in_mmu_coeff, config.training.vqa_in_mmu_coeff, config.training.clevr2_in_mmu_coeff, config.training.geo170k_in_mmu_coeff]
                probs_total = sum(probs)
                probs = [p / probs_total for p in probs]
                cum_probs = [sum(probs[:i+1]) for i in range(len(probs))]
                rand_val = random.random()
                if rand_val < cum_probs[0]:
                    pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]
                elif rand_val < cum_probs[1]:
                    pixel_values_mmu, texts_mmu = batch["vqa_flow"]["images"], batch["vqa_flow"]["input_ids"]
                elif rand_val < cum_probs[2]:
                    pixel_values_mmu, texts_mmu = batch["clevr2_flow"]["images"], batch["clevr2_flow"]["input_ids"]
                else:
                    pixel_values_mmu, texts_mmu = batch["geo170k_flow"]["images"], batch["geo170k_flow"]["input_ids"]
                pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
                
                input_ids_mmu, prompt_masks, labels_mmu = uni_prompting((image_tokens_mmu, texts_mmu), 'mmu')
                (
                    input_ids_mmu,  
                    labels_mmu,
                    p_mask_mmu,
                    answer_lengths
                ) = prepare_inputs_and_labels_for_mmu(input_ids_mmu, prompt_masks, labels_mmu)
                input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)


            input_ids = torch.cat((input_ids, input_ids_mmu.to(input_ids.device)), dim=0)
            labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)
            
            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            with accelerator.accumulate(model):
                logits, loss_t2i, loss_lm, loss_mmu = model.forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_lm=batch_size_lm,
                    batch_size_mmu=batch_size_mmu,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    p_mask_lm=p_mask_lm,
                    p_mask_mmu=p_mask_mmu,  
                    answer_lengths=answer_lengths,
                    t2i_masks=t2i_masks,
                    answer_lengths_lm=answer_lengths_lm
                )
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_t2i = accelerator.gather(loss_t2i.repeat(config.training.batch_size_t2i)).mean()
                avg_loss_lm = accelerator.gather(loss_lm.repeat(config.training.batch_size_lm)).mean()
                avg_loss_mmu = accelerator.gather(loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                loss = config.training.t2i_coeff * loss_t2i + \
                       config.training.lm_coeff * loss_lm + \
                       config.training.mmu_coeff * loss_mmu

                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_t2i)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_mmu": avg_loss_mmu.item(),
                        "step_loss_lm": avg_loss_lm.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()


                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1, uni_prompting)

                if ((global_step + 1) % config.experiment.generate_every == 0 or global_step == start_step) and accelerator.is_main_process:
                    quantative_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        force_no_cfg=False
                    )

                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        force_no_cfg=False
                    )

                    visualize_predictions(
                        model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step + 1,
                        input_ids,
                        image_tokens_ori,
                        batch["t2i_flow"]["images"],
                        texts,
                        logits,
                        accelerator
                    )
                    
                    understanding_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                    )

                    generate_chat_text(
                        model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                    )

                global_step += 1
            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step, uni_prompting)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()


@torch.no_grad()
def visualize_predictions(
        model,
        vq_model,
        uni_prompting,
        config,
        global_step,
        input_ids,
        image_tokens_ori,
        ori_images,
        texts,
        logits,
        accelerator
):
    logger.info("Visualizing predictions...")
    model.eval()

    recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
    recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predictions = logits[:config.training.batch_size_t2i, -(config.model.mmada.num_vq_tokens + 1):-1:, len(uni_prompting.text_tokenizer) + config.model.mmada.num_new_special_tokens: len(uni_prompting.text_tokenizer) + config.model.mmada.num_new_special_tokens + config.model.mmada.codebook_size]
    predictions = predictions.argmax(axis=-1)
    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id - len(uni_prompting.text_tokenizer)
    input_ids = input_ids[:config.training.batch_size_t2i, -(config.model.mmada.num_vq_tokens + 1):-1:] - len(uni_prompting.text_tokenizer)
    mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
        dim=-1) / config.model.mmada.num_vq_tokens).cpu().numpy())
    predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)
    predicted_images = vq_model.decode_code(predicted_images)
    predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    pil_images = [Image.fromarray(image) for image in predicted_images]

    # Log images
    wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} \n caption: {texts[i]}') for i, (image, r) in
                    enumerate(zip(pil_images, mask_ratio))]
    wandb.log({"Original images v.s. Reconstructed images v.s. Predicted images": wandb_images}, step=global_step)

    model.train()


@torch.no_grad()
def generate_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
        force_no_cfg = False
):
    logger.info("Generating images...")
    model.eval()

    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()


    mask_dtype = model.get_input_embeddings().weight.dtype
    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id
    image_tokens = torch.ones((len(validation_prompts), config.model.mmada.num_vq_tokens), dtype=torch.long,
                              device=accelerator.device) * mask_token_id
    input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    if not force_no_cfg and config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
        cfg_scale = config.training.guidance_scale
    else:
        uncond_input_ids = None
        uncond_attention_mask = None
        cfg_scale = 0
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    
    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=cfg_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.mmada.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    wandb.log({f"Generated images with cfg {cfg_scale}": wandb_images}, step=global_step)





@torch.no_grad()
def quantative_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
        force_no_cfg = False
):
    logger.info("Quantative images...")
    model.eval()
    clip_score_fn = partial(clip_score, model_name_or_path="/data_storage/shared/pretrained_models/")
    image_reward_model = RM.load("/data_storage/shared/pretrained_models/ImageReward/ImageReward.pt")
    # read validation prompts from file
    with open(config.validation.quantative_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()
    mask_dtype = model.get_input_embeddings().weight.dtype
    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id
    image_tokens = torch.ones((len(validation_prompts), config.model.mmada.num_vq_tokens), dtype=torch.long,
                              device=accelerator.device) * mask_token_id
    input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    if not force_no_cfg and config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
        cfg_scale = config.training.guidance_scale
    else:
        uncond_input_ids = None
        uncond_attention_mask = None
        cfg_scale = 0
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    validation_batch_size = config.validation.quantative_batch_size

    pil_images = [] 
    clip_scores = []
    image_rewards = []
    for i in range(0, len(validation_prompts), validation_batch_size):
        batch_input_ids = input_ids[i:i+validation_batch_size]
        batch_attention_mask = attention_mask[i:i+validation_batch_size]
        batch_uncond_input_ids = uncond_input_ids[i:i+validation_batch_size]
        batch_uncond_attention_mask = uncond_attention_mask[i:i+validation_batch_size]
        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            # Generate images
            gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
                input_ids=batch_input_ids,
                uncond_input_ids=batch_uncond_input_ids,
                attention_mask=batch_attention_mask,
                uncond_attention_mask=batch_uncond_attention_mask,
                guidance_scale=cfg_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                predict_all_tokens=config.training.get("predict_all_tokens", False),
                seq_len=config.model.mmada.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )
        # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
        # so we clamp them to the correct range.
        gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        image_tensor = images.to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        batch_pil_images = [Image.fromarray(image) for image in images]
        pil_images.extend(batch_pil_images)

        # calculate CLIP score
        batch_clip_score = clip_score_fn(image_tensor, validation_prompts[i:i+validation_batch_size])
        # calculate image reward score
        for j in range(validation_batch_size):
            clip_scores.append(clip_score_fn(image_tensor[j], validation_prompts[i+j]))
            image_reward_score = image_reward_model.score(validation_prompts[i+j], batch_pil_images[j])
            image_rewards.append(image_reward_score)
    
    clip_scores = torch.tensor(clip_scores)
    image_rewards = torch.tensor(image_rewards)
    logger.info(f"clip_scores: {clip_scores}, image_rewards: {image_rewards}")
    clip_scores_mean = clip_scores.mean()
    image_rewards_mean = image_rewards.mean()
    logger.info(f"CLIP score mean: {clip_scores_mean}, Image reward score mean: {image_rewards_mean}")
    accelerator.log({"clip_score": clip_scores_mean, "image_reward_score": image_rewards_mean}, step=global_step)



        # Log images
    wandb_images = [wandb.Image(image, caption=f"{validation_prompts[i]} \n CLIP score: {clip_scores[i]}, Image reward score: {image_rewards[i]}") for i, image in enumerate(pil_images[:validation_batch_size])]
    wandb.log({f"Quantative images with cfg {cfg_scale}": wandb_images}, step=global_step)


    if config.training.get("pre_encode", False):
        del vq_model
   
    model.train()

    
    
    

@torch.no_grad()
def understanding_images(
        model,
        vq_model,
        uni_prompting, # 包含了 text_tokenizer
        accelerator,
        config,
        global_step,
):
    """
    Processes images and multi-turn conversation prompts for image understanding,
    generates responses, and logs results to Weights & Biases.
    Uses tokenizer.apply_chat_template for handling conversation history.
    """
    logger.info("Understanding images (multi-turn)...")
    model.eval()
    prompts_file_path = config.dataset.params.mmu_validation_prompts_file
    image_root = config.dataset.params.mmu_image_root
    try:
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            validation_data = json.load(f) 
        logger.info(f"Successfully loaded {len(validation_data)} validation items from {prompts_file_path}")
    except Exception as e:
        logger.error(f"Error loading prompts from {prompts_file_path}: {e}. Skipping image understanding.")
        model.train() 
        return
    wandb_logs = []
    device = accelerator.device
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    for item in validation_data:
        file_name = item.get('file_name')
        messages = item.get('messages') 
        if not file_name or not messages:
            logger.warning(f"Skipping item due to missing 'file_name' or 'messages': {item}")
            continue
        image_path = os.path.join(image_root, file_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}. Skipping.")
            continue
        try:
            image_ori = Image.open(image_path).convert("RGB")
            if any(tag in file_name for tag in ['ai2d', 'clevr', 'docvqa', 'geo', 'llava']):
                 image = image_transform_squash(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
            else:
                 image = image_transform(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
            image = image.unsqueeze(0)
            image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
            batch_size = image_tokens.shape[0]
            text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            input_ids = torch.cat([
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                text_token_ids
            ], dim=1).long()
            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                output_ids = accelerator.unwrap_model(model).mmu_generate(
                    input_ids,
                    max_new_tokens=config.dataset.preprocessing.max_seq_length,
                    steps=config.dataset.preprocessing.max_seq_length // 2,
                    block_length=config.dataset.preprocessing.max_seq_length // 4,
                )
            generated_ids = output_ids[:, input_ids.shape[1]:]
            response_text = uni_prompting.text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            conversation_str = f"Image: {file_name}\n" + "="*20 + "\n"
            conversation_str = f"Image: {file_name}\n" + "="*20 + "\n"
            for msg in messages:
                role_prefix = "User: " if msg['role'] == 'user' else "Assistant: "
                conversation_str += f"{role_prefix}{msg['content']}\n"
            conversation_str += f"Assistant (Generated): {response_text}\n"
            log_image_tensor = torch.clamp((image.squeeze(0) + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
            log_image_np = log_image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(log_image_np)
            wandb_logs.append(wandb.Image(pil_image, caption=conversation_str.strip()))
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}", exc_info=True)
    if wandb_logs:
        try:
            wandb.log({"Understanding images (multi-turn)": wandb_logs}, step=global_step)
            logger.info(f"Logged {len(wandb_logs)} understanding image results to W&B for step {global_step}.")
        except Exception as e:
            logger.error(f"Failed to log understanding images to W&B: {e}")
    else:
        logger.warning("No images were successfully processed for understanding in this step.")
    model.train() 

@torch.no_grad()
def generate_chat_text(
        model,
        uni_prompting,
        accelerator,
        config,
        global_step,
):
    logger.info("Generating chat text...")
    model.eval()

    df = pandas.read_json(config.dataset.params.lm_chat_validation_jsonl, lines=True)
    prompts = df['question'].tolist()
    responses = [''] * len(prompts)

    device = accelerator.device

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    html_content = "<div style='font-family:Arial, sans-serif;'>"
    html_content += f"<h2 style='color:navy;'>Step {global_step}</h2>"

    for i, prompt in enumerate(prompts):
        original_prompt = prompt

        prompt_with_tags = "<|start_header_id|>user<|end_header_id|>\n" + f"{prompt}" + "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
        token_ids = uni_prompting.text_tokenizer([prompt_with_tags])['input_ids'][0]
        token_ids = [uni_prompting.text_tokenizer.bos_token_id] + token_ids
        input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            output_ids = accelerator.unwrap_model(model).mmu_generate(
                input_ids, 
                max_new_tokens=config.dataset.preprocessing.max_seq_length, 
                steps=config.dataset.preprocessing.max_lm_text_length // 2, 
                block_length=config.dataset.preprocessing.max_seq_length // 4
            )
        text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        responses[i] += text[0]

        escaped_prompt = html.escape(original_prompt)
        escaped_response = html.escape(responses[i])
        html_content += f"""
        <div style='border: 1px solid #ddd; margin:10px 0; padding:10px;'>
          <h4 style='margin: 0;'>Prompt</h4>
          <p style='margin: 0;'>{escaped_prompt}</p>
          <h4 style='margin: 0; margin-top:5px;'>Response</h4>
          <p style='margin: 0;'>{escaped_response}</p>
        </div>
        """

    html_content += "</div>"  

    model.train()

    wandb.log({"chat_text_generation": wandb.Html(html_content)}, step=global_step)




def save_checkpoint(model, config, accelerator, global_step, uni_prompting):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

        # save tokenizer
        uni_prompting.text_tokenizer.save_pretrained(save_path/ "unwrapped_model")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


        



if __name__ == "__main__":
    main()
