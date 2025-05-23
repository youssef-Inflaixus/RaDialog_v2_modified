#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from LLAVA.biovil_t.model import ImageModel
from LLAVA.biovil_t.pretrained import _download_biovil_t_image_model_weights
from LLAVA.biovil_t.types import ImageEncoderType
from LLAVA.llava.model.multimodal_projector.builder import build_vision_projector

try:
    from LLAVA.llava.model import *
    from LLAVA.llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
except:
    from llava.model import *
    from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    print("Model base: ", model_base)
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        # kwargs['torch_dtype'] = torch.float16
        kwargs['torch_dtype'] = torch.bfloat16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if 'LLaVAMed' in model_base:
                lora_cfg_pretrained.mm_projector_type = 'linear' #for LLaVA med
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            if model.config.mm_vision_tower == 'biovil':
                # reset mm_projector as wrong shape is loaded from pretrained base model
                model.model.mm_projector = build_vision_projector(model.config)
                model.model.mm_projector.to(device=model.device, dtype=model.dtype)

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables_extended.bin')): #TODO only for fixed runs, can be deleted later
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables_extended.bin'), map_location='cpu')
                non_lora_trainables = {(k[7:] if k.startswith('module.') else k): v for k, v in non_lora_trainables.items()}
            elif os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.bfloat16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.bfloat16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        if model.config.mm_vision_tower == 'biovil':
            # biovilt_checkpoint_path = _download_biovil_t_image_model_weights()
            # model_type = ImageEncoderType.RESNET50_MULTI_IMAGE
            # vision_tower = ImageModel(img_encoder_type=model_type,
            #                           joint_feature_size=128,
            #                           pretrained_model_path=biovilt_checkpoint_path)
            # model.model.vision_tower = vision_tower
            model_type = ImageEncoderType.UNETPP_EFFICIENTNET_MULTI_IMAGE
            vision_tower = ImageModel(
                img_encoder_type=model_type,
                joint_feature_size=128,
                pretrained_model_path=None  # No pretrained BioViL checkpoint needed
            )
            model.model.vision_tower = vision_tower
        else:
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.bfloat16)
        image_processor = vision_tower.image_processor
        # if non_lora_trainables contains something about vision_tower, load it
        if non_lora_trainables is not None and any(k.startswith('model.vision_tower.') for k in non_lora_trainables):
            new_vision_tower_state_dict = {}
            for k, v in non_lora_trainables.items():  # we need remapping, because state_dict from model is always like model.vision_tower. It should be vision_tower.
                if 'model.vision_tower.vision_tower.' in k: #original CLIP
                    new_k = k.replace('model.vision_tower.', '')
                    new_vision_tower_state_dict[new_k] = v
                elif 'model.vision_tower' in k: #biovil
                    new_k = k.replace('model.vision_tower.', '')
                    new_vision_tower_state_dict[new_k] = v
            print('Loaded additional vision tower weights...')
            vision_tower.load_state_dict(new_vision_tower_state_dict, strict=False)            # weight difference sum([torch.norm(value-vision_tower.state_dict()[key].cpu())  for key,value in new_vision_tower_state_dict.items()])

        # image_pooler = model.get_image_pooler()
        # if image_pooler is not None:
        #     image_pooler.to(device=device, dtype=torch.float16)
        #     if non_lora_trainables is not None and any(k.startswith('model.image_pooler.') for k in non_lora_trainables):
        #         new_image_pooler_state_dict = {}
        #         for k, v in non_lora_trainables.items():  # we need remapping, because state_dict from model is always like model.vision_tower. It should be vision_tower.
        #             if 'model.image_pooler.' in k:
        #                 new_k = k.replace('model.image_pooler.', '')
        #                 new_image_pooler_state_dict[new_k] = v
        #         print('Loading additional image pooler weights...')
        #         image_pooler.load_state_dict(new_image_pooler_state_dict, strict=True)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
