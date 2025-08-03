import sys
sys.path.insert(0, '../../')

import torch
import open_clip

from transformers import CLIPProcessor, CLIPModel
from transformers import AlignModel, AlignProcessor
from transformers import AltCLIPModel, AltCLIPProcessor

from tqdm import tqdm
import numpy as np

from libs.dataloader import MultiEnvDataset
import utils.const as const
import utils.sys_const as sys_const
from transformers.utils import logging
from torch.nn import DataParallel
logging.set_verbosity(40)

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP image - text model

SUPPORTED_MODELS = [const.CLIP_BASE_NAME, const.CLIP_LARGE_NAME, const.CLIP_ALIGN_NAME, const.CLIP_BIOMED_NAME, const.CLIP_ALT_NAME, const.CLIP_OPEN_VITL14,\
                    const.CLIP_OPEN_VITB32, const.CLIP_OPEN_VITH14, const.CLIP_OPEN_RN50]

def get_models(model_name):
    if model_name == const.CLIP_BASE_NAME:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=sys_const.MODEL_CACHE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=sys_const.MODEL_CACHE)
        return clip_processor, clip_model
    if model_name == const.CLIP_LARGE_NAME:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=sys_const.MODEL_CACHE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=sys_const.MODEL_CACHE)
        return clip_processor, clip_model
    elif model_name == const.CLIP_ALIGN_NAME:
        clip_processor = AlignProcessor.from_pretrained("kakaobrain/align-base", cache_dir=sys_const.MODEL_CACHE)
        clip_model = AlignModel.from_pretrained("kakaobrain/align-base", cache_dir=sys_const.MODEL_CACHE)
        return clip_processor, clip_model
    elif model_name == const.CLIP_BIOMED_NAME:
        clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir=sys_const.MODEL_CACHE)
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir=sys_const.MODEL_CACHE)
        return tokenizer, clip_model
    elif model_name == const.CLIP_ALT_NAME:
        clip_model = AltCLIPModel.from_pretrained("BAAI/AltCLIP", cache_dir=sys_const.MODEL_CACHE)
        clip_processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP", cache_dir=sys_const.MODEL_CACHE)
        return clip_processor, clip_model
    elif model_name == const.CLIP_OPEN_VITL14:
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31', cache_dir=sys_const.MODEL_CACHE)
        tokenizer = open_clip.get_tokenizer('ViT-L-14', cache_dir=sys_const.MODEL_CACHE)
        return tokenizer, clip_model
    elif model_name == const.CLIP_OPEN_VITB32:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=sys_const.MODEL_CACHE)
        tokenizer = open_clip.get_tokenizer('ViT-B-32', cache_dir=sys_const.MODEL_CACHE)
        return tokenizer, model
    elif model_name == const.CLIP_OPEN_VITH14:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir=sys_const.MODEL_CACHE)
        tokenizer = open_clip.get_tokenizer('ViT-H-14', cache_dir=sys_const.MODEL_CACHE)
        return tokenizer, model
    elif model_name == const.CLIP_OPEN_RN50:
        model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai', cache_dir=sys_const.MODEL_CACHE)
        tokenizer = open_clip.get_tokenizer('RN50', cache_dir=sys_const.MODEL_CACHE)
        return tokenizer, model
    
class TextEncoder(torch.nn.Module):
    def __init__(self, model, processor, model_type):
        super(TextEncoder, self).__init__()
        self.model_type = model_type
        self.model = model
        self.processor = processor

    def forward(self, tokens):
        if self.model_type == "openclip":
            text_features = self.model.encode_text(tokens)
        else:
            text_features = self.model.get_text_features(**tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
        return text_features
    
    def tokenize(self, texts, device="cpu"):
        if self.model_type == "openclip":
            tokens = self.processor(texts)
            tokens = tokens.to(device)
        else:
            if type(texts)==np.ndarray:
                texts = list(texts)
            tokens = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            tokens = {key: value.to(device) for key, value in tokens.items()}
        return tokens

def construct_text_encoder(model_name='clip', args=None):
    assert model_name in SUPPORTED_MODELS
    clip_processor, clip_model = get_models(model_name)
    
    if model_name  in [const.CLIP_BIOMED_NAME, const.CLIP_OPEN_VITB32, const.CLIP_OPEN_VITL14, \
                               const.CLIP_OPEN_VITH14, const.CLIP_OPEN_RN50]: # open clip models
        model_type = "openclip"
    else:
        model_type = "clip"

    model = TextEncoder(clip_model,clip_processor, model_type)
    if args:
        if args.use_data_parallel:
            model = DataParallel(model)
        model.to("cuda:0")
    model.eval()
    return model


def get_text_embedding(prompts, model):
    text_emb_all = []
    with torch.no_grad():
        if hasattr(model, "module"):
            tokens = model.module.tokenize(prompts, "cuda:0")
        else:
            tokens = model.tokenize(prompts, "cuda:0")
        text_emb_all = model(tokens)

        text_emb_all = text_emb_all / torch.norm(text_emb_all, dim=-1, keepdim=True)
        text_emb_all = text_emb_all.detach().cpu().numpy()
    return text_emb_all.squeeze()

    # return text_emb_all