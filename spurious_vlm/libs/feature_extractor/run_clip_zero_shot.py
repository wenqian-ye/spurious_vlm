import sys
sys.path.insert(0, '../../')

import torch

from transformers import CLIPProcessor, CLIPModel
from transformers import AlignProcessor, AlignModel
from transformers import BlipProcessor, BlipModel
from transformers import AutoProcessor, AutoModel, FlavaModel
from transformers import AltCLIPModel, AltCLIPProcessor

from tqdm import tqdm
import numpy as np
import os

from libs.dataloader import MultiEnvDataset
import utils.const as const
import utils.sys_const as sys_const
from torch.nn import DataParallel
from PIL import Image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    if model_name == const.CLIP_BASE_NAME:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=sys_const.MODEL_CACHE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=sys_const.MODEL_CACHE)
    if model_name == const.CLIP_LARGE_NAME:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=sys_const.MODEL_CACHE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=sys_const.MODEL_CACHE)
    elif model_name == const.CLIP_ALIGN_NAME:
        clip_processor = AlignProcessor.from_pretrained("kakaobrain/align-base", cache_dir=sys_const.MODEL_CACHE)
        clip_model = AlignModel.from_pretrained("kakaobrain/align-base", cache_dir=sys_const.MODEL_CACHE)
    elif model_name == const.CLIP_ALT_NAME:
        clip_model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        clip_processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP", cache_dir=sys_const.MODEL_CACHE)
    return clip_model, clip_processor


def _create_dataloader(filepaths, labels, metadata, processor):
    from torch.utils.data import Dataset, DataLoader
    class FileDataset(Dataset):
        def __init__(self, paths, labels, metadata, processor):
            super(FileDataset, self).__init__()
            self.paths = paths
            self.labels = labels
            self.metadata = metadata
            self.processor = processor

        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, index):
            filename = self.paths[index]
            image = Image.open(filename).convert("RGB")
            image = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            label = self.labels[index]
            metadata = self.metadata[index,:]
            return image, label, metadata
    dataset = FileDataset(filepaths, labels, metadata, processor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    return dataloader


class Featurizer(torch.nn.Module):
    def __init__(self, model):
        super(Featurizer, self).__init__()
        self.model = model
    def forward(self, x):
        input_features = self.model.get_image_features(x)
        return input_features

def extract_image_features(dataset_name, model_name, root_dir='', args=None, batch_size=32):
    assert dataset_name in const.IMAGE_DATA
    labels_text = MultiEnvDataset().get_labels(dataset_name)
    clip_model, clip_processor = get_model(model_name)
    model = Featurizer(clip_model)
    if args:
        if args.use_data_parallel:
            model = DataParallel(model)
        model.to("cuda:0")
    model.eval()
    if root_dir:
        store_dir = os.path.join(root_dir, f'features/{dataset_name}/{model_name}')
    else:
        store_dir = f'features/{dataset_name}/{model_name}'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    
    # dataloaders = MultiEnvDataset().get_dataloaders(dataset_name, batch_size)
    filepaths = MultiEnvDataset().get_file_paths(dataset_name)
    metadata_raw = MultiEnvDataset().get_raw_metadata(dataset_name).detach().cpu().numpy()
    y_raw = MultiEnvDataset().get_raw_y(dataset_name).detach().cpu().numpy()

    filepaths_train = MultiEnvDataset().get_file_paths(dataset_name, split="train")
    metadata_raw_train = MultiEnvDataset().get_raw_metadata(dataset_name, split="train").detach().cpu().numpy()
    y_raw_train = MultiEnvDataset().get_raw_y(dataset_name, split="train").detach().cpu().numpy()


    dataloader = _create_dataloader(filepaths, y_raw, metadata_raw, clip_processor)
    dataloader_train = _create_dataloader(filepaths_train, y_raw_train, metadata_raw_train, clip_processor)


    image_embeddings_all = []
    y_all = []
    metadata_all = []
    for j, labeled_batch in tqdm(enumerate(dataloader)):
        x, y, metadata = labeled_batch
        metadata = metadata.detach().cpu().numpy()
        with torch.no_grad():
            img_embedding = model(x.to("cuda:0"))
            
        image_embeddings_all.append(img_embedding.detach().cpu().numpy())
        metadata_all.append(metadata)
        y = y.detach().cpu().numpy().tolist()
        y_all.extend(y)
    image_embeddings_all = np.concatenate(image_embeddings_all)
    metadata_all = np.concatenate(metadata_all)
    y_all = np.array(y_all)


    image_embeddings_all_train = []
    y_all_train = []
    metadata_all_train = []
    for j, labeled_batch in tqdm(enumerate(dataloader_train)):
        x, y, metadata = labeled_batch
        metadata = metadata.detach().cpu().numpy()
        with torch.no_grad():
            img_embedding = model(x.to("cuda:0"))         
        image_embeddings_all_train.append(img_embedding.detach().cpu().numpy())
        metadata_all_train.append(metadata)
        y = y.detach().cpu().numpy().tolist()
        y_all_train.extend(y)
    
    
    image_embeddings_all_train = np.concatenate(image_embeddings_all_train)
    metadata_all_train = np.concatenate(metadata_all_train)
    y_all_train = np.array(y_all_train)

    np.save(os.path.join(store_dir, 'image_emb.npy'), image_embeddings_all)
    if len(labeled_batch) == 3:
        np.save(os.path.join(store_dir, 'metadata.npy'), metadata_all)
    np.save(os.path.join(store_dir, 'y.npy'), y_all)

    np.save(os.path.join(store_dir, 'image_emb_train.npy'), image_embeddings_all_train)
    if len(labeled_batch) == 3:
        np.save(os.path.join(store_dir, 'metadata_train.npy'), metadata_all_train)
    np.save(os.path.join(store_dir, 'y_train.npy'), y_all_train)

    print(f"features and metadata saved to {store_dir}")



