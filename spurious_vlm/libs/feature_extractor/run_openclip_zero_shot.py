import sys
sys.path.insert(0,  '../../')

import torch
from PIL import Image
import open_clip

from tqdm import tqdm
import numpy as np

import os
from libs.dataloader import MultiEnvDataset

import utils.const as const
import utils.sys_const as sys_const
from torch.nn import DataParallel

import torchvision.transforms as tfms


from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    if model_name == const.CLIP_OPEN_VITL14:   
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31', cache_dir=sys_const.MODEL_CACHE)  
    elif model_name == const.CLIP_OPEN_VITB32:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=sys_const.MODEL_CACHE)
    elif model_name == const.CLIP_OPEN_VITH14: 
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir=sys_const.MODEL_CACHE)
    elif model_name == const.CLIP_OPEN_RN50:
        model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai', cache_dir=sys_const.MODEL_CACHE)
    return model, preprocess


def _create_dataloader(filepaths, labels, metadata, transform):
    from torch.utils.data import Dataset, DataLoader
    class FileDataset(Dataset):
        def __init__(self, paths, labels, metadata, transform):
            super(FileDataset, self).__init__()
            self.paths = paths
            self.labels = labels
            self.metadata = metadata
            self.transform = transform

        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, index):
            filename = self.paths[index]
            image = self.transform(Image.open(filename).convert("RGB"))
            label = self.labels[index]
            metadata = self.metadata[index,:]
            return image, label, metadata
    dataset = FileDataset(filepaths, labels, metadata, transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    return dataloader

class Featurizer(torch.nn.Module):
    def __init__(self, model):
        super(Featurizer, self).__init__()
        self.model = model
    def forward(self, x):
        input_features = self.model.encode_image(x)
        return input_features
def extract_image_features(dataset_name, model_name, root_dir='', args=None, batch_size=32):
    assert dataset_name in const.IMAGE_DATA
    labels_text = MultiEnvDataset().get_labels(dataset_name)
    featurizer, preprocess = get_model(model_name)
    model = Featurizer(featurizer)
    if args:
        if args.use_data_parallel:
            model = DataParallel(model)
        model.to("cuda:0")
    if root_dir:
        store_dir = os.path.join(root_dir, f'features/{dataset_name}/{model_name}')
    else:
        store_dir = f'features/{dataset_name}/{model_name}'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    filepaths = MultiEnvDataset().get_file_paths(dataset_name)  #list
    metadata_raw = MultiEnvDataset().get_raw_metadata(dataset_name).detach().cpu().numpy()
    y_raw = MultiEnvDataset().get_raw_y(dataset_name).detach().cpu().numpy()

    filepaths_train = MultiEnvDataset().get_file_paths(dataset_name, split="train")
    metadata_raw_train = MultiEnvDataset().get_raw_metadata(dataset_name, split="train").detach().cpu().numpy()
    y_raw_train = MultiEnvDataset().get_raw_y(dataset_name, split="train").detach().cpu().numpy()
    
    image_embeddings_all = []
    y_all = []
    metadata_all = []
    dataloader = _create_dataloader(filepaths, y_raw, metadata_raw, preprocess)
   
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels, metadata_arr = batch
            image_features = model(images.to("cuda:0"))
            image_embeddings_all.append(image_features.detach().cpu().numpy())
            metadata_all.append(metadata_arr.detach().cpu().numpy())
            y_all.append(labels.detach().cpu().numpy())
    image_embeddings_all = np.concatenate(image_embeddings_all)
    metadata_all = np.concatenate(metadata_all)
    y_all = np.concatenate(y_all)

    image_embeddings_all_train = []
    y_all_train = []
    metadata_all_train = []
    dataloader_train = _create_dataloader(filepaths_train, y_raw_train, metadata_raw_train, preprocess)
    
    with torch.no_grad():
        for batch in tqdm(dataloader_train):
            images, labels, metadata_arr = batch
            image_features = model(images.to("cuda:0"))
            image_embeddings_all_train.append(image_features.detach().cpu().numpy())
            metadata_all_train.append(metadata_arr.detach().cpu().numpy())
            y_all_train.append(labels.detach().cpu().numpy())
    image_embeddings_all_train = np.concatenate(image_embeddings_all_train)
    metadata_all_train = np.concatenate(metadata_all_train)
    y_all_train = np.concatenate(y_all_train)


    
    os.makedirs(os.path.join(store_dir, str(0)), exist_ok=True)
    np.save(os.path.join(store_dir, 'image_emb.npy'), image_embeddings_all)
    np.save(os.path.join(store_dir, 'metadata.npy'), metadata_all)
    np.save(os.path.join(store_dir, 'y.npy'), y_all)
    np.save(os.path.join(store_dir, 'image_emb_train.npy'), image_embeddings_all_train)
    np.save(os.path.join(store_dir, 'metadata_train.npy'), metadata_all_train)
    np.save(os.path.join(store_dir, 'y_train.npy'), y_all_train)
    print(f"features and metadata saved to {os.path.join(store_dir)}")



