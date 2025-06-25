import sys
sys.path.insert(0, '../')
import os
import argparse

import numpy as np
import torch

from wilds import get_dataset

from libs.dataloader import MultiEnvDataset
from libs.get_clip_text_emb import get_text_embedding, construct_text_encoder
from libs.text_prompts import text_prompts
from libs.feature_extractor import hf_extractor, openclip_extractor
import utils.const as const
import utils.sys_const as sys_const

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from save_methods import *

def eval_wilds(preds, test_Y):
    if not torch.is_tensor(test_Y):
        test_Y = torch.Tensor(test_Y)
    metadata = np.load(os.path.join(load_dir, 'metadata.npy'))
    dataset = get_dataset(dataset=dataset_name, download=False, root_dir=sys_const.DATA_DIR)
    _, results_str = dataset.eval(preds, test_Y, torch.Tensor(metadata))
    print(results_str)
    return results_str

def eval_domainbed(y_pred, y_true, logits):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    metadata = np.load(os.path.join(load_dir, 'metadata.npy'))
    if len(metadata.shape) > 1:
        metadata = metadata.flatten()
    unique_domains = np.unique(metadata)
    acc_all = []
    for domain in unique_domains:
        for y in np.unique(y_true):
            d_sample_idx = np.argwhere((metadata== domain) & (y_true==y))
            if len(d_sample_idx) == 0:
                continue
            samples_y_pred = y_pred[d_sample_idx]
            samples_y_true = y_true[d_sample_idx]
            domain_acc = accuracy_score(samples_y_true, samples_y_pred)
            acc_all.append(domain_acc)
            print(domain, y, len(d_sample_idx))
    acc_all = np.array(acc_all)
    msg1 = f"avg acc = {np.mean(acc_all):.3f}"
    msg2 = f"wg acc = {np.amin(acc_all):.3f}"
    print(msg1)
    print(msg2)
    print('\n')
    result_str = f"{msg1}\n{msg2}"
    return result_str

    
def eval_cxr(y_pred, y_true, logits):
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    acc_all = []
    for y in np.unique(y_true):
        class_sample_idx = np.argwhere(y_true==y)
        group_acc = accuracy_score(y_true[class_sample_idx], y_pred[class_sample_idx])
        acc_all.append(group_acc)
        print(y, len(class_sample_idx))
    acc_all = np.array(acc_all)
    msg1 = f'avg acc = {np.mean(acc_all):.3f}'
    msg2 = f'wg acc = {np.amin(acc_all):.3f}'
    print(msg1)
    print(msg2)
    print('\n')
    result_str = f"{msg1}\n{msg2}"
    return result_str

def make_clip_preds(image_features, text_features):
    if not torch.is_tensor(image_features):
        image_features = torch.Tensor(image_features)
    if not torch.is_tensor(text_features):
        text_features = torch.Tensor(text_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return torch.argmax(text_probs, dim=1), text_probs

def group_prompt_preds(raw_preds):
    raw_preds = raw_preds.detach().cpu().numpy()
    raw_preds[np.argwhere((raw_preds == 0) | (raw_preds == 1)).flatten()] = 0
    raw_preds[np.argwhere((raw_preds == 2) | (raw_preds == 3)).flatten()] = 1
    return torch.Tensor(raw_preds)

def group_prompt_preds_multi(raw_preds, n_full_prompt, n_prompt_per_class, n_class):
    c_idx = 0
    for p_idx in range(0, n_full_prompt, n_prompt_per_class):
        idxs = []
        for cp_idx in range(n_prompt_per_class):
            idxs.extend(np.argwhere(raw_preds == p_idx + cp_idx).flatten())
        raw_preds[idxs] = c_idx
        c_idx +=1
    return torch.Tensor(raw_preds)


def evaluate(dataset_name, preds, test_Y, logits):
    eval_func = {
        const.WATERBIRDS_NAME: eval_wilds,
        const.CELEBA_NAME: eval_wilds,
        const.PACS_NAME: eval_domainbed,
        const.VLCS_NAME: eval_domainbed,
        const.CXR_NAME: eval_cxr,
    }
    if dataset_name not in [const.CXR_NAME, const.PACS_NAME, const.VLCS_NAME]:
        return eval_func[dataset_name](preds, test_Y)
    else:
        return eval_func[dataset_name](preds, test_Y, logits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run CLIP zero shot')
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-clip', '--clip_model', type=str, default='openclip_vitl14')
    parser.add_argument('-lm', '--llm', type=str, default='chatgpt')
    parser.add_argument('--device', type=int, nargs='+', default=[])
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--template_index', type=int, default=0)
    parser.add_argument('--algorithm', type=str, default='base')
    args = parser.parse_args()

    if len(args.device) > 0:
        print(args.device)
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            device_str = ",".join(map(str, args.device))
            os.environ["CUDA_VISIBLE_DEVICES"] = device_str
            device_count = torch.cuda.device_count()
            if len(args.device) > device_count:
                raise ValueError(f"Specified {len(args.device)} devices, but only {device_count} devices found.")
    args.use_data_parallel = len(args.device) > 1



    
    dataset_name = args.dataset
    clip_model = args.clip_model
    llm_model = args.llm

    assert clip_model in const.SUPPORTED_CLIP
    assert llm_model in const.SUPPORTED_LM

    labels = text_prompts[dataset_name]['labels_pure'] # get pure labels instead of labels inserted into predefined templates
    max_tokens = 100
    n_paraphrases = 0
    
    labels_text = MultiEnvDataset().dataset_dict[dataset_name]().get_labels() #e.g., person with dark hair, a landbird
    load_dir = os.path.join(sys_const.DATA_DIR, f'features/{dataset_name}/{clip_model}')

    cached_features = False
    if os.path.isdir(load_dir):
        if len(os.listdir(load_dir)):
            cached_features = True

    if not cached_features:
        print(f"extracting CLIP features...")
        if clip_model in const.HF_CLIP:
            hf_extractor(dataset_name, clip_model, sys_const.DATA_DIR, args)
        elif clip_model in const.OPEN_CLIP:
            openclip_extractor(dataset_name, clip_model, sys_const.DATA_DIR, args)
    
    print(f'CLIP MODEL = {clip_model}')
    test_X = np.load(os.path.join(load_dir, 'image_emb.npy'))
    test_Y = np.load(os.path.join(load_dir, 'y.npy'))


    
    dataloader = get_embed_loader(test_X, test_Y)

    if args.algorithm == 'base':
        algorithm = ZeroShotClassifier(labels, clip_model, args)
        logit_all, pred_all, ent_all = algorithm.predict(dataloader)
        avg_ent = ent_all.mean()
        result_str = evaluate(dataset_name, pred_all, test_Y, logit_all)
        eles = result_str.strip().split('\n')
        with open(f"base_{args.dataset}_{args.clip_model}_templates.txt", "a") as f:
            f.write(f"{algorithm.template} {eles[0]} {eles[-1]} ent: {avg_ent:.6f}\n")
    elif args.algorithm == "base_ensemble":
        algorithm =  ZeroShotClassifierEnsemble(labels, clip_model, args)
        logit_all, pred_all = algorithm.predict(dataloader)
        result_str = evaluate(dataset_name, pred_all, test_Y, logit_all)
    elif args.algorithm == "save":
        algorithm = DiffPrompts(labels, clip_model, args)
        logit_all, pred_all = algorithm.predict(dataloader)
        result_str = evaluate(dataset_name, pred_all, test_Y, logit_all)
        eles = result_str.strip().split('\n')
        with open(f"ours_{args.dataset}_models.txt", "a") as f:
            f.write(f"K = {args.K} | {labels[0]}, {labels[1]} | {args.clip_model} {eles[0]} {eles[-1]}\n")
    