    
from libs.get_clip_text_emb import get_text_embedding, construct_text_encoder
from utils import templates as TEMPLATES
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import random_split, DataLoader

import open_clip
from torch.nn import CosineSimilarity


class EmbedDataset(Dataset):
    def __init__(self, embeds, labels):
        super(EmbedDataset, self).__init__()
        self.embeds = embeds
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        embed = self.embeds[index]
        label = self.labels[index]
        return embed, label

def get_embed_loader(embeds, labels, batch_size=128):
    dataset = EmbedDataset(embeds, labels)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=8)
    return dataloader

class ZeroShotClassifier:
    def __init__(self, classnames, clip_model, args):
        text_encoder = construct_text_encoder(clip_model, args)
        template_index = args.template_index
        template = TEMPLATES.imagenet_templates[template_index]
        texts = []
        for classname in classnames:
            texts.append(template.format(classname))
        self.text_embeds = torch.tensor(get_text_embedding(texts, text_encoder))
        self.template = template

    def process_batch(self, input_features):
        input_features /= input_features.norm(dim=-1, keepdim=True)
        text_embeds = self.text_embeds / self.text_embeds.norm(dim=-1, keepdim=True)

        logits = input_features @ text_embeds.T
        text_probs = logits.softmax(dim=-1)
        ents = (-text_probs * F.log_softmax(logits,dim=-1)).sum(dim=-1)
        return text_probs, ents
    
    def predict(self, dataloader):
        logit_all = []
        pred_all = []
        ent_all = []
        for x, y in tqdm(dataloader):
            logits, ents = self.process_batch(x)
            preds = torch.argmax(logits, dim=-1)
            logit_all.append(logits)
            pred_all.append(preds)
            ent_all.append(ents)
        logit_all = torch.cat(logit_all).detach().cpu().numpy()
        pred_all = torch.cat(pred_all).detach().cpu().numpy() 
        ent_all = torch.cat(ent_all).detach().cpu().numpy() 
        return logit_all, pred_all, ent_all
    

class ZeroShotClassifierEnsemble:
    def __init__(self, classnames, clip_model, args):
        text_encoder = construct_text_encoder(clip_model, args)
        self.classnames = classnames
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in TEMPLATES.imagenet_templates] #format with class
            text_embeds = get_text_embedding(texts, text_encoder)
            zeroshot_weights.append(torch.tensor(text_embeds)) # (N, D)
        self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1) # (N, C, D)

    def process_batch(self, input_features):
        feas = input_features / torch.norm(input_features, dim=-1, keepdim=True) # (B,D)
        # text_embeds = self.zeroshot_weights / self.zeroshot_weights.norm(dim=-1, keepdim=True) # (N, C, D)
        # B, D = feas.shape
        # logits = torch.matmul(feas.view(B, 1, 1, D), text_embeds.permute(0,2,1).unsqueeze(0)).squeeze(2) # (B, 1, 1, D) x (1, N, D, C) -> (B, N, 1, C) -> (B, N, C)
        # logits = logits.mean(dim=1) # (B, C)

        text_embeds = self.zeroshot_weights.mean(dim=0) # (C, D)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)
        logits = torch.matmul(feas, text_embeds.T) 
        return logits
    
    def predict(self, dataloader):
        logit_all = []
        pred_all = []
        for x, y in tqdm(dataloader):
            logits = self.process_batch(x)
            preds = torch.argmax(logits, dim=-1)
            logit_all.append(logits)
            pred_all.append(preds)
        logit_all = torch.cat(logit_all).detach().cpu().numpy()
        pred_all = torch.cat(pred_all).detach().cpu().numpy() 
        return logit_all, pred_all
    
class TransductivePrompts:
    def __init__(self, classnames, clip_model, args):
        text_encoder = construct_text_encoder(clip_model, args)
        self.classnames = classnames
        zeroshot_weights = {}
        for classname in classnames:
            texts = [template.format(classname) for template in TEMPLATES.imagenet_templates] #format with class
            text_embeds = get_text_embedding(texts, text_encoder)
            zeroshot_weights[classname] = torch.tensor(text_embeds)
        self.zeroshot_weights = zeroshot_weights

    def process_batch(self, input_features):
        sel_weights = []
        input_features = input_features / torch.norm(input_features, dim=-1, keepdim=True)
        for i, classname in enumerate(self.classnames):
            cls_weights = self.zeroshot_weights[classname].unsqueeze(0)
            sims = torch.matmul(input_features.unsqueeze(1), cls_weights.permute(0, 2, 1)).squeeze(1)
            
            min_sims, min_indices = torch.min(sims, dim=1)
            
            prompt_counts = torch.zeros(sims.shape[1], device=sims.device)
            for idx in min_indices:
                prompt_counts[idx] += 1
                
            worst_prompt_threshold = max(1, int(0.3 * len(min_indices)))
            worst_prompt_mask = prompt_counts >= worst_prompt_threshold
            
            if worst_prompt_mask.sum() > 0:
                worst_indices = torch.where(worst_prompt_mask)[0]
                worst_embeddings = torch.index_select(cls_weights.squeeze(0), 0, worst_indices)
                
                worst_sims = torch.index_select(sims, 1, worst_indices)
                
                prompt_variance = torch.var(worst_sims, dim=0)
                _, top_var_indices = torch.topk(prompt_variance, min(3, len(prompt_variance)))
                
                selected_worst = torch.index_select(worst_embeddings, 0, top_var_indices)
                class_embedding = torch.mean(selected_worst, dim=0)
            else:
                prompt_variance = torch.var(sims, dim=0)
                _, var_indices = torch.topk(prompt_variance, min(5, len(prompt_variance)))
                discriminative_embeddings = torch.index_select(cls_weights.squeeze(0), 0, var_indices)
                class_embedding = torch.mean(discriminative_embeddings, dim=0)
            
            class_embedding = class_embedding / class_embedding.norm()
            
            batch_weights = []
            for b in range(input_features.shape[0]):
                input_vec = input_features[b]
                
                sample_sims = sims[b]
                sample_mean = torch.mean(sample_sims)
                sample_std = torch.std(sample_sims)
                
                is_potential_worst = sample_mean < (torch.mean(sims) - 0.5 * torch.std(torch.mean(sims, dim=1)))
                
                embedding = class_embedding.clone()
                
                projection = (embedding * input_vec).sum() * embedding
                residual = input_vec - projection
                
                if is_potential_worst and residual.norm() > 0.2:
                    for _ in range(3):
                        spurious_dir = residual / (residual.norm() + 1e-8)
                        
                        removal_strength = 1.0
                        if sample_mean < torch.mean(sims) - torch.std(torch.mean(sims, dim=1)):
                            removal_strength = 1.5
                            
                        proj_amount = (embedding * spurious_dir).sum() * removal_strength
                        embedding = embedding - proj_amount * spurious_dir
                        embedding = embedding / (embedding.norm() + 1e-8)
                        
                        projection = (embedding * input_vec).sum() * embedding
                        residual = input_vec - projection
                        
                        if residual.norm() < 0.15:
                            break
                
                batch_weights.append(embedding)
            
            batch_weights = torch.stack(batch_weights)
            sel_weights.append(batch_weights)
        
        sel_weights = torch.stack(sel_weights, dim=2)
        
        raw_logits = torch.matmul(input_features.unsqueeze(1), sel_weights).squeeze(1)
        
        confidences = F.softmax(raw_logits, dim=1).max(dim=1)[0]
        
        adaptive_temp = torch.ones_like(confidences)
        low_conf_mask = confidences < torch.median(confidences)
        adaptive_temp[low_conf_mask] = 0.8
        
        logits = raw_logits / adaptive_temp.unsqueeze(1)
        
        return logits

    def predict(self, dataloader):
        logit_all = []
        pred_all = []
        for x, y in tqdm(dataloader):
            logits = self.process_batch(x)
            preds = torch.argmax(logits, dim=-1)
            logit_all.append(logits)
            pred_all.append(preds)
        logit_all = torch.cat(logit_all).detach().cpu().numpy()
        pred_all = torch.cat(pred_all).detach().cpu().numpy() 
        return logit_all, pred_all
    

class SimpleWorstPrompts:
    def __init__(self, classnames, clip_model, args):
        text_encoder = construct_text_encoder(clip_model, args)
        self.classnames = classnames
        zeroshot_weights = {}
        for classname in classnames:
            texts = [template.format(classname) for template in TEMPLATES.imagenet_templates] #format with class
            text_embeds = get_text_embedding(texts, text_encoder)
            zeroshot_weights[classname] = torch.tensor(text_embeds)
        self.zeroshot_weights = zeroshot_weights

    def process_batch(self, input_features):
        sel_weights = []
        input_features = input_features / torch.norm(input_features, dim=-1, keepdim=True)
        for i, classname in enumerate(self.classnames):
            cls_weights = self.zeroshot_weights[classname].unsqueeze(0)
            num_dims = cls_weights.shape[2]
            logits = torch.matmul(input_features.unsqueeze(1), cls_weights.permute(0,2,1)).squeeze(1) # (B, 1, D) x (1, D, N) -> (B, 1, N)
            indexes = torch.argmin(logits, dim=1) # B
            weights = torch.gather(cls_weights.squeeze(0), 0, indexes.unsqueeze(1).expand(len(indexes), num_dims))
            sel_weights.append(weights)
        sel_weights = torch.stack(sel_weights, dim=2) # B, D, C
       
        logits = torch.matmul(input_features.unsqueeze(1), sel_weights).squeeze(1) # (B,1,D) x (B,D,C) -> (B, 1, C)
        return logits
    
    def predict(self, dataloader):
        logit_all = []
        pred_all = []
        for x, y in tqdm(dataloader):
            logits = self.process_batch(x)
            preds = torch.argmax(logits, dim=-1)
            logit_all.append(logits)
            pred_all.append(preds)
        logit_all = torch.cat(logit_all).detach().cpu().numpy()
        pred_all = torch.cat(pred_all).detach().cpu().numpy() 
        return logit_all, pred_all
    

class DiffPrompts:
    def __init__(self, classnames, clip_model, args):
        text_encoder = construct_text_encoder(clip_model, args)
        self.classnames = classnames
        self.templates = TEMPLATES.imagenet_templates
     
        zeroshot_weights = []
        for c, classname in enumerate(classnames):
            texts = [template.format(classname) for template in self.templates] #format with class
            text_embeds = get_text_embedding(texts, text_encoder)
            zeroshot_weights.append(torch.tensor(text_embeds))
        self.zeroshot_weights = zeroshot_weights
        self.avg_weights = torch.cat(zeroshot_weights).mean(dim=0,keepdim=True)
        self.avg_weights = self.avg_weights / torch.norm(self.avg_weights, dim=1,keepdim=True)
        self.K = args.K
    def process_batch(self, input_features):
        input_features = input_features / torch.norm(input_features, dim=-1, keepdim=True) 
        sel_weights = []
        B, D = input_features.shape
        K = self.K
        all_weights = torch.stack([self.zeroshot_weights[c] for c in range(len(self.classnames))], dim=1) # (N, C, D)
        sims = torch.matmul(input_features.view(B,1,1,D), all_weights.permute(0,2,1).unsqueeze(0)).squeeze(2) # (B, 1, 1, D) x (1, N, D, C) -> (B, N, C)
        sel_indexes = torch.stack([torch.argsort(sims[:,:,c],dim=1)[:,0:K] for c in range(len(self.classnames))], dim=1) # (B, C, K)
        all_logits = torch.cat([torch.gather(sims, 1, sel_indexes[:,c].unsqueeze(2).expand(B,K,len(self.classnames))) for c in range(len(self.classnames))],dim=1)
        logits = all_logits.mean(dim=1)
        return logits
    
    def predict(self, dataloader):
        logit_all = []
        pred_all = []
        for x, y in tqdm(dataloader):
            logits = self.process_batch(x)
            preds = torch.argmax(logits, dim=-1)
            logit_all.append(logits)
            pred_all.append(preds)
        logit_all = torch.cat(logit_all).detach().cpu().numpy()
        pred_all = torch.cat(pred_all).detach().cpu().numpy() 
        return logit_all, pred_all

class RandomClassifier:
    def __init__(self, classnames, clip_model, args):
        self.classnames = classnames
        self.templates = TEMPLATES.imagenet_templates
        self.text_encoder = construct_text_encoder(clip_model, args)
        self.K = args.K  # you can tune this value
        self.counts = None

    def process_batch(self, image_features):
        image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
        B, D = image_features.shape
        T = len(self.templates)
        C = len(self.classnames)

        all_prompts = [t.format(c) for t in self.templates for c in self.classnames] #T*C
        all_text_embeddings = torch.tensor(get_text_embedding(all_prompts, self.text_encoder))
        text_embeds = all_text_embeddings.view(T,C,D)

        topk_indices = torch.randint(low=0, high=80, size=(B, self.K))  # [B, K]
        
        text_embeds = text_embeds.permute(1, 0, 2)  # [C, T, D]
        final_text_embeddings = torch.zeros(B, C, self.K, D)
        for b in range(B):
            indices = topk_indices[b] # [K]
            final_text_embeddings[b] = text_embeds[:, indices, :]  # [K, D]

        text_embeds = final_text_embeddings.mean(dim=2)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=2, keepdim=True)
        image_features = image_features.unsqueeze(1) 
        logits = (image_features * text_embeds).sum(dim=-1)

        return logits


    def predict(self, dataloader):
        logit_all = []
        pred_all = []
        for x, y in tqdm(dataloader):
            logits = self.process_batch(x)
            preds = torch.argmax(logits, dim=-1)
            logit_all.append(logits)
            pred_all.append(preds)
        logit_all = torch.cat(logit_all).detach().cpu().numpy()
        pred_all = torch.cat(pred_all).detach().cpu().numpy()
        return logit_all, pred_all

class SAGE:
    def __init__(self, classnames, clip_model, args):
        self.classnames = classnames
        self.templates = TEMPLATES.imagenet_templates
        self.text_encoder = construct_text_encoder(clip_model, args)
        self.K = args.K  # you can tune this value
        self.counts = None

    def process_batch(self, image_features):
        image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
        B, D = image_features.shape
        T = len(self.templates)
        C = len(self.classnames)

        all_prompts = [t.format(c) for t in self.templates for c in self.classnames] #T*C
        all_text_embeddings = torch.tensor(get_text_embedding(all_prompts, self.text_encoder))
        text_embeds = all_text_embeddings.view(T,C,D)

        image_features_exp = image_features.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        text_embeds_exp = text_embeds.unsqueeze(0)  # [1, T, C, D]
        sim = F.cosine_similarity(image_features_exp, text_embeds_exp, dim=-1)  # [B, T, C]

        diff = sim.max(dim=2).values - sim.min(dim=2).values  # [B, T]
        topk_diff, topk_indices = torch.topk(diff, k=self.K, dim=1)
        
        text_embeds = text_embeds.permute(1, 0, 2)  # [C, T, D]
        final_text_embeddings = torch.zeros(B, C, self.K, D)
        for b in range(B):
            indices = topk_indices[b] # [K]
            final_text_embeddings[b] = text_embeds[:, indices, :]  # [K, D]

        text_embeds = final_text_embeddings.mean(dim=2)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=2, keepdim=True)
        image_features = image_features.unsqueeze(1) 
        logits = (image_features * text_embeds).sum(dim=-1)

        return logits


    def predict(self, dataloader):
        logit_all = []
        pred_all = []
        for x, y in tqdm(dataloader):
            logits = self.process_batch(x)
            preds = torch.argmax(logits, dim=-1)
            logit_all.append(logits)
            pred_all.append(preds)
        logit_all = torch.cat(logit_all).detach().cpu().numpy()
        pred_all = torch.cat(pred_all).detach().cpu().numpy()
        return logit_all, pred_all