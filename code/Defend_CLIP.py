import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import STL10, CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from copy import deepcopy as dcopy
import logging
import time
# from utils import one_hot_embedding
# from models.model import *
from utils import *
from attacks import *
import torch.nn.functional as F
import functools
from autoattack import AutoAttack
from replace.datasets import caltech, country211, dtd,eurosat, fgvc_aircraft, food101, \
                             flowers102, oxford_iiit_pet, pcam, stanford_cars, sun397
from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from func import clip_img_preprocessing, multiGPU_CLIP
import random
from replace.datasets.folder import ImageNetFolder
from torchvision.datasets import ImageFolder
tinyimagenet_root = "./data/tiny-imagenet-200"
attacks_to_run=['apgd-ce', 'apgd-dlr']

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()



seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


test_stepsize = 1/255
test_eps = 1/255
test_numsteps = 10

device = 'cuda'
model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)

for p in model.parameters():
    p.requires_grad = False
convert_models_to_fp32(model)


model = torch.nn.DataParallel(model)
model.eval()
prompter = NullPrompter()
add_prompter = TokenPrompter(0)
prompter = torch.nn.DataParallel(prompter).cuda()
add_prompter = torch.nn.DataParallel(add_prompter).cuda()

# criterion to compute attack loss, the reduction of 'sum' is important for effective attacks
criterion_attack = torch.nn.CrossEntropyLoss(reduction='sum').to(device)



def threshold_defense_clean(images, target, texts, dataset_name, num_trials, sigma_lst = [0.02,0.05]):
    with autocast():
        text_tokens = clip.tokenize(texts).to(device)

        with torch.no_grad():
            clean_output,_,f_source,text_features = multiGPU_CLIP(
                        None,None,None, model, prompter(clip_img_preprocessing(images)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name 
                    )
            
            zs_clean_acc = accuracy(clean_output, target, topk=(1,))[0].item()

            sigma_event_lst = []
            l2_drift_lst = [] 

            for sigma in sigma_lst:
                f_anchor_trials = []
                for _ in range(num_trials):
                    mixed_images = images + sigma * torch.randn_like(images)

                    with torch.no_grad():
                        _,_,f_anchor,_ = multiGPU_CLIP(
                                None,None,None, model, prompter(clip_img_preprocessing(mixed_images)),
                                text_tokens, prompt_token=None, dataset_name=dataset_name 
                            )


                    f_anchor_trials.append(f_anchor)
                    
                f_anchor_trials = torch.stack(f_anchor_trials) # [10, bs, dim]
                f_anchors = (torch.sum(f_anchor_trials, dim=0)) / num_trials 
                l2_drift_anchors = torch.norm(f_anchors - f_source, dim=-1) # [bs]
                l2_drift_lst.append(l2_drift_anchors)

                
            delta_drift = (l2_drift_lst[1] - l2_drift_lst[0]) / l2_drift_lst[0]
            
            
            # wherever clean put 1, otherwise 0
            clean_score = torch.where((delta_drift < 0.35), 1.0, 0.0)

            adaptive_sigma = torch.where((clean_score == 1.0), 0.02, 0.1)

            adaptive_alpha = torch.where(clean_score == 1.0, 0.0, 1.2)

            # Defense

            # Normalize shapes for broadcasting
            adaptive_sigma = adaptive_sigma.view(-1, 1, 1, 1)
            adaptive_alpha = adaptive_alpha.view(-1, 1)

            f_anchor_trials = []

            for _ in range(num_trials):
                noise = adaptive_sigma * torch.randn_like(images)
                noisy_images = images + noise

                with torch.no_grad():
                    _, _, f_anchor, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(noisy_images)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name
                    )
                f_anchor_trials.append(f_anchor)

            f_anchor_trials = torch.stack(f_anchor_trials) # [10, bs, dim]

            f_anchor_mean = f_anchor_trials.mean(dim=0)  # [bs, dim]

            f_final = adaptive_alpha * f_anchor_mean + (1 - adaptive_alpha) * f_source
            f_final_nrom = f_final / f_final.norm(dim=-1, keepdim=True)

            final_logits = f_final_nrom @ text_features.t() * model.module.logit_scale.exp()
            clean_acc = accuracy(final_logits, target, topk=(1,))
            clean_accuracy = clean_acc[0].item()

        return clean_accuracy, zs_clean_acc


def threshold_defense_adv(images, target, texts, dataset_name, num_trials, sigma_lst = [0.02,0.05]):
    with autocast():
        text_tokens = clip.tokenize(texts).to(device)
        
        torch.cuda.empty_cache()
        if attack_type == 'pgd':
            delta_prompt = attack_pgd(None, prompter, model, None, None, add_prompter, criterion_attack,
                                    images, target, test_stepsize, test_numsteps, 'l_inf',
                                    text_tokens=text_tokens, epsilon=test_eps, dataset_name=dataset_name) # for PGD attack
        elif attack_type == 'CW':
            delta_prompt = attack_CW(None, prompter, model, None, None, add_prompter, criterion_attack,
                                                 images, target, text_tokens,
                                                 test_stepsize, test_numsteps, 'l_inf', epsilon=test_eps) # for CW attack

        attacked_images = images + delta_prompt

        with torch.no_grad():
            adv_output,_,f_source,text_features = multiGPU_CLIP(
                        None,None,None, model, prompter(clip_img_preprocessing(attacked_images)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name 
                    )
            
            zs_robust_acc = accuracy(adv_output, target, topk=(1,))[0].item()

            sigma_event_lst = []
            l2_drift_lst = []

            for sigma in sigma_lst:
                f_anchor_trials = []
                for _ in range(num_trials):
                    mixed_images = attacked_images + sigma * torch.randn_like(attacked_images)

                    with torch.no_grad():
                        _,_,f_anchor,_ = multiGPU_CLIP(
                                None,None,None, model, prompter(clip_img_preprocessing(mixed_images)),
                                text_tokens, prompt_token=None, dataset_name=dataset_name 
                            )


                    f_anchor_trials.append(f_anchor)
                
                f_anchor_trials = torch.stack(f_anchor_trials) # [10, bs, dim]
                f_anchors = (torch.sum(f_anchor_trials, dim=0)) / num_trials 
                l2_drift_anchors = torch.norm(f_anchors - f_source, dim=-1) # [bs] 
                l2_drift_lst.append(l2_drift_anchors) 

                
            delta_drift = (l2_drift_lst[1] - l2_drift_lst[0]) / l2_drift_lst[0] 
            
            
            # wherever clean put 1, otherwise 0
            clean_score = torch.where((delta_drift < 0.35), 1.0, 0.0) 

            adaptive_sigma = torch.where((clean_score == 1.0), 0.02, 0.1) 

            adaptive_alpha = torch.where(clean_score == 1.0, 0.0, 1.2)

            # Defense

            # Normalize shapes for broadcasting
            adaptive_sigma = adaptive_sigma.view(-1, 1, 1, 1)
            adaptive_alpha = adaptive_alpha.view(-1, 1)

            f_anchor_trials = []

            for _ in range(num_trials):
                noise = adaptive_sigma * torch.randn_like(attacked_images)
                noisy_images = attacked_images + noise

                with torch.no_grad():
                    _, _, f_anchor, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(noisy_images)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name
                    )
                f_anchor_trials.append(f_anchor)

            f_anchor_trials = torch.stack(f_anchor_trials) # [10, bs, dim]

            f_anchor_mean = f_anchor_trials.mean(dim=0)  # [bs, dim]

            f_final = adaptive_alpha * f_anchor_mean + (1 - adaptive_alpha) * f_source
            f_final_nrom = f_final / f_final.norm(dim=-1, keepdim=True)

            final_logits = f_final_nrom @ text_features.t() * model.module.logit_scale.exp()
            adv_acc = accuracy(final_logits, target, topk=(1,))
            robust_accuracy = adv_acc[0].item()
            
        return robust_accuracy, zs_robust_acc




##############################################


def get_texts(val_dataset_list, val_dataset_name, template='This is a photo of a {}.'):
    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts
            # class_names = each.clip_prompts
        else:
            class_names = each.classes if hasattr(each, 'classes') else each.clip_categories
            # breakpoint()
            if val_dataset_name[cnt] in ['ImageNet', 'tinyImageNet']:
                refined_data = read_json(f"./support/{val_dataset_name[cnt].lower()}_refined_labels.json")
                clean_class_names = [refined_data[ssid]['clean_name'] for ssid in class_names]
                class_names = clean_class_names
                
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)
    return texts_list



preprocess224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
preprocess224_caltech = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor()
])

logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


dataset_names = ['dtd', 'cifar10', 'cifar100', 'STL10', 'flowers102', 'oxfordpet', 'Caltech101', 'Caltech256', 'fgvc_aircraft', 'StanfordCars', 'EuroSAT', 'tinyImageNet', 'Food101', 'PCAM', 'SUN397']

for name in dataset_names:
    dataset_name = [name]
    if name == 'STL10':
        dataset = STL10(root="./data", split="test", transform=preprocess224, download=True)
    elif name == 'cifar10':
        dataset = CIFAR10(root="./data", transform=preprocess224, download=True, train=False)
    elif name == 'cifar100':
        dataset = CIFAR100(root="./data", transform=preprocess224, download=True, train=False)
    elif name == 'dtd':
        dataset = dtd.DTD(root="./data", split='test', transform=preprocess224, download=True)
    elif name == 'oxfordpet':
        dataset = oxford_iiit_pet.OxfordIIITPet(root="./data", split='test', transform=preprocess224, download=True)
    elif name == 'flowers102':
        dataset = flowers102.Flowers102(root="./data", split='test', transform=preprocess224, download=True)
    elif name == 'fgvc_aircraft':
        dataset = fgvc_aircraft.FGVCAircraft(root="./data", split='test', transform=preprocess224, download=True)
    elif name == 'Caltech101':
        dataset = caltech.Caltech101(root="./data", target_type='category', transform=preprocess224_caltech, download=True)
    elif name == 'Caltech256':
        dataset = caltech.Caltech256(root="./data", transform=preprocess224_caltech, download=True)
    elif name == 'Food101':
        dataset = food101.Food101(root="./data", split='test', transform=preprocess224, download=True)
    elif name == 'StanfordCars':
        dataset = stanford_cars.StanfordCars(root="./data", split='test', transform=preprocess224, download=True)
    elif name == 'PCAM':
        dataset = pcam.PCAM(root="./data", split='test', transform=preprocess224, download=True)
    elif name == 'Country211':
        dataset = country211.Country211(root="./data", split='test', transform=preprocess224, download=True)
    elif name == 'EuroSAT':
        dataset = eurosat.EuroSAT(root="./data", transform=preprocess224, download=True)
    elif name == 'SUN397':
        dataset = sun397.SUN397(root="./data", transform=preprocess224, download=True)
    elif name == 'tinyImageNet':
        dataset = ImageNetFolder(os.path.join(tinyimagenet_root, 'val_'), transform=preprocess224)
    elif name == 'ImageNet':
        dataset = ImageNetFolder("./data/imagenet-val", transform=preprocess224)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    texts = get_texts([dataset], dataset_name)[0]
    

    num_trials = 10 # number of noise samples
    total_robust_acc = 0
    total_clean_acc = 0
    total_zs_robust_acc = 0
    total_zs_clean_acc = 0
    count = 0
    
    attack_type = 'pgd'

    for idx, batch in enumerate(tqdm(dataloader)):

        images = batch[0].to(device)
        labels = batch[1].to(device)

        clean_acc, zs_clean_acc = threshold_defense_clean(images, labels, texts, dataset_name, num_trials)
        robust_acc, zs_robust_acc = threshold_defense_adv(images, labels, texts, dataset_name, num_trials)

        total_clean_acc += clean_acc * images.shape[0]
        total_robust_acc += robust_acc * images.shape[0]
        total_zs_clean_acc += zs_clean_acc * images.shape[0]
        total_zs_robust_acc += zs_robust_acc * images.shape[0]

        count += images.shape[0]

    logging.info(f"{name}: \n"
    f"Attack type = {attack_type} \n"
    f"Robust accuracy = {total_robust_acc / count:.4f} \n"
    f"Clean accuracy = {total_clean_acc / count:.4f} \n"
    f"Zero-shot clean accuracy = {total_zs_clean_acc / count:.4f} \n"
    f"Zero-shot robust accuracy = {total_zs_robust_acc / count:.4f}")

    