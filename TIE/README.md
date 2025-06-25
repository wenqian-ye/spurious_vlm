## Installation

### Requirements
- [open_clip](https://github.com/mlfoundations/open_clip)
- requirements (in requirements.txt)

### Installing
1. Install required packages: `pip install -r requirements.txt`
2. Install open_clip_torch: `pip install open_clip_torch`

## Dataset Preparation

1. **CelebA Dataset**
    - Download from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    - Place in the directory above the repository root

2. **Waterbirds Dataset**
    - Download from: https://github.com/p-lambda/wilds
    - Place in the directory above the repository root

3. **ISIC Dataset**
    - Run the provided code snippet to download and extract
```
import os
import gdown
import zipfile

data_root = '..'  # Set your ROOT directory
os.makedirs(data_root, exist_ok=True)
output = 'isic.zip'
url = 'https://drive.google.com/uc?id=1Os34EapIAJM34DrwZMw2rRRJij3HAUDV'

if not os.path.exists(os.path.join(data_root, 'isic')):
    gdown.download(url, os.path.join(data_root, output), quiet=False)
    with zipfile.ZipFile(os.path.join(data_root, output), 'r') as zip_ref:
        zip_ref.extractall(data_root)
```

4. **COVID-19 Dataset**
    - Download from: https://github.com/ieee8023/covid-chestxray-dataset

  
5. **FMOW Dataset**
   - FMOW can be downloaded by [Wilds](https://github.com/p-lambda/wilds/) 

## Reproducing Experiments
### Getting Started

To run **TIE**, we provide a Jupyter Notebook (`.ipynb`) file that contains all historical data for reference. Follow these steps:

### Running **TIE**
Ensure you have an active Conda environment before proceeding.
To reproduce **TIE** results for different datasets, execute the corresponding Jupyter Notebook:

1. **Waterbirds**: Run `Table1-WB.ipynb`
2. **CelebA**: Run `Table2-CelebA.ipynb`
3. **ISIC**: Run `Table3-ISIC.ipynb`
4. **COVID-19**: Run `Table3-Covid.ipynb`

### Running **TIE***  
To enable **TIE***, modify the following line in the code:

```
a = True
```
Change it to
```
a = False
 ```
This will automatically prevent the use of the spurious label, leverage zero-shot capability to infer the spurious label, and change the model to **TIE***.

### Changing the CLIP Model
If you want to use a different CLIP model, update the following lines in the code: 

Current (Default) CLIP Model – ViT-L/14

``` 
model,_, preprocess =  open_clip.create_model_and_transforms("ViT-L-14", pretrained='laion2b_s32b_b82k') #ViTL/14
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-L-14')
```
    
**Switching to CLIP ViT-B/32**

To use the ViT-B/32 model instead, comment out the above lines and uncomment the following:

```
model,_, preprocess =  open_clip.create_model_and_transforms("ViT-B/32", pretrained='openai') #ViTB/32
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
```
    
Both **TIE** and **TIE*** support this modification.



## Acknowledgment
We sincerely thank the contributors of open-source repositories that have supported this project, especially:

[DISC](https://github.com/Wuyxin/DISC) 

[OpenCLIP](https://github.com/mlfoundations/open_clip) 

[Wilds](https://github.com/p-lambda/wilds/)

[CLIP](https://github.com/openai/CLIP)



