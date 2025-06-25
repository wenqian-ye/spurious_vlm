# Self-Adaptive Prompt Exploration


### Downloading datasets: ###
- WILDS datasets (Waterbirds, CelebA): The code enables automatic download of WILDS datasets (thanks to the amazing [WILDS benchmark package](https://wilds.stanford.edu/)!). No extra steps needed here!
- DomainBed datasets (PACS, VLCS): Download the datasets from [DomainBed suit](https://github.com/facebookresearch/DomainBed)



### Configure 
1. Put in the `absolute` path of to download your datasets in `utils/sys_const.py` under the `DATA_DIR` constant.


### Running the code
```bash
python save_main.py -d=waterbirds -clip openclip_vitl14 --algorithm save --K 1
```
Flags:
- `-d`: select dataset (waterbirds/celebA/pacs/vlcs)
- `-clip`: select CLIP model (align/alt/openclip_vitl14/openclip_vitb32/openclip_vith14)
- `--K`: set the number of prompts used for each class. Set `K=80` to run SAVE-All
