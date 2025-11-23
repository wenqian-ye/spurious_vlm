# SAGE: Spuriousness-Aware Guided Prompt Exploration
[Paper PDF](https://arxiv.org/pdf/2511.13005)

### Downloading datasets: ###
- WILDS datasets (Waterbirds, CelebA): The code enables automatic download of WILDS datasets (thanks to the amazing [WILDS benchmark package](https://wilds.stanford.edu/)!). No extra steps needed here!
- DomainBed datasets (PACS, VLCS): Download the datasets from [DomainBed suit](https://github.com/facebookresearch/DomainBed)



### Configure 
1. Put in the `absolute` path of to download your datasets in `utils/sys_const.py` under the `DATA_DIR` constant.


### Running the code
```bash
python save_main.py -d=waterbirds -clip openclip_vitl14 --algorithm sage 
```
Flags:
- `-d`: select dataset (waterbirds/celebA/pacs/vlcs)
- `-clip`: select CLIP model (align/alt/openclip_vitl14/openclip_vitb32/openclip_vith14)

### Acknowledgement
This codebase is built upon the [RoboShot](https://github.com/SprocketLab/roboshot) codebase. We thank the original authors for making it publicly available.

If you find this work useful, please cite:

```bibtex
@inproceedings{ye2026sage,
  title={SAGE: Spuriousness-Aware Guided Prompt Exploration},
  author={Ye, Wenqian and Wang, Di and Zheng, Guangtao and Liu, Bohan and Zhang, Aidong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```