<div align="center">

<h1>KEEP: Kalman-Inspired Feature Propagation for Video Face Super-Resolution</h1>
<div>
    <a href='https://jnjaby.github.io/' target='_blank'>Ruicheng Feng</a>&emsp;
    <a href='https://li-chongyi.github.io/' target='_blank'>Chongyi Li</a>&emsp;
    <a href='https://www.mmlab-ntu.com/person/ccloy/' target='_blank'>Chen Change Loy</a>
</div>
<div>
    S-Lab, Nanyang Technological University&emsp; 
</div>

<div>
    <strong>ECCV 2024</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2408.05205" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-KEEP-b31b1b.svg">
        </a>
        <a href="https://jnjaby.github.io/projects/KEEP/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://www.youtube.com/watch?v=Qr0cseESPqM/" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        <a href="https://github.com/jnjaby/KEEP/"><img src="https://img.shields.io/github/stars/jnjaby/KEEP">
        </a>
        </a>
    </h4>
</div>


<p align="center">
  <img src="./assets/images/KEEP_showcase.gif" alt="showcase">
  <br>
  🔥 For more results, visit our <a href="https://jnjaby.github.io/projects/KEEP/"><strong>project page</strong></a> 🔥
  <br>
  ⭐ If you found this project helpful to your projects, please help star this repo. Thanks! 🤗
</p>

</div>


# Update
- **2024.08**: We released the initial version of the inference code and models. Stay tuned for continuous updates!
- **2024.07**: This repo is created!


# Getting Started

## Dependencies and Installation

- Pytorch >= 1.7.1
- CUDA >= 10.1
- Other required packages in `requirements.txt`
```
# git clone this repository. Don't forget to add --recursive!!
git clone --recursive https://github.com/jnjaby/KEEP
cd KEEP

# create new anaconda env
conda create -n keep python=3.8 -y
conda activate keep

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib # only for face detection or cropping with dlib
conda install -c conda-forge ffmpeg
```

[Optional] If you forget to clone the repo with `--recursive`, you can update the submodule by 
```
git submodule init
git submodule update
```

## Quick Inference

### Download Pre-trained Models
All pretrained models can also be automatically downloaded during the first inference.
You can also download our pretrained models from [Releases V0.1.0](https://github.com/jnjaby/KEEP/releases/tag/v0.1.0) to the `weights` folder.


### Prepare Testing Data
We provide both synthetic (VFHQ) and real (collected) examples in `assets/examples` folder. If you would like to test your own face videos, place them in the same folder.
You can also download the full synthetic and real test data from [[Google Drive](https://drive.google.com/drive/folders/16yqGKQnjCzrdVK_SQSzFhULEfhSxMUH_?usp=sharing)].



### Inference
**[Note]** If you want to compare KEEP in your paper, please make sure the face alignment is consistent and run the following command with `--has_aligned` to indicate faces are already cropped and aligned. The results will be saved in the `results` folder.


🧑🏻 Video Face Restoration for synthetic data (cropped and aligned face)
```
# For cropped and aligned faces
python inference_keep.py -i=./assets/examples/synthetic_1.mp4 -o=results/ --has_aligned --save_video -s=1
```

🎬 Video Face Restoration for real data (in the wild)
```
# For whole video
# Add '--bg_upsampler realesrgan' to enhance the background regions with Real-ESRGAN
# Add '--face_upsample' to further upsample restorated face with Real-ESRGAN
# Add '--draw_box' to show the bounding box of detected faces.
python inference_keep.py -i=./assets/examples/real_1.mp4 -o=results/ --draw_box --save_video -s=1 --bg_upsampler=realesrgan
```


## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
@InProceedings{feng2024keep,
      title     = {Kalman-Inspired FEaturE Propagation for Video Face Super-Resolution},
      author    = {Feng, Ruicheng and Li, Chongyi and Loy, Chen Change},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year      = {2024}
}
   ```


## License and Acknowledgement

This project is open sourced under [NTU S-Lab License 1.0](https://github.com/jnjaby/KEEP/blob/main/LICENSE). Redistribution and use should follow this license.
The code framework is mainly modified from [CodeFormer](https://github.com/sczhou/CodeFormer/). Please refer to the original repo for more usage and documents.


## Contact

If you have any question, please feel free to contact us via `ruicheng002@ntu.edu.sg`.
