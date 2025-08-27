# UniDepth + YOLO

This repository builds on the work of UniDepth. It is not affiliated with ETH Zurich. Please consider their original [repo](https://github.com/lpiccinelli-eth/UniDepth) and [paper](https://arxiv.org/abs/2502.20110) for more information. 

## Installation

Requirements are not in principle hard requirements, but there might be some differences (not tested):
- Linux
- Python 3.10+ 
- CUDA 11.8+

The following should work on both SSH and Jetson.

Install the environment needed to run UniDepth with:
```shell
export VENV_DIR=<YOUR-VENVS-DIR>
export NAME=Unidepth

python -m venv $VENV_DIR/$NAME
source $VENV_DIR/$NAME/bin/activate
```
## Install UniDepth and dependencies, cuda >11.8 work fine, too.
```shell
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

*Note*: Make sure that your compilation CUDA version and runtime CUDA version match.  
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

*Note*: xFormers may raise the the Runtime "error": `Triton Error [CUDA]: device kernel image is invalid`.  
This is related to xFormers mismatching system-wide CUDA and CUDA shipped with torch.  
It may considerably slow down inference.

Run UniDepth on the given assets to test your installation (you can check this script as guideline for further usage):
```shell
python ./scripts/demo.py
```
If everything runs correctly, `demo.py` should print: `ARel: 7.45%`.

If you encounter `Segmentation Fault` after running the demo, you may need to uninstall torch via pip (`pip uninstall torch`) and install the torch version present in [requirements](requirements.txt) with `conda`.

## Depth Estimation + Object Detection

Install Ultralytics into the environment
```shell
pip install ultralytics
```
- Scripts
  - [depth_jetson.py](scripts/depth_jetson.py) is optimzed for the Jetson
  - [depth.py](scripts/depth.py) gives the best results, try on SSH but not recommended for Jetson

- YOLO Weights
  - A fine-tuned YOLO model is employed- [yolo11n-uav-vehicle-bbox.pt](yolo_models/yolo11n-uav-vehicle-bbox.pt)
  - Optional: employ out of the box models such as [yolov8m.pt](yolo_models/yolov8m.pt) or [yolov8n.pt](yolo_models/yolov8n.pt)

- Inference

```shell
python scripts/depth_jetson.py
```
or similarly

```shell
python scripts/depth_jetson.py --video videos/video_name.mp4 --fps desired fps --conf desired confidence 
```
The annotated video will save in the output folder. If running on an SSH, you will need to download the video onto your local machine to play it.

## Citation (All from original README)

If you find our work useful in your research please consider citing our publications:
```bibtex
@inproceedings{piccinelli2024unidepth,
    title     = {{U}ni{D}epth: Universal Monocular Metric Depth Estimation},
    author    = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```

```bibtex
@misc{piccinelli2025unidepthv2,
      title={{U}ni{D}epth{V2}: Universal Monocular Metric Depth Estimation Made Simpler}, 
      author={Luigi Piccinelli and Christos Sakaridis and Yung-Hsu Yang and Mattia Segu and Siyuan Li and Wim Abbeloos and Luc Van Gool},
      year={2025},
      eprint={2502.20110},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20110}, 
}
```

## License

This software is released under Creatives Common BY-NC 4.0 license. You can view a license summary [here](LICENSE).


## Acknowledgement

We would like to express our gratitude to [@niels](https://huggingface.co/nielsr) for helping integrating UniDepth in HuggingFace.

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Automated Cars Europe).
