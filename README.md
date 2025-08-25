This repository build on the work of UniDepth from the following papers below. Please consider their original README for more information. https://github.com/lpiccinelli-eth/UniDepth. It provides scripts that combine YOLO with UniDepth for metric depth estimation of detected cars.

> [**UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler**](https://arxiv.org/abs/2502.20110),  
> Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, Luc Van Gool,  
> under submission,  
> *Paper at [arXiv 2502.20110](https://arxiv.org/abs/2502.20110.pdf)*  

> [**UniDepth: Universal Monocular Metric Depth Estimation**](https://arxiv.org/abs/2403.18913),  
> Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu,  
> CVPR 2024,  
> *Paper at [arXiv 2403.18913](https://arxiv.org/pdf/2403.18913.pdf)*  

## Installation

The following should work on both SSH and Jetson.

Requirements are not in principle hard requirements, but there might be some differences (not tested):
- Linux
- Python 3.10+ 
- CUDA 11.8+

Install the environment needed to run UniDepth with:
```shell
export VENV_DIR=<YOUR-VENVS-DIR>
export NAME=Unidepth

python -m venv $VENV_DIR/$NAME
source $VENV_DIR/$NAME/bin/activate
```
# Install UniDepth and dependencies, cuda >11.8 work fine, too.
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
#Need to put this here!
```

Have a look at the script folder:
depth_jetson.py and depth_jetson1.py are optimzed for the Jetson
depth.py gives the best results but extremely slow on Jetson

Focusing on depth_jetson1.py:

The script is set at a YOLO confidence of 0.25, an output FPS of 1 and to run inference on the waterloo video.

To run it: 

```shell
python scripts/depth_jetson1.py
```

The annotated video will save in the output folder. If running on an SSH, you will need to download the video onto your local machine to play it.

## Citation

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

We would like to express our gratitude to [@niels](https://huggingface.co/nielsr) for helping intergrating UniDepth in HuggingFace.

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Automated Cars Europe).
