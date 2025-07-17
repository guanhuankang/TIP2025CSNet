## A Contrastive-Learning Framework for Unsupervised Salient Object Detection

## Training & Inference Env Installation
We use `uv` to manage our environment.
```shell
uv venv
source .venv/bin/activate
uv pip install torch torchvision numpy pillow progressbar thop pandas opencv-python tqdm joblib albumentations tensorboard
uv pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```
Or `uv pip install -r requirements.txt` and then `uv pip install git+https://github.com/lucasb-eyer/pydensecrf.git`.

## Training
```shell
source .venv/bin/activate
python train.py --name training_csnet
```
The output is located in `assets/output/training_csnet`. 

### Inference
```shell
source .venv/bin/activate
python testmodel.py --weights assets/model_cards/csnet_init_13200.pth --name inference_csnet_init_13200
```
The output is located in `assets/output/inference_csnet_init_13200`. If you want results without CRF, run `python testmodel.py --weights assets/model_cards/csnet_init_13200.pth --name inference_csnet_init_13200 --crf_round int:0` (faster, No CRF).

### Model Complexity
```shell
python flops.py --name flops
```

### Model Cards

### config.json
`crf`: The number of CRF iteration during training step. [only valid for training]  
`crf_round`: The number of CRF iteration for evaluation. default: 10. If it is set to 0, no CRF is applied. If less than 0, apply bilateral solver.  
`weights`: Full checkpoint.  
`backboneWeight`: backbone checkpoint. 

