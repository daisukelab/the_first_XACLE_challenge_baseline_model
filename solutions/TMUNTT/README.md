# TMUNTT solution for XACLE 2025

This repository provides everything about our simple solution for XACLE 2025.

### 1-1. Install packages
Make sure you installed the following, in addition to the requirements of [EVAR](https://github.com/nttcslab/eval-audio-repr) and [M2D](https://github.com/nttcslab/m2d).

- torch>=2.6.0
- torchinfo

Note that `torch.nn.functional.mse_loss` requires `torch>=2.6.0`.

### 1-2. Clone EVAR and M2D
Step 1: This solution is built based on [EVAR](https://github.com/nttcslab/eval-audio-repr).

```sh
git clone https://github.com/nttcslab/eval-audio-repr.git evar
cd evar
curl https://raw.githubusercontent.com/daisukelab/general-learning/master/MLP/torch_mlp_clf2.py -o evar/utils/torch_mlp_clf2.py
curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/sampler.py -o evar/sampler.py
curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/cnn14_decoupled.py -o evar/cnn14_decoupled.py
curl https://raw.githubusercontent.com/XinhaoMei/WavCaps/master/retrieval/tools/utils.py -o evar/utils/wavcaps_utils.py
cd ..
```

Step 2: This solution uses an audio foundation model M2D-CLAP (2025): [M2D](https://github.com/nttcslab/m2d).

```sh
(in this folder)
cd evar/external
git clone https://github.com/nttcslab/m2d.git
cd m2d
curl -o util/lars.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lars.py
curl -o util/lr_decay.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_decay.py
curl -o util/lr_sched.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_sched.py
curl -o util/misc.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py
curl -o util/analyze_repr.py https://raw.githubusercontent.com/daisukelab/general-learning/master/SSL/analyze_repr.py
curl -o m2d/pos_embed.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py
curl -o train_audio.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o speech/train_speech.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o audioset/train_as.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o clap/clap_only.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o clap/train_clap.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o mae_train_audio.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o m2d/engine_pretrain_m2d.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/engine_pretrain.py
curl -o m2d/models_mae.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py
curl -o m2d/timm_layers_pos_embed.py https://raw.githubusercontent.com/huggingface/pytorch-image-models/e9373b1b925b2546706d78d25294de596bad4bfe/timm/layers/pos_embed.py
patch -p1 < patch_m2d.diff
cd ../../..
```

Step 3: Download the M2D-CLAP weight.

```sh
wget https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip
unzip m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip
```

### 1-3. Dataset and metadata
Step 1: Make sure you have root/datasets/XACLE_dataset.

    this_repository/
      datasets/
        XACLE_dataset/
          wav/
          metadata/
            test.csv, train.csv, train_average.csv, validation.csv, validation_average.csv

Step 2: Create a symbolic link for EVAR.
```sh
(in this folder)
mkdir -p evar/work/16k
ln -s ../../../../../datasets/XACLE_dataset/wav evar/work/16k/xacle
```

Step 3: Open [Notte_metadata.ipynb](./Note_metadata.ipynb) and run all the steps, and it will create `evar/evar/metadata/xacle.csv` and `xacle_test.csv` files.

### 1-4. Train and test
Training takes about a half day.

```sh
EVAR=./evar python -m xacle config_m2d-clap.yaml xacle --seed 42
```

Once training is done, you will see a log like:

```console
    :
Epoch [98] iter: 50/59, elapsed: 5.719s, lr: 0.00000020 loss: 2.05708933                                                                             
validating                                                
M2D-CLAP2025-XACLE_xacle_69141b68-lr0003fm40tm192Olars | epoch/iter 98/58: val srcc: 0.59450, loss: 2.26845, best: 0.60528@38                        
Epoch [99] iter: 0/59, elapsed: 20.355s, lr: 0.00000017 loss: 2.68400788                                                                             
Epoch [99] iter: 10/59, elapsed: 5.706s, lr: 0.00000015 loss: 2.19344354                                                                             
Epoch [99] iter: 20/59, elapsed: 5.697s, lr: 0.00000013 loss: 1.71632731                                                                             
Epoch [99] iter: 30/59, elapsed: 5.719s, lr: 0.00000012 loss: 2.23739696                                                                             
Epoch [99] iter: 40/59, elapsed: 5.705s, lr: 0.00000011 loss: 1.86376655                                                                             
Epoch [99] iter: 50/59, elapsed: 5.724s, lr: 0.00000010 loss: 1.98289084                                                                             
Epoch [99] iter: 50/59, elapsed: 5.724s, lr: 0.00000010 loss: 1.98289084
validating
M2D-CLAP2025-XACLE_xacle_69141b68-lr0003fm40tm192Olars | epoch/iter 99/58: val srcc: 0.59417, loss: 2.27783, best: 0.60528@38
Load best weight from logs/M2D-CLAP2025-XACLE_xacle_69141b68/weights_ep38it58-0.60528_loss2.9196.pth
testing
Final test srcc: 0.60528
Finetuning M2D-CLAP2025-XACLE_xacle_69141b68-lr0003fm40tm192Olars on xacle -> mean score: 0.60528, best weight: logs/M2D-CLAP2025-XACLE_xacle_69141b68/weights_ep38it58-0.60528_loss2.9196.pth ...
```

Then, the following test step will create submission.csv for the test set.

```sh
EVAR=./evar python -m xacle config_m2d-clap.yaml xacle_test --test_only logs/M2D-CLAP2025-XACLE_xacle_69141b68/weights_ep38it58-0.60528_loss2.9196.pth
```

Find your result (`logs/M2D-CLAP2025-XACLE_xacle_69141b68/submission.csv` in the following example) in the last part:

```console
    :
Evaluated logs/M2D-CLAP2025-XACLE_xacle_69141b68/weights_ep38it58-0.60528_loss2.9196.pth on the test set.
Saved test results to logs/M2D-CLAP2025-XACLE_xacle_69141b68/submission.csv
```
