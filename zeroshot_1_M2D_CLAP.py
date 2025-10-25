"""Zeroshot inference example solution for XACLE Challenge (a part of ICASSP 2026 SP Grand Challenge)
 - The first x-to-audio alignment challenge -

## M2D-CLAP example

This example simply calculates audio-text similarities and map them to a [0, 10] range on the validation set.
The inference result will be stored in `zs_inference_scores.csv`.

Example run:

```sh
$ python zeroshot_1_M2D_CLAP.py
Downloading portable_m2d.py...
Downloading m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip... (It takes long)
done.

Loading M2D-CLAP ...
 using 166 parameters from m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth
 (included audio_proj params: ['audio_proj.sem_token', 'audio_proj.sem_blocks.0.norm1.weight', 'audio_proj.sem_blocks.0.norm1.bias', 'audio_proj.sem_blocks.0.attn.qkv.weight', 'audio_proj.sem_blocks.0.attn.qkv.bias']
 (included text_proj params: []
 (dropped: [] )
<All keys matched successfully>
 using model.text_encoder from m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth
......................... (3000 dots goes on)
..........................

M2D-CLAP (2025) zeroshot inference result is srcc: 0.42608136024182597
```

"""

import sys
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def download_M2D_CLAP_2025():
    import requests
    import zipfile
    import io

    print('Downloading portable_m2d.py...')
    response = requests.get('https://raw.githubusercontent.com/nttcslab/m2d/refs/heads/master/examples/portable_m2d.py')
    response.raise_for_status()
    with open('./portable_m2d.py', "wb") as f:
        f.write(response.content)

    print('Downloading m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip... (It takes long)')
    response = requests.get('https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip')
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall('.')
    print('done.\n')

try:
    import portable_m2d
except:
    download_M2D_CLAP_2025()
    import portable_m2d


def zeroshot_inference_M2D_CLAP_2025(data_dir='datasets/XACLE_dataset'):
    print('Loading M2D-CLAP ...')
    clap = portable_m2d.PortableM2D('m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth', flat_features=True).to('cuda:0')
    clap.eval()

    val = pd.read_csv(data_dir + '/meta_data/validation_average.csv')

    sims = []
    for f, cap in val[['wav_file_name', 'text']].values:
        wav, _ = librosa.load(data_dir + '/wav/validation/' + f, sr=None, mono=True)
        with torch.no_grad():
            za = clap.encode_clap_audio(torch.tensor(wav).unsqueeze(0).to('cuda:0'))
            zt = clap.encode_clap_text([cap])
        sim = F.cosine_similarity(za, zt).cpu().squeeze(0).numpy()
        sims.append(sim)
        print('.', end='', flush=True)
    print('\n')
    sims = np.array(sims)

    minmax_score = (sims - sims.min()) / (sims.max() - sims.min()) * 10.0
    val['pred_score'] = minmax_score

    return val


if __name__ == "__main__":
    data_dir  = sys.argv[1] if len(sys.argv) >= 2 else 'datasets/XACLE_dataset'
    save_file = sys.argv[2] if len(sys.argv) >= 3 else 'zs_inference_scores.csv'

    df = zeroshot_inference_M2D_CLAP_2025(data_dir=data_dir)
    srcc = spearmanr(df.average_score, df.pred_score).correlation

    print(f'M2D-CLAP (2025) zeroshot inference result is srcc: {srcc}')

    df[['wav_file_name', 'pred_score']].to_csv(save_file, index=None)
