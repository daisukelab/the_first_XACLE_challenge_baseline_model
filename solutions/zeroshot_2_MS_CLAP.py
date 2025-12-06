"""Zeroshot inference example solution for XACLE Challenge (a part of ICASSP 2026 SP Grand Challenge)
 - The first x-to-audio alignment challenge -

## MS-CLAP example

This example simply calculates audio-text similarities and map them to a [0, 10] range on the validation set.
The inference result will be stored in `zs_inference_scores.csv`.

Example run:

```sh
$ python zeroshot_2_MS_CLAP.py
Loading MS-CLAP ...
......................... (3000 dots goes on)
....................

MS-CLAP (2023) zeroshot inference result is srcc: 0.35915776005202416
```

"""

import sys
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from msclap import CLAP


def zeroshot_inference_MSCLAP(data_dir='datasets/XACLE_dataset', version='2023'):
    print('Loading MS-CLAP ...')
    clap = CLAP(version=version)
    # clap.eval() -> No eval() available with MS-CLAP

    val = pd.read_csv(data_dir + '/meta_data/validation_average.csv')

    msc_sims = []
    for f, cap in val[['wav_file_name', 'text']].values:
        with torch.no_grad():
            za = clap.get_audio_embeddings([data_dir + '/wav/validation/' + f])
            zt = clap.get_text_embeddings([cap])
        sim = F.cosine_similarity(za, zt).cpu().squeeze(0).numpy()
        msc_sims.append(sim)
        print('.', end='', flush=True)
    print('\n')
    sims = np.array(msc_sims)

    minmax_score = (sims - sims.min()) / (sims.max() - sims.min()) * 10.0
    val['pred_score'] = minmax_score

    return val


if __name__ == "__main__":
    data_dir  = sys.argv[1] if len(sys.argv) >= 2 else 'datasets/XACLE_dataset'
    version   = sys.argv[2] if len(sys.argv) >= 3 else '2023'
    save_file = sys.argv[3] if len(sys.argv) >= 4 else 'zs_inference_scores.csv'

    df = zeroshot_inference_MSCLAP(data_dir=data_dir, version=version)
    srcc = spearmanr(df.average_score, df.pred_score).correlation

    print(f'MS-CLAP ({version}) zeroshot inference result is srcc: {srcc}')

    df[['wav_file_name', 'pred_score']].to_csv(save_file, index=None)
