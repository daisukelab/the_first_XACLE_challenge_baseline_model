"""XACLE dataset class.
"""

from evar.common import (np, torch)
import logging
from evar.data import WavDataset


class WavCapScoreDataset(WavDataset):
    def __getitem__(self, index):
        wav, _ = super().__getitem__(index)
        cap    = self.df.caption.values[index]
        score  = self.df.score.values[index].astype(np.float32)
        return wav, cap, score


def create_xacle_dataloader(cfg, fold=1, seed=42, batch_size=None, always_one_hot=False, pin_memory=False, num_workers=8):
    batch_size = batch_size or cfg.batch_size
    train_dataset = WavCapScoreDataset(cfg, 'train', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=True)
    valid_dataset = WavCapScoreDataset(cfg, 'valid', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=False,
        classes=train_dataset.classes)
    test_dataset = WavCapScoreDataset(cfg, 'test', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=False,
        classes=train_dataset.classes)
    logging.info(f' classes: {train_dataset.classes}')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                            num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)

    return (train_loader, valid_loader, test_loader, train_dataset.multi_label)
