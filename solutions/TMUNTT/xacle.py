"""XACLE TMUNTT solver.
"""

import sys, os
sys.path.append(os.getenv('EVAR'))

from evar.common import (np, pd, EasyDict, kwarg_cfg,
    torch, F, logging, append_to_csv, app_setup_logger, seed_everything, RESULT_DIR)
import fire
import time
from sklearn import metrics, utils
from scipy.stats import spearmanr
import torchinfo

import torchaudio
import timm.scheduler
import timm.optim

from data import create_xacle_dataloader
from evar.model_utils import set_layers_trainable, show_layers_trainable, MLP
from lineareval import *


torch.backends.cudnn.benchmark = True


class Aggregator(torch.nn.Module):
    def __init__(self, d_model, nhead=1, num_layers=2, wide_ffdim=4, dropout=0.2):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * wide_ffdim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pre_norm = torch.nn.BatchNorm1d(d_model, affine=False)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, text_emb, audio_emb, mask=None):
        """text_emb [B, 1, D], audio_emb [B, :, D]"""
        x = torch.cat([text_emb, audio_emb], dim=1)
        B, T, D = x.shape
        x = self.pre_norm(x.view(B*T, D)).view(B, T, D)
        enc_out = self.encoder(src=x, mask=mask) 
        enc_out = self.norm(enc_out)
        out = enc_out[:, 0]
        return out  # (B, D)


class TaskNetwork(torch.nn.Module):
    def __init__(self, cfg, ar):
        super().__init__()
        self.cfg = EasyDict(cfg.copy())
        self.ar = ar
        self.agg  = Aggregator(cfg.feature_d)
        self.mlp = MLP(input_size=cfg.feature_d, hidden_sizes=cfg.runtime_cfg.hidden, output_size=1, mean=0.0, std=0.01, bias=0.)

        # Load text encoder
        ar.encode_text(['An example audio caption.'])

        if cfg.freeze_ar:
            print(' froze ar model.')
            set_layers_trainable(ar, trainable=False)

        str_ar = show_layers_trainable(self.ar, show_all_trainable=False, print_str=False)
        logging.info(f'Backbone encoder:\n{str_ar}\n')
        str_agg = show_layers_trainable(self.agg, print_str=False)
        logging.info(f'Aggregator:\n{str_agg}\n')
        str_head = show_layers_trainable(self.mlp, print_str=False)
        logging.info(f'Head:\n{str_head}\n')

    def forward(self, batch_audio, batch_caption, return_both=False):
        if not isinstance(batch_caption[0], str):           # for torchinfor compatibility
            batch_caption = ['An example audio caption.']   #  "
        za = self.ar.encode_audio(batch_audio).unsqueeze(1) if self.cfg.use_clap_audio_feature else self.ar.encode_frames(batch_audio).transpose(1, 2)
        zt = self.ar.encode_text(batch_caption).unsqueeze(1)
        z = self.agg(zt, za)
        #z = torch.cat([zt.squeeze(1), z], dim=1)
        z = self.mlp(z).squeeze(-1).to(torch.float)
        if return_both:
            za = za.squeeze(1) if self.cfg.use_clap_audio_feature else self.ar.runtime.backbone.audio_proj(za)
            sim = torch.nn.functional.cosine_similarity(zt.squeeze(1), za)
            return z, sim
        return z


def standard_normal(mean, std):
    r = mean + (std * np.random.standard_normal())
    return r


# copied and modified from https://github.com/nttcslab/byol-a
import random
class RandomResizeCrop(torch.nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.2), freq_scale=(1.0, 0.1), time_scale=(1.0, 0.2)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(standard_normal(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(standard_normal(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward_one(self, lms):
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
                             .to(torch.float).to(lms.device))
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y+h, x:x+w] = lms
        # get random area
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i+h, j:j+w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = torch.nn.functional.interpolate(crop.unsqueeze(0), size=lms.shape[-2:],
            mode=self.interpolation, align_corners=True).squeeze(0)
        return lms.to(torch.float)

    def forward(self, lms):
        if len(lms.shape) == 3:
            return self.forward_one(lms)
        for i in range(len(lms)):
            lms[i] = self.forward_one(lms[i])
        return lms

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
        format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))
        return format_string


class SpecAugment:
    @staticmethod
    def is_required(freqm, timem):
        if freqm > 0:
            return True
        if timem > 0:
            return True
        return False

    def __init__(self, freqm, timem):
        self.freqmask = torchaudio.transforms.FrequencyMasking(freqm) if freqm > 0 else None
        self.timemask = torchaudio.transforms.TimeMasking(timem) if timem > 0 else None

    def __call__(self, lms):
        if self.freqmask is not None:
            lms = self.freqmask(lms)
        if self.timemask is not None:
            lms = self.timemask(lms)
        return lms


class AudioFineuneAug:
    def __init__(self, freqm, timem, rrc=False):
        self.spec_aug = SpecAugment(freqm, timem) if SpecAugment.is_required(freqm, timem) else None
        self.rrc = RandomResizeCrop() if rrc else None
        if self.spec_aug is not None:
            logging.info(f' using SpecAugmentation with {freqm}, {timem}.')
        if self.rrc is not None:
            logging.info(f' using {self.rrc}')

    def __call__(self, lms):
        lms = lms if self.spec_aug is None else self.spec_aug(lms)
        lms = lms if self.rrc is None else self.rrc(lms)
        return lms


def loss_mse(logits, gts, loss_weight):
    weights = {
        0: torch.ones_like(gts),
        1: torch.clamp(gts/20.0 + 0.5, min=0, max=1.0),  # test1
        2: torch.clamp(gts/10.0 + 0.1, min=0, max=1.0),  # test2
        3: torch.clamp(gts/5.0 - 0.4, min=0, max=1.0)  # test3
    }[loss_weight]
    return F.mse_loss(logits, gts, weight=weights)


def eval_srcc(y_score, y_true, sims, clap_ensemble=False):
    org_y_score = y_score.copy()
    # CLAP score ensembling
    clap_based_scores = (sims - sims.min()) / (sims.max() - sims.min()) * 10.0  # min-max scaling to [0,10] range
    if clap_ensemble:
        loc = sims < (sims.mean() - sims.std()*2)
        y_score[loc] = clap_based_scores[loc]

    srcc = spearmanr(y_true, y_score).correlation
    return srcc, pd.DataFrame({'GT': y_true, 'pred': y_score, 'sim': sims, 'org_pred': org_y_score, 'clap_based_score': clap_based_scores})


class Mixup(object):
    def __init__(self, mixup_alpha=0.1):
        self.mixup_alpha = mixup_alpha
        logging.info(f' using mixup with alpha={mixup_alpha}')

    def get_lambda(self, batch_size, device):
        lambdas = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
        self.lambdas = torch.tensor(lambdas).to(torch.float).to(device)
        self.counter_indexes = np.random.permutation(batch_size)

    def __call__(self, x_and_y):
        if self.mixup_alpha == 0.0:
            return x_and_y
        def do_mixup(x, mixup_lambda):
            x = x.transpose(0, -1)
            out = x * mixup_lambda + x[..., self.counter_indexes] * (1.0 - mixup_lambda)
            return out.transpose(0, -1)
        self.get_lambda(len(x_and_y[0]), x_and_y[0].device)
        x_and_y = [do_mixup(z, self.lambdas) for z in x_and_y]
        return x_and_y


def evaluate(model, loader, device, eval_fn, clap_ensemble=False):
    model.eval()
    all_probs, all_gts, all_sims= [], [], []
    for batch in loader:
        with torch.no_grad():
            X, cap, y_gt = batch
            prob, sim = model(X.to(device), cap, return_both=True)
            all_probs.append(prob.detach().cpu().numpy())
            all_sims.append(sim.detach().cpu().numpy())
        all_gts.append(y_gt.numpy())
    y_score = np.hstack(all_probs)
    y_true = np.hstack(all_gts)
    sims = np.hstack(all_sims)

    return eval_fn(y_score, y_true, sims, clap_ensemble=clap_ensemble)


def arg_conf_str(args, defaults={
    'lr': (0.0, 'lr', 'z'),
    'mixup': (0.0, 'mu', 'z'),
    'freq_mask': (0, 'fm', 'asis'),
    'time_mask': (0, 'tm', 'asis'),
    'balanced': (False, 'bal', 'b'),
    'warmup_epochs': (5, 'wu', 'asis'),
    'seed': (42, 's', 'asis'),
    'training_mask': (0.0, 'tx', 'z'),
    'rrc': (False, 'R', 'b'),
    'optim': ('sgd', 'O', 'asis'),
    'unit_sec': (None, 's', 'asis'),
}):
    confstr = ''
    for k in defaults:
        try:
            arg_value = eval('args.' + k)
        except:
            continue # no parameter k for the run.
        if arg_value == defaults[k][0]:
            continue
        arg_key, value_format = defaults[k][1:]
        value = str(arg_value)
        if value_format == 'z':
            value = value.replace('0.', '')
        elif value_format == 'b':
            value = '' # nothing to add
        elif value_format == 'head':
            value = value[:1]
        confstr += arg_key + value
    return confstr


def _train(cfg, ar_model, device, logpath, train_loader, valid_loader, test_loader, loss_fn, multi_label, seed, lr, balanced, verbose):
    loss_fn = loss_mse
    eval_fn = eval_srcc
    crit_str = 'srcc'

    if cfg.optim == 'adamw': optimizer = torch.optim.AdamW(ar_model.parameters(), lr=lr, weight_decay=0.0001, betas=(0.9, 0.95), eps=1e-08, amsgrad=True)
    elif cfg.optim == 'sgd': optimizer = torch.optim.SGD(ar_model.parameters(), lr, momentum=0.9, weight_decay=0)
    elif cfg.optim == 'lars': optimizer = timm.optim.Lars(ar_model.parameters(), lr, momentum=0.9, weight_decay=0)
    elif cfg.optim == 'lamb': optimizer = timm.optim.Lamb(ar_model.parameters(), lr)
    assert cfg.optim in ['adamw', 'sgd', 'lars', 'lamb']

    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=cfg.ft_epochs, lr_min=1e-7, warmup_t=cfg.warmup_epochs, warmup_lr_init=0)
    logging.info(f'Using {loss_fn.__name__} with weight type {cfg.loss_weight}, {eval_fn.__name__}, and {optimizer}')

    # Training begins here.
    time1 = time.time()
    best_result, best_path, best_epoch = 0.0, None, 0
    epoch_iters = len(train_loader)
    console_iters = max(10, epoch_iters // 10)

    # Set test set as validation set if not available; i.e., val result = test result in this case.
    if len(valid_loader.dataset) == 0:
        print(' ** Fine-tuning using Evaluation set result as test result **')
        valid_loader = test_loader

    # Augmentations for fine tuning
    # NOT implemented: mixup = Mixup(mixup_alpha=cfg.mixup)
    aug_fn = AudioFineuneAug(cfg.ft_freq_mask, cfg.ft_time_mask, rrc=cfg.ft_rrc)
    ar_model.module.ar.set_augment_tf_feature_fn(aug_fn)

    # Name this session
    name  = f'{cfg.id}{"" if cfg.weight_file != "" else "/rnd"}-'
    name += arg_conf_str(EasyDict({'mixup': cfg.mixup, 'freq_mask': cfg.ft_freq_mask, 'time_mask': cfg.ft_time_mask,
        'rrc': cfg.ft_rrc, 'lr': lr, 'warmup_epochs': cfg.warmup_epochs, 'balanced': balanced, 'seed': seed, 'training_mask': cfg.training_mask,
        'optim': cfg.optim, 'unit_sec': cfg.unit_sec}))

    for epoch in range(cfg.ft_epochs):
        for iter, batch in enumerate(train_loader):
            # Train
            ar_model.train()
            # X_aug, y_aug = mixup(batch)
            # if not isinstance(X_aug, list):
            #     X_aug, y_aug = X_aug.to(device), y_aug.to(device)
            X_aug, cap, y_aug = batch
            X_aug, y_aug = X_aug.to(device), y_aug.to(device)

            probs = ar_model(X_aug, cap)
            loss = loss_fn(probs, y_aug, cfg.loss_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            micro_epoch = epoch + iter/epoch_iters
            scheduler.step(micro_epoch)

            if iter % console_iters == 0:
                logging.info(f'Epoch [{epoch}] iter: {iter}/{epoch_iters}, elapsed: {time.time() - time1:.3f}s,'
                           + f' lr: {optimizer.param_groups[0]["lr"]:.8f} loss: {float(loss):.8f}')
                time1 = time.time()

            # balanced training = infinity training iterations -> manually break
            if balanced and iter + 1 >= epoch_iters:
                break

        # Epoch done -> Evaluate
        print('validating')
        val_result, df = evaluate(ar_model, valid_loader, device, eval_fn, cfg.validate_by_clap_ensemble)
        report = f'{name} | epoch/iter {epoch}/{iter}: '
        report += f'val {crit_str}: {val_result:.5f}, loss: {float(loss):.5f}'

        # Save the best model
        new_best_record = best_result < val_result
        if new_best_record: # following PANNs implementation, measuring potential performance.
            best_result = val_result
            best_epoch = epoch
            if best_path is not None:
                best_path.unlink()
            best_path = logpath/f'weights_ep{epoch}it{iter}-{val_result:.5f}_loss{loss:.4f}.pth'
            torch.save(ar_model.state_dict(), best_path)
            logging.info(f'Saved weight as {best_path}')
            df.to_csv(logpath/f'ep{epoch}it{iter}-{val_result:.5f}.csv')
        report += f', best: {best_result:.5f}@{best_epoch}'

        # Report to log and dashboard
        logging.info(report)

        # Stop condition
        if cfg.ft_early_stop_epochs > 0 and epoch > best_epoch + cfg.ft_early_stop_epochs:
            logging.info(f'Early stopping now, the best epoch was {best_epoch}.')
            break

    # Test result
    if valid_loader != test_loader:
        logging.info(f'Load best weight from {best_path}')
        ar_model.load_state_dict(torch.load(best_path))
        print('testing')
        best_result, df = evaluate(ar_model, test_loader, device, eval_fn, cfg.validate_by_clap_ensemble)
        logging.info(f'Final test {crit_str}: {best_result:.5f}')
    else:
        logging.info(f'Best {crit_str}: {best_result:.5f}')

    return best_result, best_path, name


def run_eval(config_file, task='xacle', options='', seed=42, lr=None, hidden=(512,), mixup=None, batch_size=None,
                          epochs=None, early_stop_epochs=None, warmup_epochs=None,
                          freq_mask=None, time_mask=None, rrc=None, training_mask=None,
                          optim='sgd', unit_sec=None, verbose=True, test_only=False, freeze_ar=None, loss_fn=None, data_path='work'):
    cfg, n_folds, balanced = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec)
    lr = lr or cfg.ft_lr
    cfg.mixup = mixup if mixup is not None else cfg.mixup
    cfg.ft_early_stop_epochs = early_stop_epochs if early_stop_epochs is not None else cfg.ft_early_stop_epochs
    cfg.warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg.warmup_epochs
    cfg.ft_epochs = epochs or cfg.ft_epochs
    cfg.ft_freq_mask = freq_mask if freq_mask is not None else cfg.ft_freq_mask
    cfg.ft_time_mask = time_mask if time_mask is not None else cfg.ft_time_mask
    cfg.ft_rrc = rrc if rrc is not None else (cfg.ft_rrc if 'ft_rrc' in cfg else False)
    cfg.freeze_ar = freeze_ar if freeze_ar is not None else cfg.freeze_ar
    cfg.training_mask = training_mask if training_mask is not None else (cfg.training_mask if 'training_mask' in cfg else 0.0)
    cfg.ft_bs = batch_size or cfg.ft_bs
    cfg.optim = optim
    cfg.unit_sec = unit_sec
    cfg.data_path = data_path

    cfg.name = f'test{cfg.loss_weight}s{seed}'

    # Make audio representation model and downstream task model.
    train_loader, _, _, _ = create_xacle_dataloader(cfg, fold=0, seed=seed, batch_size=cfg.ft_bs)

    cfg.runtime_cfg = kwarg_cfg(lr=lr, seed=seed, hidden=hidden, mixup=cfg.mixup, bs=cfg.ft_bs,
                                freq_mask=cfg.ft_freq_mask, time_mask=cfg.ft_time_mask, rrc=cfg.ft_rrc, freeze_ar=cfg.freeze_ar, epochs=cfg.ft_epochs,
                                early_stop_epochs=cfg.ft_early_stop_epochs, n_class=len(train_loader.dataset.classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)
    logpath = app_setup_logger(cfg, level=logging.INFO) # Add this when debugging deeper: level=logging.DEBUG

    scores = []
    for fold in range(1, n_folds + 1):
        logging.info(f'\n🚀 Start fine-tuning {f"fold#{fold}/{n_folds}" if n_folds > 1 else ""} with logging in {logpath}')

        # Dataloaders for current fold.
        train_loader, valid_loader, test_loader, multi_label = create_xacle_dataloader(cfg, fold=fold, seed=seed, batch_size=cfg.ft_bs, always_one_hot=True)
        logging.info(f'Train:{len(train_loader.dataset)}, valid:{len(valid_loader.dataset)}, test:{len(test_loader.dataset)}, multi label:{multi_label}, balanced:{balanced}')

        # Make a fresh model
        ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
        if hasattr(train_loader, 'lms_mode') and train_loader.lms_mode:
            ar.precompute_lms(device, train_loader)
        else:
            ar.precompute(device, train_loader)
        task_model = TaskNetwork(cfg, ar).to(device)
        print(task_model)
        task_model_dp = torch.nn.DataParallel(task_model).to(device)

        if test_only:
            weights = torch.load(test_only, map_location=device, weights_only=False)
            task_model_dp.load_state_dict(weights)
            eval_fn, crit_str = eval_srcc, 'srcc'
            best_result, df = evaluate(task_model_dp, test_loader, device, eval_fn, cfg.validate_by_clap_ensemble)
            test_results_csv = Path(test_only).parent/'submission.csv'
            # XACLE compliant submission format
            sub_df = test_loader.dataset.df[['file_name']].copy()
            sub_df['wav_file_name'] = sub_df['file_name'].apply(lambda x: x.split('/')[-1])
            sub_df['pred_score'] = df.pred
            sub_df[['wav_file_name', 'pred_score']].to_csv(test_results_csv, index=False)

            import torchinfo
            model_profile= torchinfo.summary(task_model_dp, [(1, 160000), (1, 30)], dtypes=[torch.float, torch.int])
            print(model_profile)

            return [best_result], test_only, 'eval only', cfg, test_results_csv

        best_result, best_path, name = _train(cfg, task_model_dp, device, logpath, train_loader, valid_loader, test_loader,
            loss_fn, multi_label, seed, lr, balanced, verbose)

        scores.append(best_result)
        if n_folds > 1:
            print(f' fold={fold}: {best_result:.5f}')

    return scores, best_path, name, cfg, logpath


def finetune_main(config_file, task='xacle', options='', seed=42, lr=None, hidden=(512,), epochs=None, early_stop_epochs=None, warmup_epochs=None,
                  mixup=None, freq_mask=None, time_mask=None, rrc=None, training_mask=None, batch_size=None,
                  optim='lars', unit_sec=None, verbose=False, test_only=False, freeze_ar=None, loss_fn=None, data_path='work'):
    scores, best_path, name, cfg, logpath = run_eval(config_file, task, options=options, seed=seed, lr=lr, hidden=hidden, mixup=mixup,
        batch_size=batch_size, epochs=epochs, early_stop_epochs=early_stop_epochs, warmup_epochs=warmup_epochs,
        freq_mask=freq_mask, time_mask=time_mask, rrc=rrc, training_mask=training_mask, optim=optim,
        unit_sec=unit_sec, verbose=verbose, test_only=test_only, freeze_ar=freeze_ar, loss_fn=loss_fn, data_path=data_path)
    mean_score = np.mean(scores)
    report = f'Finetuning {name} on {cfg.task_name} -> mean score: {mean_score:.5f}'
    if test_only:
        print(report)
        logging.info(f'Evaluated {best_path} on the test set.')
        logging.info(f'Saved test results to {logpath}')
        return

    score_file = logpath/f'{cfg.task_name}_{cfg.audio_repr.replace("AR_", "").replace("_", "-")}-FT_{cfg.id[-8:]}_{mean_score:.5f}.csv'
    best_report = logpath/(best_path.stem.split('_')[1] + '.csv')
    best_report.rename(score_file)

    if len(scores) > 1:
        report += ', scores: [' + ', '.join([f'{score:.5f}' for score in scores]) + ']'
    report += f', best weight: {best_path}, score file: {score_file}, config: {cfg}'
    logging.info(report)

    result_df = pd.DataFrame({
        'representation': [cfg.id.split('.')[-1][3:-9] if '.AR_' in cfg.id else cfg.id[:-9]], # AR name
        'task': [cfg.task_name],
        'score': [mean_score],
        'run_id': [cfg.id],
        'report': [report],
    })
    append_to_csv(f'{RESULT_DIR}/ft-scores.csv', result_df)
    return report, scores, best_path, name, cfg, logpath


if __name__ == '__main__':
    fire.Fire(finetune_main)
