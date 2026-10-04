import os, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('[warn] xgboost unavailable')

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAIN_CSV     = '../data/Enhanced_5G_Dataset_100K_v311.csv'
META_CSV     = '../data/Enhanced_5G_Dataset_100K_v3_metadata11.csv'
OOD_CSV      = '../data/Enhanced_5G_Dataset_OOD_5K_v311.csv'
OOD_META_CSV = '../data/Enhanced_5G_Dataset_OOD_5K_v311_metadata.csv'
OUT          = '../results'
os.makedirs(OUT, exist_ok=True)

FEATURE_COLS = [
    'EstChannelGain','K_factor_PDP_dB','K_factor_Moment_dB',
    'Est_RMS_DelaySpread_ns','Est_Coherence_BW_MHz','Est_Num_Paths',
    'Est_Max_Delay_ns','FreqDomain_GainVariability_dB',
    'Rx_Power','Rx_Power_Std','Rx_Phase_Mean',
    'Equalized_Signal_Power','Equalized_Signal_Std','Kurtosis',
    'Num_Tx_Antennas','Num_Rx_Antennas']
ALL_IDX = list(range(16))

LOG10_COLS = ['EstChannelGain','Est_Coherence_BW_MHz','Rx_Power','Rx_Power_Std',
              'Equalized_Signal_Power','Equalized_Signal_Std','Kurtosis']
LOG1P_COLS = ['Est_RMS_DelaySpread_ns','Est_Max_Delay_ns']

ALL_PROFILES = ['CDL-A','CDL-B','CDL-C','CDL-D']
CDL_NAMES = {'CDL-A':'Indoor Office','CDL-B':'Urban LOS',
             'CDL-C':'Urban NLOS','CDL-D':'Canyon/Highway'}
OOD_BLOCKS = [1250, 1000, 1750, 1000]

SNR_EDGES = np.arange(-15, 26, 5)
BATCH_TRAIN, BATCH_FT = 512, 128
EPOCHS_PRE, EPOCHS_FT = 60, 20
LR_PRE, LR_FT = 3e-3, 1e-3
WD, CLIP_GRAD = 1e-4, 1.0

SENS_FRACS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
SENS_REPEATS = 3
SATURATION_DELTA = 0.02

MODEL_COLORS = {'CEF':'#D62728', 'ResNet1D':'#FFBB78', 'Dilated-CNN':'#FF7F0E',
                'MLP-16':'#7F7F7F', 'XGBoost':'#8C564B'}

def sep(n=72, c='='): print(c * n)
def hdr(t): sep(); print(f"  {t}"); sep()

def load_main():
    df = pd.read_csv(MAIN_CSV)
    meta = pd.read_csv(META_CSV)
    if len(df) != len(meta):
        raise ValueError(f'row count mismatch: {len(df)} vs {len(meta)}')
    if not np.allclose(df['SNR_dB'].values, meta['SNR_true'].values, atol=1e-3):
        raise ValueError('SNR_dB and SNR_true not aligned')
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f'missing feature column: {c}')
    cdl = meta['CDL_Profile'].map({p:i for i,p in enumerate(ALL_PROFILES)}).values
    if np.any(pd.isna(cdl)):
        raise ValueError('unknown CDL in metadata')
    S = np.stack([meta['SNR_LS'].values, meta['SNR_ML'].values,
                  meta['SNR_EVM'].values, meta['SNR_DD'].values], 1).astype(np.float64)
    return dict(X=df[FEATURE_COLS].values.astype(np.float64),
                y=df['SNR_dB'].values.astype(np.float64),
                cdl=cdl.astype(int), S=S,
                cls={'SNR_LS':  meta['SNR_LS'].values.astype(np.float64),
                     'SNR_ML':  meta['SNR_ML'].values.astype(np.float64),
                     'SNR_EVM': meta['SNR_EVM'].values.astype(np.float64),
                     'SNR_DD':  meta['SNR_DD'].values.astype(np.float64)})

def load_ood():
    df = pd.read_csv(OOD_CSV)
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f'missing OOD column: {c}')
    cdl = np.repeat(np.arange(4), OOD_BLOCKS)
    if len(cdl) != len(df):
        base = len(df) // 4
        cdl = np.repeat(np.arange(4), [base, base, base, len(df)-3*base])
    if os.path.exists(OOD_META_CSV):
        ood_meta = pd.read_csv(OOD_META_CSV)
        if len(ood_meta) != len(df):
            raise ValueError('OOD metadata row mismatch')
        if not np.allclose(df['SNR_dB'].values, ood_meta['SNR_true'].values, atol=1e-3):
            raise ValueError('OOD features and metadata not aligned')
        S_ood = np.stack([ood_meta['SNR_LS'].values, ood_meta['SNR_ML'].values,
                          ood_meta['SNR_EVM'].values, ood_meta['SNR_DD'].values], 1).astype(np.float64)
        print('  [OK] OOD metadata loaded')
    else:
        raise FileNotFoundError(f'OOD metadata missing: {OOD_META_CSV}')
    return dict(X=df[FEATURE_COLS].values.astype(np.float64),
                y=df['SNR_dB'].values.astype(np.float64),
                cdl=cdl.astype(int), S=S_ood)

def fixed_transform(X):
    Xt = X.copy()
    for c in LOG10_COLS:
        j = FEATURE_COLS.index(c)
        Xt[:, j] = np.log10(np.maximum(Xt[:, j], 0.0) + 1e-12)
    for c in LOG1P_COLS:
        j = FEATURE_COLS.index(c)
        Xt[:, j] = np.log10(1.0 + np.maximum(Xt[:, j], 0.0))
    return Xt

class Standardizer:
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        self.ymu, self.ysd = float(y.mean()), float(y.std())
        return self
    def x(self, X): return np.clip((X - self.mu) / self.sd, -8, 8)
    def y(self, y): return (y - self.ymu) / self.ysd

def onehot(idx, n=4):
    c = np.zeros((len(idx), n), dtype=np.float32)
    c[np.arange(len(idx)), idx] = 1.0
    return c

class SNRDataset(Dataset):
    def __init__(self, X, C, S, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.C[i], self.S[i], self.y[i]

def make_loader(X, C, S, y, bs, shuffle=True):
    return DataLoader(SNRDataset(X, C, S, y), batch_size=bs, shuffle=shuffle,
                      drop_last=False, num_workers=0)

class GridNet(nn.Module):
    def __init__(self, use_cdl, use_fusion, s_idx=None,
                 n_feat=16, n_scn=4, n_cls=4, hidden=256):
        super().__init__()
        self.use_cdl, self.use_fusion = use_cdl, use_fusion
        self.s_idx = s_idx
        n_cls_eff = 0
        if use_fusion:
            n_cls_eff = len(s_idx) if s_idx is not None else n_cls
        in_dim = n_feat + (n_scn if use_cdl else 0) + n_cls_eff
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 128), nn.GELU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1))
    def forward(self, x, c, s):
        parts = [x]
        if self.use_cdl: parts.append(c)
        if self.use_fusion:
            parts.append(s[:, self.s_idx] if self.s_idx is not None else s)
        return self.net(torch.cat(parts, dim=1)).squeeze(1)

class MLP16(nn.Module):
    def __init__(self, in_dim=16, n_scn=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + n_scn, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, x, c, s=None):
        return self.net(torch.cat([x, c], dim=1)).squeeze(1)

class DilatedCNN(nn.Module):
    def __init__(self, in_dim=16, n_scn=4, n_cls=4, channels=32):
        super().__init__()
        self.inp = nn.Conv1d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv1d(channels, channels, 3, padding=d, dilation=d)
                                      for d in [1, 2, 4, 8]])
        self.bn = nn.BatchNorm1d(channels)
        self.head = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, c, s):
        seq = torch.cat([x, c, s], dim=1).unsqueeze(1)
        h = F.relu(self.inp(seq))
        for b in self.blocks: h = F.relu(b(h))
        h = F.relu(self.bn(h)).mean(dim=2)
        return self.head(h).squeeze(1)

class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch))
    def forward(self, x): return F.relu(x + self.b(x))

class ResNet1D(nn.Module):
    def __init__(self, in_dim=16, n_scn=4, n_cls=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 64, 3, padding=1, bias=False),
                                  nn.BatchNorm1d(64), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock1D(64), ResBlock1D(64), ResBlock1D(64))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, c, s):
        seq = torch.cat([x, c, s], dim=1).unsqueeze(1)
        return self.head(self.blocks(self.stem(seq)).mean(dim=2)).squeeze(1)

class CauchyLoss(nn.Module):
    def __init__(self, delta=1.0): super().__init__(); self.delta = delta
    def forward(self, p, y):
        return torch.mean(torch.log1p(((p - y) / self.delta) ** 2))

def huber(y_sd, delta_db=3.0):
    return nn.HuberLoss(delta=delta_db / y_sd)

def make_loss(name, hd, y_sd):
    if name == 'mse':    return nn.MSELoss()
    if name == 'huber':  return nn.HuberLoss(delta=hd / y_sd)
    if name == 'cauchy': return CauchyLoss(delta=hd / y_sd)
    raise ValueError(name)

def train_loop(model, loader, opt, crit):
    model.train(); tot = 0.0
    for xb, cb, sb, yb in loader:
        xb, cb, sb, yb = xb.to(DEVICE), cb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
        if xb.size(0) < 2: continue
        opt.zero_grad()
        loss = crit(model(xb, cb, sb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        opt.step()
        tot += loss.item() * xb.size(0)
    return tot / max(len(loader.dataset), 1)

@torch.no_grad()
def predict(model, X, C, S, bs=4096):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=DEVICE)
        cb = torch.tensor(C[i:i+bs], dtype=torch.float32, device=DEVICE)
        sb = torch.tensor(S[i:i+bs], dtype=torch.float32, device=DEVICE)
        out.append(model(xb, cb, sb).cpu().numpy())
    return np.concatenate(out)

def fit(model, Xtr, Ctr, Str, ytr, Xva, Cva, Sva, yva, crit, epochs, lr, bs):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    ldr = make_loader(Xtr, Ctr, Str, ytr, bs, shuffle=True)
    best, best_state = float('inf'), None
    for _ in range(epochs):
        train_loop(model, ldr, opt, crit); sch.step()
        p = predict(model, Xva, Cva, Sva)
        v = float(np.sqrt(mean_squared_error(yva, p)))
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return model, best

def get_metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(RMSE=float(np.sqrt(np.mean(e**2))),
                MAE=float(np.mean(np.abs(e))),
                Bias=float(np.mean(e)),
                R2=float(1 - np.sum(e**2)/ss) if ss > 0 else float('nan'),
                N=int(len(y)))

def bootstrap_ci(y, p, n=500):
    rng = np.random.default_rng(SEED); out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        out.append(np.sqrt(mean_squared_error(y[idx], p[idx])))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def bin_metrics(y, p):
    rows = []
    for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
        m = (y >= lo) & (y < hi)
        if m.sum() > 0:
            e = p[m] - y[m]
            rows.append(dict(bin_lo=int(lo),
                             RMSE=float(np.sqrt(np.mean(e**2))),
                             Bias=float(np.mean(e)),
                             N=int(m.sum())))
    return rows

def split_target(cdl_val, D, seed_tag):
    te_idx = np.where(D['cdl'] == cdl_val)[0]
    rng_e = np.random.default_rng(10007 * SEED + 131 * seed_tag + 7)
    te_idx = te_idx.copy(); rng_e.shuffle(te_idx)
    n_test, n_val = int(0.50 * len(te_idx)), int(0.10 * len(te_idx))
    return te_idx[:n_test], te_idx[n_test:n_test + n_val], te_idx[n_test + n_val:]

def zeros4(n): return np.zeros((n, 4), dtype=np.float32)

hdr("LOADING DATA")
D = load_main(); O = load_ood()
print(f"Main : {len(D['y']):,} | CDL counts = {np.bincount(D['cdl']).tolist()}")
print(f"OOD  : {len(O['y']):,} | CDL counts = {np.bincount(O['cdl']).tolist()}")
print(f"Device: {DEVICE}")

Xt = fixed_transform(D['X'])
Oxt = fixed_transform(O['X'])
C_all, S_all = onehot(D['cdl']), D['S']
Co_all, So_all = onehot(O['cdl']), O['S']

hdr("TABLE III - CLASSICAL ESTIMATORS PER CDL")
cls_names = {'LS-SNR':'SNR_LS','ML-SNR':'SNR_ML','EVM-SNR':'SNR_EVM','DD-SNR':'SNR_DD'}
rows3 = []
for p_i, p in enumerate(ALL_PROFILES):
    m = D['cdl'] == p_i
    for name, col in cls_names.items():
        rows3.append({'CDL':p, 'Method':name, **get_metrics(D['y'][m], D['cls'][col][m])})
df3 = pd.DataFrame(rows3); df3.to_csv(f'{OUT}/TABLE_III_classical.csv', index=False)
print(df3.to_string(index=False))

hdr("SENSITIVITY SWEEP - CEF on CDL-A target")
TARGET_SENS = 0
train_pool = np.where(D['cdl'] != TARGET_SENS)[0]
target_pool = np.where(D['cdl'] == TARGET_SENS)[0]
rng = np.random.default_rng(SEED); rng.shuffle(target_pool)
split_idx = int(0.60 * len(target_pool))
ft_pool_idx, eval_idx = target_pool[:split_idx], target_pool[split_idx:]

sc_sens = Standardizer().fit(Xt[train_pool], D['y'][train_pool])
Xtr_sens = sc_sens.x(Xt[train_pool]); ytr_sens = D['y'][train_pool]
Ctr_sens = C_all[train_pool]; Str_sens = S_all[train_pool]
Xft_sens = sc_sens.x(Xt[ft_pool_idx]); yft_sens = D['y'][ft_pool_idx]
Cft_sens = C_all[ft_pool_idx]; Sft_sens = S_all[ft_pool_idx]
Xev_sens = sc_sens.x(Xt[eval_idx]); yev_sens = D['y'][eval_idx]
Cev_sens = C_all[eval_idx]; Sev_sens = S_all[eval_idx]

crit_sens = huber(sc_sens.ysd)
sens_data = {f: [] for f in SENS_FRACS}
for rep in range(SENS_REPEATS):
    torch.manual_seed(SEED + rep); np.random.seed(SEED + rep)
    m = GridNet(False, True).to(DEVICE)
    tr_split = int(0.9 * len(train_pool)); perm = np.random.permutation(len(train_pool))
    tr_i, va_i = perm[:tr_split], perm[tr_split:]
    m, _ = fit(m, Xtr_sens[tr_i], Ctr_sens[tr_i], Str_sens[tr_i], ytr_sens[tr_i],
               Xtr_sens[va_i], Ctr_sens[va_i], Str_sens[va_i], ytr_sens[va_i],
               crit_sens, EPOCHS_PRE, LR_PRE, BATCH_TRAIN)
    pre_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    for frac in SENS_FRACS:
        n = max(int(frac * len(ft_pool_idx)), 32)
        idx = np.random.default_rng(SEED + rep * 100).choice(len(ft_pool_idx), n, replace=False)
        mt = GridNet(False, True).to(DEVICE); mt.load_state_dict(pre_state)
        mt, _ = fit(mt, Xft_sens[idx], Cft_sens[idx], Sft_sens[idx], yft_sens[idx],
                    Xft_sens, Cft_sens, Sft_sens, yft_sens,
                    crit_sens, EPOCHS_FT, LR_FT, BATCH_FT)
        p = predict(mt, Xev_sens, Cev_sens, Sev_sens)
        sens_data[frac].append(float(np.sqrt(mean_squared_error(yev_sens, p))))
    print(f"  repeat {rep+1}/{SENS_REPEATS} done")

sens_mean = {int(f * len(ft_pool_idx)): float(np.mean(v)) for f, v in sens_data.items()}
sens_std = {int(f * len(ft_pool_idx)): float(np.std(v)) for f, v in sens_data.items()}
sns_x = sorted(sens_mean.keys())

fig, ax = plt.subplots(figsize=(6, 4.2))
mu = [sens_mean[n] for n in sns_x]; sd = [sens_std[n] for n in sns_x]
ax.plot(sns_x, mu, 'o-', color=MODEL_COLORS['CEF'], lw=2, ms=6)
ax.fill_between(sns_x, [m-s for m,s in zip(mu,sd)], [m+s for m,s in zip(mu,sd)],
                alpha=0.2, color=MODEL_COLORS['CEF'])
ax.axvline(1500, color='k', ls='--', lw=1.5, label='chosen budget: 1500 samples')
ax.set_xlabel('Fine-tune samples'); ax.set_ylabel('RMSE (dB)')
ax.set_title('Sensitivity - CEF, CDL-A target'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/Fig3_Sensitivity.png', dpi=150, bbox_inches='tight'); plt.close()
print("  Saved Fig3_Sensitivity.png")

optimal_ft = 1500

hdr("TABLE IV - CROSS-CDL LEAVE-ONE-OUT")
experiments = list(enumerate(ALL_PROFILES))
all_results, all_bins, fitted_models = {}, {}, {}

for exp_id, target in experiments:
    t_idx = ALL_PROFILES.index(target)
    train_profiles = [p for p in ALL_PROFILES if p != target]
    print(f"\n{'#'*60}\n  E{exp_id+1}  Train [{'+'.join(train_profiles)}] -> Test [{target}]  "
          f"{CDL_NAMES[target]}\n{'#'*60}")
    tr_idx = np.where(np.isin(D['cdl'], [ALL_PROFILES.index(p) for p in train_profiles]))[0]
    test_idx, val_idx, ftpool_idx = split_target(t_idx, D, exp_id)

    sc = Standardizer().fit(Xt[tr_idx], D['y'][tr_idx])
    Xtr = sc.x(Xt[tr_idx]); ytr = D['y'][tr_idx]; Ctr = C_all[tr_idx]; Str = S_all[tr_idx]
    Xval = sc.x(Xt[val_idx]); yval = D['y'][val_idx]; Cval = C_all[val_idx]; Sval = S_all[val_idx]
    Xtest = sc.x(Xt[test_idx]); ytest = D['y'][test_idx]; Ctest = C_all[test_idx]; Stest = S_all[test_idx]
    n_ft = min(optimal_ft, len(ftpool_idx))
    Xft = sc.x(Xt[ftpool_idx[:n_ft]]); yft = D['y'][ftpool_idx[:n_ft]]
    Cft = C_all[ftpool_idx[:n_ft]]; Sft = S_all[ftpool_idx[:n_ft]]

    all_results[exp_id], all_bins[exp_id] = {}, {}
    crit = huber(sc.ysd)

    torch.manual_seed(SEED); np.random.seed(SEED)
    perm = np.random.permutation(len(Xtr)); tr_split = int(0.9 * len(Xtr))
    m = GridNet(False, True).to(DEVICE)
    m, _ = fit(m, Xtr[perm[:tr_split]], Ctr[perm[:tr_split]], Str[perm[:tr_split]], ytr[perm[:tr_split]],
               Xtr[perm[tr_split:]], Ctr[perm[tr_split:]], Str[perm[tr_split:]], ytr[perm[tr_split:]],
               crit, EPOCHS_PRE, LR_PRE, BATCH_TRAIN)
    pre_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    mt = GridNet(False, True).to(DEVICE); mt.load_state_dict(pre_state)
    mt, _ = fit(mt, Xft, Cft, Sft, yft, Xval, Cval, Sval, yval, crit, EPOCHS_FT, LR_FT, BATCH_FT)
    p = predict(mt, Xtest, Ctest, Stest)
    mm = get_metrics(ytest, p); ci = bootstrap_ci(ytest, p)
    mm['RMSE_CI_lo'], mm['RMSE_CI_hi'] = ci
    all_results[exp_id]['CEF'] = mm
    all_bins[exp_id]['CEF'] = bin_metrics(ytest, p)
    fitted_models[(exp_id, 'CEF')] = (mt, 'nn', sc)
    print(f"    {'CEF':<12} RMSE={mm['RMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]  "
          f"MAE={mm['MAE']:.4f}  Bias={mm['Bias']:+.4f}  R2={mm['R2']:.4f}")

    for name in ['MLP-16', 'Dilated-CNN', 'ResNet1D']:
        torch.manual_seed(SEED); np.random.seed(SEED)
        if name == 'MLP-16': m = MLP16().to(DEVICE); use_fusion_input = False
        elif name == 'Dilated-CNN': m = DilatedCNN().to(DEVICE); use_fusion_input = True
        else: m = ResNet1D().to(DEVICE); use_fusion_input = True
        Str_b = Str if use_fusion_input else zeros4(len(Str))
        Sft_b = Sft if use_fusion_input else zeros4(len(Sft))
        Sval_b = Sval if use_fusion_input else zeros4(len(Sval))
        Stest_b = Stest if use_fusion_input else zeros4(len(Stest))
        m, _ = fit(m, Xtr[perm[:tr_split]], Ctr[perm[:tr_split]], Str_b[perm[:tr_split]], ytr[perm[:tr_split]],
                   Xtr[perm[tr_split:]], Ctr[perm[tr_split:]], Str_b[perm[tr_split:]], ytr[perm[tr_split:]],
                   crit, EPOCHS_PRE, LR_PRE, BATCH_TRAIN)
        pre_state_b = {k: t.detach().clone() for k, t in m.state_dict().items()}
        if name == 'MLP-16': mb = MLP16().to(DEVICE)
        elif name == 'Dilated-CNN': mb = DilatedCNN().to(DEVICE)
        else: mb = ResNet1D().to(DEVICE)
        mb.load_state_dict(pre_state_b)
        mb, _ = fit(mb, Xft, Cft, Sft_b, yft, Xval, Cval, Sval_b, yval, crit, EPOCHS_FT, LR_FT, BATCH_FT)
        p = predict(mb, Xtest, Ctest, Stest_b)
        mm = get_metrics(ytest, p); ci = bootstrap_ci(ytest, p)
        mm['RMSE_CI_lo'], mm['RMSE_CI_hi'] = ci
        all_results[exp_id][name] = mm
        all_bins[exp_id][name] = bin_metrics(ytest, p)
        fitted_models[(exp_id, name)] = (mb, 'nn', sc)
        print(f"    {name:<12} RMSE={mm['RMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]  "
              f"MAE={mm['MAE']:.4f}  Bias={mm['Bias']:+.4f}  R2={mm['R2']:.4f}")

    if HAS_XGB:
        X_train = np.concatenate([Xtr[perm[:tr_split]], Ctr[perm[:tr_split]], Str[perm[:tr_split]]], 1)
        X_val_x = np.concatenate([Xtr[perm[tr_split:]], Ctr[perm[tr_split:]], Str[perm[tr_split:]]], 1)
        xgb = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist',
                            early_stopping_rounds=30, random_state=SEED, n_jobs=4)
        xgb.fit(X_train, ytr[perm[:tr_split]], eval_set=[(X_val_x, ytr[perm[tr_split:]])], verbose=False)
        X_test_flat = np.concatenate([Xtest, Ctest, Stest], 1)
        p = xgb.predict(X_test_flat)
        mm = get_metrics(ytest, p); ci = bootstrap_ci(ytest, p)
        mm['RMSE_CI_lo'], mm['RMSE_CI_hi'] = ci
        all_results[exp_id]['XGBoost'] = mm
        all_bins[exp_id]['XGBoost'] = bin_metrics(ytest, p)
        fitted_models[(exp_id, 'XGBoost')] = (xgb, 'xgb', sc)
        print(f"    {'XGBoost':<12} RMSE={mm['RMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]  "
              f"MAE={mm['MAE']:.4f}  Bias={mm['Bias']:+.4f}  R2={mm['R2']:.4f}")

kept_models = ['CEF', 'MLP-16', 'Dilated-CNN', 'ResNet1D'] + (['XGBoost'] if HAS_XGB else [])
tableIV = [dict(Test_CDL=t, Model=v, **all_results[e][v])
           for e, t in experiments for v in kept_models if v in all_results[e]]
df4 = pd.DataFrame(tableIV); df4.to_csv(f'{OUT}/TABLE_IV_crosscdl.csv', index=False)
print('\n' + df4.round(4).to_string(index=False))

hdr("TABLE IV-b - PARAMETERS AND INFERENCE LATENCY")

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def measure_latency(model, n_warmup=30, n_measure=300):
    model.eval()
    x = torch.zeros(1, 16).to(DEVICE); c = torch.zeros(1, 4).to(DEVICE); s = torch.zeros(1, 4).to(DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup): model(x, c, s)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_measure): model(x, c, s)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_measure * 1000.0

lat_rows = []
for name, builder in [
    ('CEF', lambda: GridNet(False, True).to(DEVICE)),
    ('MLP-16', lambda: MLP16().to(DEVICE)),
    ('ResNet1D', lambda: ResNet1D().to(DEVICE)),
    ('Dilated-CNN', lambda: DilatedCNN().to(DEVICE))]:
    m = builder()
    n_params = count_params(m)
    lat_ms = measure_latency(m)
    lat_rows.append({'Model': name, 'Params': n_params, 'Latency_ms': lat_ms})
    print(f"  {name:<12} Params={n_params:>8,}   Latency={lat_ms:.3f} ms/sample")

if HAS_XGB:
    xgb_ref = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                            random_state=SEED, n_jobs=4)
    xgb_ref.fit(np.random.randn(100, 24), np.random.randn(100))
    t0 = time.perf_counter()
    for _ in range(300): xgb_ref.predict(np.random.randn(1, 24))
    lat_ms = (time.perf_counter() - t0) / 300 * 1000.0
    n_est = 600 * 2 ** 6
    lat_rows.append({'Model': 'XGBoost', 'Params': n_est, 'Latency_ms': lat_ms})
    print(f"  {'XGBoost':<12} Params~{n_est:>7,}   Latency={lat_ms:.3f} ms/sample")

pd.DataFrame(lat_rows).to_csv(f'{OUT}/TABLE_IVb_latency.csv', index=False)

hdr("TABLE V - 2x2 GRID (FUSION x EMBEDDING)")

def run_cell(use_cdl, use_fusion, exp_id, target, t_idx):
    tr_idx = np.where(D['cdl'] != t_idx)[0]
    test_idx, val_idx, ftpool_idx = split_target(t_idx, D, exp_id)
    sc = Standardizer().fit(Xt[tr_idx], D['y'][tr_idx])
    n_ft = min(optimal_ft, len(ftpool_idx))
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = GridNet(use_cdl, use_fusion).to(DEVICE)
    Xtr_, ytr_, Ctr_, Str_ = sc.x(Xt[tr_idx]), D['y'][tr_idx], C_all[tr_idx], S_all[tr_idx]
    perm = np.random.permutation(len(Xtr_)); tr_split = int(0.9 * len(Xtr_))
    crit = huber(sc.ysd)
    m, _ = fit(m, Xtr_[perm[:tr_split]], Ctr_[perm[:tr_split]], Str_[perm[:tr_split]], ytr_[perm[:tr_split]],
               Xtr_[perm[tr_split:]], Ctr_[perm[tr_split:]], Str_[perm[tr_split:]], ytr_[perm[tr_split:]],
               crit, EPOCHS_PRE, LR_PRE, BATCH_TRAIN)
    Xft_ = sc.x(Xt[ftpool_idx[:n_ft]]); yft_ = D['y'][ftpool_idx[:n_ft]]
    Cft_ = C_all[ftpool_idx[:n_ft]]; Sft_ = S_all[ftpool_idx[:n_ft]]
    Xva_ = sc.x(Xt[val_idx]); yva_ = D['y'][val_idx]; Cva_ = C_all[val_idx]; Sva_ = S_all[val_idx]
    m, _ = fit(m, Xft_, Cft_, Sft_, yft_, Xva_, Cva_, Sva_, yva_, crit, EPOCHS_FT, LR_FT, BATCH_FT)
    p = predict(m, sc.x(Xt[test_idx]), C_all[test_idx], S_all[test_idx])
    return get_metrics(D['y'][test_idx], p)

grid_cells = {'A_feat_noCDL': (False, False), 'B_feat_CDL': (True, False),
              'C_fusion_noCDL': (False, True), 'D_fusion_CDL': (True, True)}
grid_results = {c: [] for c in grid_cells}
for exp_id, target in experiments:
    t_idx = ALL_PROFILES.index(target)
    for cell, (use_cdl, use_fusion) in grid_cells.items():
        mm = run_cell(use_cdl, use_fusion, exp_id, target, t_idx)
        grid_results[cell].append(mm)
        print(f"  {target}  {cell:<16} RMSE={mm['RMSE']:.4f}  Bias={mm['Bias']:+.4f}  R2={mm['R2']:.4f}")

grid_rows = []
for cell in grid_cells:
    vals = grid_results[cell]
    grid_rows.append({'Cell': cell,
                      'RMSE': float(np.mean([v['RMSE'] for v in vals])),
                      'MAE':  float(np.mean([v['MAE']  for v in vals])),
                      'Bias': float(np.mean([v['Bias'] for v in vals])),
                      'R2':   float(np.mean([v['R2']   for v in vals]))})
df5 = pd.DataFrame(grid_rows)
rmse = {r['Cell']: r['RMSE'] for r in grid_rows}
fusion_delta = rmse['D_fusion_CDL'] - rmse['B_feat_CDL']
emb_delta_feat = rmse['B_feat_CDL'] - rmse['A_feat_noCDL']
emb_delta_fused = rmse['D_fusion_CDL'] - rmse['C_fusion_noCDL']

deltas = pd.DataFrame([
    {'Delta': 'Fusion delta (D - B)', 'RMSE_change_dB': fusion_delta,
     'Interpretation': 'value of classical-estimator fusion, given embedding'},
    {'Delta': 'Embedding delta, features-only (B - A)', 'RMSE_change_dB': emb_delta_feat,
     'Interpretation': 'embedding effect when NOT competing with fusion'},
    {'Delta': 'Embedding delta, fused (D - C)', 'RMSE_change_dB': emb_delta_fused,
     'Interpretation': 'embedding effect WITH fusion present'}])
df5.to_csv(f'{OUT}/TABLE_V_grid.csv', index=False)
deltas.to_csv(f'{OUT}/TABLE_V_deltas.csv', index=False)
print('\n' + df5.round(4).to_string(index=False))
print('\n' + deltas.round(4).to_string(index=False))

hdr("CONSISTENCY CHECK")
mlp16_mean = float(np.mean([all_results[e]['MLP-16']['RMSE'] for e, _ in experiments]))
cellB_mean = rmse['B_feat_CDL']
gap = abs(mlp16_mean - cellB_mean)
print(f"  MLP-16 mean RMSE : {mlp16_mean:.4f} dB")
print(f"  Cell B  mean RMSE : {cellB_mean:.4f} dB")
print(f"  |gap| = {gap:.4f} dB")

hdr("TABLE V-b - PER-ESTIMATOR ABLATION")
S_NAMES = ['LS', 'ML', 'EVM', 'DD']

def run_per_est(exp_id, target, t_idx, s_idx_keep):
    tr_idx = np.where(D['cdl'] != t_idx)[0]
    test_idx, val_idx, ftpool_idx = split_target(t_idx, D, exp_id)
    sc = Standardizer().fit(Xt[tr_idx], D['y'][tr_idx])
    n_ft = min(optimal_ft, len(ftpool_idx))
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = GridNet(False, True, s_idx=s_idx_keep).to(DEVICE)
    Xtr_, ytr_, Ctr_, Str_ = sc.x(Xt[tr_idx]), D['y'][tr_idx], C_all[tr_idx], S_all[tr_idx]
    perm = np.random.permutation(len(Xtr_)); tr_split = int(0.9 * len(Xtr_))
    crit = huber(sc.ysd)
    m, _ = fit(m, Xtr_[perm[:tr_split]], Ctr_[perm[:tr_split]], Str_[perm[:tr_split]], ytr_[perm[:tr_split]],
               Xtr_[perm[tr_split:]], Ctr_[perm[tr_split:]], Str_[perm[tr_split:]], ytr_[perm[tr_split:]],
               crit, EPOCHS_PRE, LR_PRE, BATCH_TRAIN)
    Xft_ = sc.x(Xt[ftpool_idx[:n_ft]]); yft_ = D['y'][ftpool_idx[:n_ft]]
    Cft_ = C_all[ftpool_idx[:n_ft]]; Sft_ = S_all[ftpool_idx[:n_ft]]
    Xva_ = sc.x(Xt[val_idx]); yva_ = D['y'][val_idx]; Cva_ = C_all[val_idx]; Sva_ = S_all[val_idx]
    m, _ = fit(m, Xft_, Cft_, Sft_, yft_, Xva_, Cva_, Sva_, yva_, crit, EPOCHS_FT, LR_FT, BATCH_FT)
    p = predict(m, sc.x(Xt[test_idx]), C_all[test_idx], S_all[test_idx])
    return get_metrics(D['y'][test_idx], p)

per_est_results = {'Full': [all_results[e]['CEF'] for e, _ in experiments],
                   'drop_LS': [], 'drop_ML': [], 'drop_EVM': [], 'drop_DD': []}
for exp_id, target in experiments:
    t_idx = ALL_PROFILES.index(target)
    for drop_idx, drop_name in [(0,'drop_LS'), (1,'drop_ML'), (2,'drop_EVM'), (3,'drop_DD')]:
        keep = [i for i in range(4) if i != drop_idx]
        mm = run_per_est(exp_id, target, t_idx, keep)
        per_est_results[drop_name].append(mm)
        print(f"  {target}  w/o {S_NAMES[drop_idx]:<4}  RMSE={mm['RMSE']:.4f}  "
              f"Bias={mm['Bias']:+.4f}  R2={mm['R2']:.4f}")

per_est_rows = []
for variant, vals in per_est_results.items():
    per_est_rows.append({'Variant': variant,
                         'RMSE': float(np.mean([v['RMSE'] for v in vals])),
                         'MAE':  float(np.mean([v['MAE']  for v in vals])),
                         'Bias': float(np.mean([v['Bias'] for v in vals])),
                         'R2':   float(np.mean([v['R2']   for v in vals]))})
df5b = pd.DataFrame(per_est_rows); df5b.to_csv(f'{OUT}/TABLE_Vb_per_estimator.csv', index=False)
print('\n' + df5b.round(4).to_string(index=False))

full_rmse = df5b[df5b['Variant'] == 'Full']['RMSE'].values[0]
print(f"\n  Impact of removing each estimator (positive = useful):")
for variant, drop_name in [('drop_LS','LS'), ('drop_ML','ML'), ('drop_EVM','EVM'), ('drop_DD','DD')]:
    r = df5b[df5b['Variant'] == variant]['RMSE'].values[0]
    print(f"    w/o {drop_name:<4}  RMSE={r:.4f}  Δ = {r - full_rmse:+.4f} dB")

hdr("TABLE VI - BIAS PER SNR BIN (CEF)")
bin_rows = []
for exp_id, target in experiments:
    for b in all_bins[exp_id]['CEF']:
        bin_rows.append(dict(Test_CDL=target, **b))
df6 = pd.DataFrame(bin_rows); df6.to_csv(f'{OUT}/TABLE_VI_bias_bins.csv', index=False)
print(df6.round(4).to_string(index=False))

hdr("TABLE VI-b - LOSS COMPARISON (CEF, CDL-A target)")

def run_loss_variant(loss_name, exp_id=0, target='CDL-A'):
    t_idx = ALL_PROFILES.index(target)
    tr_idx = np.where(D['cdl'] != t_idx)[0]
    test_idx, val_idx, ftpool_idx = split_target(t_idx, D, exp_id)
    sc = Standardizer().fit(Xt[tr_idx], D['y'][tr_idx])
    n_ft = min(optimal_ft, len(ftpool_idx))
    crit = make_loss(loss_name, 3.0, sc.ysd)
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = GridNet(False, True).to(DEVICE)
    Xtr_, ytr_, Ctr_, Str_ = sc.x(Xt[tr_idx]), D['y'][tr_idx], C_all[tr_idx], S_all[tr_idx]
    perm = np.random.permutation(len(Xtr_)); tr_split = int(0.9 * len(Xtr_))
    m, _ = fit(m, Xtr_[perm[:tr_split]], Ctr_[perm[:tr_split]], Str_[perm[:tr_split]], ytr_[perm[:tr_split]],
               Xtr_[perm[tr_split:]], Ctr_[perm[tr_split:]], Str_[perm[tr_split:]], ytr_[perm[tr_split:]],
               crit, EPOCHS_PRE, LR_PRE, BATCH_TRAIN)
    Xft_ = sc.x(Xt[ftpool_idx[:n_ft]]); yft_ = D['y'][ftpool_idx[:n_ft]]
    Cft_ = C_all[ftpool_idx[:n_ft]]; Sft_ = S_all[ftpool_idx[:n_ft]]
    Xva_ = sc.x(Xt[val_idx]); yva_ = D['y'][val_idx]; Cva_ = C_all[val_idx]; Sva_ = S_all[val_idx]
    m, _ = fit(m, Xft_, Cft_, Sft_, yft_, Xva_, Cva_, Sva_, yva_, crit, EPOCHS_FT, LR_FT, BATCH_FT)
    p = predict(m, sc.x(Xt[test_idx]), C_all[test_idx], S_all[test_idx])
    return get_metrics(D['y'][test_idx], p)

loss_rows = []
for loss_name in ['mse', 'huber', 'cauchy']:
    mm = run_loss_variant(loss_name)
    loss_rows.append({'Loss': loss_name.upper(), **mm})
    print(f"  {loss_name.upper():<8} RMSE={mm['RMSE']:.4f}  MAE={mm['MAE']:.4f}  "
          f"Bias={mm['Bias']:+.4f}  R2={mm['R2']:.4f}")
df6b = pd.DataFrame(loss_rows); df6b.to_csv(f'{OUT}/TABLE_VIb_loss.csv', index=False)

hdr("TABLE VII - OOD EVALUATION")
ood_rows = []
for exp_id, target in experiments:
    t_idx = ALL_PROFILES.index(target)
    ood_idx = np.where(O['cdl'] == t_idx)[0]
    if len(ood_idx) == 0: continue
    S_ood_input = So_all[ood_idx]
    for vname in kept_models:
        if (exp_id, vname) not in fitted_models: continue
        obj, kind, sc = fitted_models[(exp_id, vname)]
        if kind == 'xgb':
            X_ood = np.concatenate([sc.x(Oxt[ood_idx]), Co_all[ood_idx], S_ood_input], 1)
            p = obj.predict(X_ood)
        else:
            p = predict(obj, sc.x(Oxt[ood_idx]), Co_all[ood_idx], S_ood_input)
        mm = get_metrics(O['y'][ood_idx], p)
        ood_rows.append({'Test_CDL': target, 'Model': vname,
                         'OOD_RMSE': mm['RMSE'], 'OOD_MAE': mm['MAE'],
                         'OOD_Bias': mm['Bias'], 'OOD_R2': mm['R2'],
                         'ID_RMSE': all_results[exp_id][vname]['RMSE'],
                         'Delta': mm['RMSE'] - all_results[exp_id][vname]['RMSE']})
df7 = pd.DataFrame(ood_rows); df7.to_csv(f'{OUT}/TABLE_VII_ood.csv', index=False)
print(df7.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.5))
n_v = len(kept_models); w = 0.8 / n_v
for i, vn in enumerate(kept_models):
    vals = [all_results[e][vn]['RMSE'] if vn in all_results[e] else np.nan for e, _ in experiments]
    ax.bar(np.arange(4) + (i - n_v/2 + 0.5) * w, vals, w,
           color=MODEL_COLORS.get(vn, '#888'),
           edgecolor='black', linewidth=(1.8 if vn == 'CEF' else 0.4), label=vn)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(ALL_PROFILES)
ax.set_ylabel('RMSE (dB)'); ax.set_title('Cross-CDL RMSE')
ax.legend(fontsize=8, ncol=len(kept_models)); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/Fig2_CrossCDL.png', dpi=150, bbox_inches='tight'); plt.close()
print('\n  Saved Fig2_CrossCDL.png')
