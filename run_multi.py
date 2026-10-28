#!/usr/bin/env python3
"""Multi-channel StressMAE (chest ECG, EDA, EMG, RESP, TEMP, ACC; 100 Hz, 5 s windows, 0.5 s patches).
Pretext = masked patch reconstruction (ratio 0.6) of all channels + cross-modal masking: for half of the windows one
whole channel is removed from the input and must be rebuilt from the other channels. Per-channel loss weights 1/var.
  pretrain : one encoder per subject group, trained on the unlabelled recordings of the other 12 subjects
  loso     : leave-one-subject-out fine-tuning (12 train / 2 val), saves probabilities of test + val recordings
  cv       : literature-style stratified 10-fold inside each subject (comparison only), frozen encoder + linear/SVM head
"""
import argparse, copy, json, os, time
import numpy as np, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import run_experiments as rx
from run_experiments import SUBJECTS, GROUPS, SEED, WIN, PATCH, set_seed, batches, robust_norm, group_of, metrics

C = 6
MDIR = os.path.join(rx.CACHE_DIR, "multi")
ENC = os.path.join(rx.RES_DIR, "encoders_multi")


def load_multi(sid, classes=(1, 2, 3)):
    z = np.load(os.path.join(MDIR, f"{sid}.npz")); S, lab = z["sig"], z["lab"]
    def win(starts):
        X = S[:, starts[:, None] + np.arange(WIN)].transpose(1, 0, 2).copy()     # (n, C, WIN)
        X[:, 0] = robust_norm(X[:, 0]); return X.astype(np.float32)
    starts = np.arange(0, S.shape[1] - WIN + 1, WIN)
    seg = lab[starts[:, None] + np.arange(WIN)]
    pure = (seg == seg[:, :1]).all(1) & np.isin(seg[:, 0], classes)
    return {"X": win(starts[pure]), "y": seg[pure, 0].astype(int) - 1, "t": starts[pure], "U": win(starts)}


class MultiMAE(nn.Module):
    def __init__(self, dim=128, enc_layers=4, dec_layers=2, heads=8):
        super().__init__()
        self.n, self.dim = WIN // PATCH, dim
        self.backbone = nn.Sequential(nn.Conv1d(C, 48, 9, padding=4), nn.GELU(),
                                      nn.Conv1d(48, 64, 5, stride=2, padding=2), nn.GELU())
        self.patch_embed = nn.Conv1d(64, dim, PATCH // 2, stride=PATCH // 2)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.n, dim)); self.pos_dec = nn.Parameter(torch.zeros(1, self.n, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = lambda: nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer(), enc_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerEncoder(layer(), dec_layers, enable_nested_tensor=False)
        self.reconstruct = nn.Linear(dim, PATCH * C)
        self.register_buffer("w", torch.ones(C))
        for p in (self.pos_enc, self.pos_dec, self.mask_token):
            nn.init.normal_(p, std=0.02)

    def tokens(self, x):
        return self.patch_embed(self.backbone(x)).transpose(1, 2) + self.pos_enc

    def encode(self, x):
        return self.encoder(self.tokens(x)).mean(1)

    def mae_loss(self, x, ratio=0.6, p_drop=0.5):
        b = x.shape[0]
        drop = torch.where(torch.rand(b) < p_drop, torch.randint(0, C, (b,)), torch.full((b,), -1))
        dmask = torch.zeros(b, C, dtype=torch.bool); rows = drop >= 0; dmask[rows, drop[rows]] = True
        t = self.tokens(x.masked_fill(dmask[:, :, None], 0.0))
        _, n, d = t.shape; k = int(round(n * ratio))
        perm = torch.rand(b, n).argsort(1); mask_idx, keep_idx = perm[:, :k], perm[:, k:]
        enc = self.encoder(torch.gather(t, 1, keep_idx[..., None].expand(-1, -1, d)))
        full = self.mask_token.expand(b, n, d).clone().scatter(1, keep_idx[..., None].expand(-1, -1, d), enc)
        rec = self.reconstruct(self.decoder(full + self.pos_dec)).view(b, n, C, PATCH)
        target = x.view(b, C, n, PATCH).transpose(1, 2)                          # (b, n, C, PATCH)
        tm = torch.zeros(b, n, dtype=torch.bool).scatter(1, mask_idx, True)
        sel = tm[:, :, None] | dmask[:, None, :]                                 # masked patches + dropped channel
        err = ((rec - target) ** 2).mean(-1) * self.w                            # (b, n, C)
        return err[sel].mean()


class Head(nn.Module):
    def __init__(self, mae, n_cls=3):
        super().__init__(); self.mae = mae
        self.head = nn.Sequential(nn.Linear(mae.dim, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.head(self.mae.encode(x))


def pretrain(U, epochs, log, bs=128, lr=1e-3, warm_ep=2):
    mae = MultiMAE()
    mae.w = torch.tensor(1.0 / (U.transpose(1, 0, 2).reshape(C, -1).var(1) + 1e-6), dtype=torch.float32)
    mae.w = mae.w / mae.w.mean()
    opt = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.05)
    spe = int(np.ceil(len(U) / bs)); tot, warm = epochs * spe, warm_ep * spe
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                            0.5 * (1 + np.cos(np.pi * (s - warm) / (tot - warm))))
    Ut = torch.from_numpy(U); curve = []
    for ep in range(epochs):
        mae.train(); tl = 0; nb = 0; t0 = time.time()
        for bi in batches(len(Ut), bs, True):
            loss = mae.mae_loss(Ut[bi]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(mae.parameters(), 1.0); opt.step(); sch.step(); tl += loss.item(); nb += 1
        curve.append(tl / nb); log(f"  pretrain ep {ep + 1}/{epochs} loss {tl / nb:.4f} ({time.time() - t0:.0f}s)")
    return mae, curve


@torch.no_grad()
def proba(model, X, bs=256):
    model.eval(); Xt = torch.from_numpy(X)
    return np.concatenate([torch.softmax(model(Xt[bi]), 1).numpy() for bi in batches(len(Xt), bs, False)])


@torch.no_grad()
def embed(mae, X, bs=256):
    mae.eval(); Xt = torch.from_numpy(X)
    return np.concatenate([mae.encode(Xt[bi]).numpy() for bi in batches(len(Xt), bs, False)])


@torch.no_grad()
def proba_tta(model, X, bs=256):
    """test-time sensor averaging: mean of the predictions with all sensors and with each sensor removed in turn"""
    outs = [proba(model, X, bs)]
    for c in range(C):
        Xc = X.copy(); Xc[:, c] = 0.0; outs.append(proba(model, Xc, bs))
    return np.mean(outs, 0)


def drop_channels(x, p=0.5):
    """sensor dropout for fine-tuning: with probability p, one random sensor of the window is set to zero"""
    b = x.shape[0]; rows = torch.rand(b) < p
    if rows.any():
        x = x.clone(); x[rows, torch.randint(0, C, (int(rows.sum()),))] = 0.0
    return x


def fit(model, Xtr, ytr, Xva, yva, epochs, lr, patience=6, bs=128, aug=False):
    k = model.head[-1].out_features
    cnt = np.bincount(ytr, minlength=k).astype(float)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (k * np.maximum(cnt, 1)), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr); best, state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        for bi in batches(len(Xt), bs, True):
            xb = drop_channels(Xt[bi]) if aug else Xt[bi]
            loss = crit(model(xb), yt[bi]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        f = f1_score(yva, proba(model, Xva).argmax(1), average="macro")
        if f > best:
            best, state, bad = f, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); return model, ep + 1


def feats(X):
    """simple per-window statistics of channels 1-5 (EDA, EMG, RESP, TEMP, ACC): mean, std, min, max, slope"""
    s = X[:, 1:]; tt = np.linspace(-1, 1, s.shape[-1])
    return np.concatenate([s.mean(-1), s.std(-1), s.min(-1), s.max(-1), (s * tt).mean(-1) / (tt ** 2).mean()], 1)


def enc_path(g):
    return os.path.join(ENC, f"multi_group{g}.pt")


def load_enc(sid):
    mae = MultiMAE(); mae.load_state_dict(torch.load(enc_path(group_of(sid)))); return mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pretrain", "pretrain_all", "loso", "loso_cd", "cv", "pooled", "pooled_ft", "lowlabel"])
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--groups", type=int, nargs="*", default=list(range(5)))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); os.makedirs(ENC, exist_ok=True)
    logf = open(os.path.join(rx.RES_DIR, f"log_multi_{a.mode}{a.tag}.txt"), "a")
    def log(m):
        line = time.strftime("%H:%M:%S ") + m; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    cls = (1, 2, 3, 4) if a.classes == 4 else (1, 2, 3)
    data = {s: load_multi(s, cls) for s in SUBJECTS}
    if a.classes == 2:   # dedicated stress vs non-stress model (baseline and amusement = non-stress)
        for s in SUBJECTS:
            data[s]["y"] = (data[s]["y"] == 1).astype(int)
        cls = (0, 1)

    if a.mode == "pretrain":
        for g in a.groups:
            if os.path.exists(enc_path(g)):
                continue
            set_seed(SEED + g); others = [s for s in SUBJECTS if s not in GROUPS[g]]
            U = np.concatenate([data[s]["U"] for s in others]); log(f"[multi pretrain group {g}] {len(U)} windows")
            t0 = time.time(); mae, curve = pretrain(U, a.epochs, log)
            torch.save(mae.state_dict(), enc_path(g))
            json.dump({"curve": curve, "minutes": (time.time() - t0) / 60}, open(enc_path(g)[:-3] + ".json", "w"))
        return

    if a.mode == "pretrain_all":   # for the NON-STRICT protocols only: unlabelled recordings of all 15 subjects
        p = os.path.join(ENC, "multi_all.pt")
        if not os.path.exists(p):
            set_seed(SEED); U = np.concatenate([data[s]["U"] for s in SUBJECTS]); log(f"[multi pretrain ALL] {len(U)} windows")
            t0 = time.time(); mae, curve = pretrain(U, a.epochs, log); torch.save(mae.state_dict(), p)
            json.dump({"curve": curve, "minutes": (time.time() - t0) / 60}, open(p[:-3] + ".json", "w"))
        return

    if a.mode == "loso":
        out = os.path.join(rx.RES_DIR, "multi_loso"); os.makedirs(out, exist_ok=True)
        for i, test in enumerate(SUBJECTS):
            if (a.subjects and test not in a.subjects) or os.path.exists(os.path.join(out, f"{test}.npz")):
                continue
            t0 = time.time(); others = [s for s in SUBJECTS if s != test]
            val = [others[(i + j) % len(others)] for j in (0, 7)]; train = [s for s in others if s not in val]
            Xtr = np.concatenate([data[s]["X"] for s in train]); ytr = np.concatenate([data[s]["y"] for s in train])
            Xva = np.concatenate([data[s]["X"] for s in val]); yva = np.concatenate([data[s]["y"] for s in val])
            mae = load_enc(test); res = {}
            runs = [(f"StressMAE-multi_s{k}", lambda: Head(copy.deepcopy(mae)), 1e-4, k) for k in a.seeds]
            runs += [("No-pretrain-multi_s0", lambda: Head(MultiMAE()), 3e-4, 0)]
            for name, make, lr, k in runs:
                set_seed(SEED + 1000 * k); model, ep = fit(make(), Xtr, ytr, Xva, yva, 20, lr)
                res[name] = proba(model, data[test]["U"])
                for v in val:
                    res[f"{name}|val_{v}"] = proba(model, data[v]["U"])
                idx = data[test]["t"] // WIN
                log(f"  {test} {name}: {ep} ep, 5 s F1 {f1_score(data[test]['y'], res[name][idx].argmax(1), average='macro'):.3f}")
            # multi-sensor feature baselines (ECG HRV features + statistics of the other channels), same calibration
            H = {s: v for s, v in np.load(os.path.join(rx.CACHE_DIR, "hrv_U.npz")).items()}
            sn = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-6)
            Fall = {s: np.hstack([sn(H[s]), feats(data[s]["U"])]) for s in SUBJECTS}
            fit_s = train + val
            Ftr = np.concatenate([Fall[s][data[s]["t"] // WIN] for s in fit_s]); yf = np.concatenate([data[s]["y"] for s in fit_s])
            for name, clf in [("Features-multi+LR", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                              ("Features-multi+RF", RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=a.threads, random_state=SEED))]:
                clf.fit(Ftr, yf); res[name] = clf.predict_proba(Fall[test])
                log(f"  {test} {name}: 5 s F1 {f1_score(data[test]['y'], res[name][data[test]['t'] // WIN].argmax(1), average='macro'):.3f}")
            np.savez_compressed(os.path.join(out, f"{test}.npz"), **{k.replace(' ', '_'): v for k, v in res.items()})
            log(f"[multi LOSO {test}] {(time.time() - t0) / 60:.1f} min")
        return

    if a.mode == "loso_cd":   # StressMAE-multi with sensor-dropout fine-tuning and test-time sensor averaging
        aug = "noaug" not in a.tag
        out = os.path.join(rx.RES_DIR, ("multi_loso_cd" if aug else "multi_loso_plain") + ("" if a.classes == 3 else str(a.classes)))
        os.makedirs(out, exist_ok=True)
        for i, test in enumerate(SUBJECTS):
            if (a.subjects and test not in a.subjects) or os.path.exists(os.path.join(out, f"{test}.npz")):
                continue
            t0 = time.time(); others = [s for s in SUBJECTS if s != test]
            val = [others[(i + j) % len(others)] for j in (0, 7)]; train = [s for s in others if s not in val]
            Xtr = np.concatenate([data[s]["X"] for s in train]); ytr = np.concatenate([data[s]["y"] for s in train])
            Xva = np.concatenate([data[s]["X"] for s in val]); yva = np.concatenate([data[s]["y"] for s in val])
            mae = load_enc(test); res = {}
            for k in a.seeds:
                set_seed(SEED + 1000 * k)
                model, ep = fit(Head(copy.deepcopy(mae), len(cls)), Xtr, ytr, Xva, yva, 20, 1e-4, aug=aug)
                name = f"StressMAE-multi-cd_s{k}" if aug else f"StressMAE-multi_s{k}"; res[name] = proba_tta(model, data[test]["U"])
                for v in val:
                    res[f"{name}|val_{v}"] = proba_tta(model, data[v]["U"])
                log(f"  {test} {name}: {ep} ep, 5 s F1 {f1_score(data[test]['y'], res[name][data[test]['t'] // WIN].argmax(1), average='macro'):.3f}")
            np.savez_compressed(os.path.join(out, f"{test}.npz"), **res)
            log(f"[multi LOSO-cd {test}] {(time.time() - t0) / 60:.1f} min")
        return

    if a.mode == "lowlabel":   # strict LOSO with a fraction of the labels (training AND validation subjects subsampled)
        out = os.path.join(rx.RES_DIR, "multi_lowlabel"); os.makedirs(out, exist_ok=True)
        fracs = (0.01, 0.05, 0.10, 0.25)
        for i, test in enumerate(SUBJECTS):
            if (a.subjects and test not in a.subjects) or os.path.exists(os.path.join(out, f"{test}.json")):
                continue
            others = [s for s in SUBJECTS if s != test]
            val = [others[(i + j) % len(others)] for j in (0, 7)]; train = [s for s in others if s not in val]
            mae = load_enc(test); idx = data[test]["t"] // WIN; res = {}
            for fr in fracs:
                rng = np.random.default_rng(SEED)
                def sub(sids):
                    X, Y = [], []
                    for s in sids:     # stratified per subject and class, at least one window per class
                        y = data[s]["y"]
                        for c in np.unique(y):
                            ii = np.where(y == c)[0]; k = max(1, int(round(fr * len(ii))))
                            sel = rng.choice(ii, k, replace=False); X.append(data[s]["X"][sel]); Y.append(y[sel])
                    return np.concatenate(X), np.concatenate(Y)
                Xtr, ytr = sub(train); Xva, yva = sub(val)
                for name, make, lr in [("StressMAE-multi", lambda: Head(copy.deepcopy(mae)), 1e-4), ("No-pretrain-multi", lambda: Head(MultiMAE()), 3e-4)]:
                    set_seed(SEED); model, ep = fit(make(), Xtr, ytr, Xva, yva, 40, lr, patience=8, bs=32)
                    P = proba(model, data[test]["X"]); f = f1_score(data[test]["y"], P.argmax(1), average="macro")
                    res[f"{name}|{fr}"] = {"f1": f, "acc": float((P.argmax(1) == data[test]["y"]).mean()), "n_train": int(len(ytr))}
                    log(f"  {test} frac {fr} {name}: {len(ytr)} labelled windows, {ep} ep, F1 {f:.3f}")
            json.dump(res, open(os.path.join(out, f"{test}.json"), "w"), indent=1)
        return

    if a.mode == "pooled_ft":   # NON-STRICT pooled protocol, full fine-tuning, the study's own single split
        from sklearn.model_selection import train_test_split
        X = np.concatenate([data[s]["X"] for s in SUBJECTS]); y = np.concatenate([data[s]["y"] for s in SUBJECTS])
        idx = np.arange(len(y)); tr, te = train_test_split(idx, test_size=a.test_frac, stratify=y, random_state=SEED)
        tr2, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
        Ps = []
        for k in a.seeds:
            mae = MultiMAE(); mae.load_state_dict(torch.load(os.path.join(ENC, "multi_all.pt"))); set_seed(SEED + 1000 * k)
            model, ep = fit(Head(mae, len(cls)), X[tr2], y[tr2], X[va], y[va], 40, 2e-4, patience=8)
            Ps.append(proba(model, X[te])); pr = Ps[-1].argmax(1)
            log(f"  pooled-ft {a.classes}-class seed {k}: {ep} ep, acc {(pr == y[te]).mean():.4f} F1 {f1_score(y[te], pr, average='macro'):.4f}")
        pr = np.mean(Ps, 0).argmax(1)
        r = {"acc": float((pr == y[te]).mean()), "f1": f1_score(y[te], pr, average="macro"), "test_frac": a.test_frac, "n_test": int(len(te))}
        json.dump(r, open(os.path.join(rx.RES_DIR, f"multi_pooledft{a.classes}_{int(a.test_frac * 100)}.json"), "w"), indent=1)
        log(f"[pooled-ft {a.classes}-class, test {a.test_frac}] seed-ensemble acc {r['acc']:.4f} F1 {r['f1']:.4f}")
        return

    if a.mode == "pooled":   # NON-STRICT pooled protocol (Li & Liu 2020, Abdelfattah 2025): windows of all people mixed
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.semi_supervised import LabelSpreading
        mae = MultiMAE(); mae.load_state_dict(torch.load(os.path.join(ENC, "multi_all.pt")))
        sn = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-6)
        X = np.concatenate([sn(embed(mae, data[s]["U"]))[data[s]["t"] // WIN] for s in SUBJECTS])
        y = np.concatenate([data[s]["y"] for s in SUBJECTS]); nc = len(cls)
        heads = {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")),
                 "SVM": lambda: make_pipeline(StandardScaler(), SVC(C=10.0, class_weight="balanced", probability=True, random_state=SEED)),
                 "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5, weights="distance")),
                 "GB": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=SEED)}
        P = {h: np.zeros((len(y), nc)) for h in list(heads) + ["LS"]}
        for f, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y)):
            for h, mk in heads.items():
                clf = mk().fit(X[tr], y[tr]); P[h][np.ix_(te, clf.classes_.astype(int))] = clf.predict_proba(X[te])
            yl = np.full(len(y), -1); yl[tr] = y[tr]; Xs = StandardScaler().fit(X[tr]).transform(X)
            ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=100).fit(Xs, yl)
            P["LS"][np.ix_(te, ls.classes_.astype(int))] = ls.label_distributions_[te]
            log(f"  pooled fold {f + 1}/10 done")
        P["Ensemble"] = np.mean([P[h] for h in P], 0); res = {}
        for h, Q in P.items():
            pred = Q.argmax(1)
            res[f"StressMAE-multi + {h}"] = {"acc": float((pred == y).mean()), "f1": f1_score(y, pred, average="macro")}
        json.dump(res, open(os.path.join(rx.RES_DIR, f"multi_pooled{a.classes}.json"), "w"), indent=1)
        log(f"[pooled {a.classes}-class] " + " | ".join(f"{k.split('+ ')[1]} {v['acc']:.4f}/{v['f1']:.4f}" for k, v in res.items()))
        return

    if a.mode == "cv":
        out = os.path.join(rx.RES_DIR, f"multi_cv{a.classes}" + ("_chrono" if "chrono" in a.tag else "") + ("_v2" if "v2" in a.tag else ""))
        os.makedirs(out, exist_ok=True)
        # Same-person protocol: the person's own labelled training folds are used (as in the cited studies).
        # Encoder = the group encoder (pre-trained without labels on the other 12 subjects). Windows do not overlap and
        # no temporal context is used, so no test window shares samples with a training window.
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        sn = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-6)
        nc = len(cls)
        v2 = "v2" in a.tag   # v2: SVM with probabilities joins the ensemble, plus transductive label spreading
        heads = {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")),
                 "SVM": lambda: make_pipeline(StandardScaler(), SVC(C=10.0, class_weight="balanced", probability=v2, random_state=SEED)),
                 "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5, weights="distance")),
                 "GB": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=SEED)}
        for test in SUBJECTS:
            if (a.subjects and test not in a.subjects) or os.path.exists(os.path.join(out, f"{test}.json")):
                continue
            t0 = time.time(); d = data[test]; y = d["y"]; idx = d["t"] // WIN; mae = load_enc(test)
            X = sn(embed(mae, d["U"]))[idx]
            P = {h: np.zeros((len(y), nc)) for h in list(heads) + ["FT"]}
            if "chrono" in a.tag:   # first 2/3 of each condition trains, last 1/3 tests (as Liu et al. 2021)
                tr_m = np.zeros(len(y), bool)
                for c in np.unique(y):
                    ii = np.where(y == c)[0]; tr_m[ii[: int(round(len(ii) * 2 / 3))]] = True
                splits = [(np.where(tr_m)[0], np.where(~tr_m)[0])]
            else:
                splits = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))
            for f, (tr, te) in enumerate(splits):
                if v2:   # label spreading over the person's own windows (test windows unlabelled)
                    from sklearn.semi_supervised import LabelSpreading
                    yl = np.full(len(y), -1); yl[tr] = y[tr]
                    Xs = StandardScaler().fit(X[tr]).transform(X)
                    ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=100).fit(Xs, yl)
                    P.setdefault("LS", np.zeros((len(y), nc)))[np.ix_(te, ls.classes_.astype(int))] = ls.label_distributions_[te]
                for h, mk in heads.items():
                    clf = mk().fit(X[tr], y[tr])
                    if h == "SVM" and not v2:
                        P[h][te] = np.eye(nc)[clf.predict(X[te])]
                    else:
                        P[h][np.ix_(te, clf.classes_.astype(int))] = clf.predict_proba(X[te])
                if "ft" in a.tag:   # personal fine-tuning of the whole StressMAE on this person's training folds
                    tr2, va = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=SEED)
                    set_seed(SEED + f)
                    model, _ = fit(Head(copy.deepcopy(mae), nc), d["X"][tr2], y[tr2], d["X"][va], y[va], 30, 2e-4, patience=8, bs=32)
                    P["FT"][te] = proba(model, d["X"][te])
            if "ft" not in a.tag:
                del P["FT"]
            P["Ensemble"] = np.mean([P[h] for h in P if v2 or h != "SVM"], 0)
            res = {}
            ev = np.concatenate([te for _, te in splits])
            for h, Q in P.items():
                pred = Q.argmax(1)[ev]; yy = y[ev]; yb, pb = (yy == 1).astype(int), (pred == 1).astype(int)
                res[f"StressMAE-multi + {h}"] = {"acc": float((pred == yy).mean()), "f1": f1_score(yy, pred, average="macro"),
                                                 "bacc": float((pb == yb).mean()), "bf1": f1_score(yb, pb, average="macro")}
            json.dump(res, open(os.path.join(out, f"{test}.json"), "w"), indent=1)
            log(f"[multi CV{a.classes} {test}] {time.time() - t0:.0f}s " + " | ".join(f"{k.split('+ ')[1]} {v['acc']:.3f}/{v['f1']:.3f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
