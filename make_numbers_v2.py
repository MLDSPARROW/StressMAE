#!/usr/bin/env python3
"""Write every number used by the revised manuscript and the response letter from the saved results.
Outputs (into ../stressmae_extracted/tables and ../stressmae_extracted/figures):
  v2_numbers.tex   \newcommand macros (no number is typed by hand in the paper)
  v2_strict.tex    Table: strict leave-one-subject-out comparison
  v2_nonstrict.tex Table: same-person protocols
  v2_lowlabel.tex  Table: label efficiency
  strict_per_subject.png"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.environ.get("PAPER_DIR", os.path.join(ROOT, "..", "stressmae_extracted"))
TAB, FIG = os.path.join(PAPER, "tables"), os.path.join(PAPER, "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results")
SUBJ = [f"S{i}" for i in range(2, 18) if i != 12]
f3 = lambda x: f"{x:.3f}"
macros = {}


def M(name, val):
    assert name.isalpha(), name
    macros[name] = val


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, j in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[j])); adj[j] = run
    return adj


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ strict
R = json.load(open(os.path.join(RES, "final_strict.json")))
ST = json.load(open(os.path.join(RES, "final_strict_stats.json")))
rows = [("StressMAE (ours)", "StressMAE-multi", "consensus", "Ours"),
        ("\\quad without consensus vote", "StressMAE-multi", "post", "OursPost"),
        ("\\quad without any post-processing", "StressMAE-multi", "raw", "OursRaw"),
        ("Same network, no pre-training", "No pre-training (multi)", "consensus", "Scratch"),
        ("StressMAE, ECG channel only", "StressMAE ECG-only", "consensus", "Ecg"),
        ("Handcrafted features + LR", "Features + LR", "consensus", "FeatLR"),
        ("Handcrafted features + RF", "Features + RF", "consensus", "FeatRF")]


def vals(model, var, key):
    if var == "raw":
        S = R[model]["post"]; k = {"acc": "raw_acc", "f1": "raw_f1", "bacc": "raw_bacc", "bf1": "raw_bf1"}[key]
        return np.array([S[s][k] for s in SUBJ])
    S = R[model][var]
    return np.array([S[s][key] for s in SUBJ])


lines = []
for label, model, var, tag in rows:
    cells = []
    for key in ("acc", "f1", "bacc", "bf1"):
        v = vals(model, var, key)
        M(f"S{tag}{dict(acc='Acc', f1='F', bacc='Bacc', bf1='Bf')[key]}", f3(v.mean()))
        M(f"S{tag}{dict(acc='Acc', f1='F', bacc='Bacc', bf1='Bf')[key]}sd", f3(v.std(ddof=1)))
        cells.append(f"{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}")
    if tag == "Ours":
        ptxt = ""
    elif var == "consensus":
        st = ST[model]; ptxt = f"{pfmt(st['p_holm'])} ({st['wins']}/15)"
        M(f"S{tag}P", pfmt(st["p_holm"])); M(f"S{tag}Wins", str(st["wins"]))
    else:
        x = vals("StressMAE-multi", "consensus", "f1") - vals(model, var, "f1")
        ptxt = f"{pfmt(wilcoxon(x).pvalue)}$^\\dagger$ ({int((x > 0).sum())}/15)"
    lines.append(f"{label} & " + " & ".join(cells) + f" & {ptxt} \\\\")
for model, tag in [("No pre-training (multi)", "Scratch"), ("Features + LR", "FeatLR"), ("Features + RF", "FeatRF")]:
    M(f"S{tag}RawF", f3(vals(model, "raw", "f1").mean()))      # single windows, no post-processing
pe = ST["pretraining_effect_post"]
M("SPretrainP", pfmt(pe["p"])); M("SPretrainWins", str(pe["wins"])); M("SPretrainDF", f3(pe["mean_dF1"]))
ours_f1 = vals("StressMAE-multi", "consensus", "f1")
M("SOursMinF", f3(ours_f1.min())); M("SOursMaxF", f3(ours_f1.max()))
M("SOursPerfect", str(int((ours_f1 >= 0.9995).sum()))); M("SOursAbove", str(int((ours_f1 >= 0.9).sum())))
worst = [SUBJ[i] for i in np.argsort(ours_f1)[:3]]
M("SWorstA", worst[0]); M("SWorstB", worst[1]); M("SWorstC", worst[2])
for i, w in enumerate(("A", "B", "C")):
    M(f"SWorst{w}F", f3(np.sort(ours_f1)[i]))
tab = r"""\begin{table}[H]
\centering\addcolor
\caption{\add{Strict leave-one-subject-out results (15 subjects; the test person is never used for pre-training, fine-tuning or any model choice). Mean $\pm$ SD over subjects. Every model uses the same 12 training and 2 validation subjects and the same label-free inference step (score calibration, temporal decoding and consensus vote, settings chosen on the validation subjects). ``Without consensus vote'': the best of no post-processing, a causal moving average over the last 12 windows (1\,min) and Viterbi decoding with $D\in\{12,24,60,120,240\}$ windows, each with or without score calibration, chosen on the validation subjects; ``without any post-processing'': the fine-tuned model applied to each 5\,s window alone. Two-class results (stress vs.\ non-stress) are derived from the three-class predictions. $p$: two-sided Wilcoxon signed-rank test of the per-subject three-class macro F1 against StressMAE, Holm-corrected over the four compared models ($^\dagger$: uncorrected); in brackets, the number of subjects for which StressMAE is better.}}
\label{tab:strict}
\resizebox{\linewidth}{!}{
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{Three classes}} & \multicolumn{2}{c}{\textbf{Two classes}} & \\
\textbf{Model} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{$p$ (wins)} \\
\midrule
""" + "\n".join(lines[:3]) + "\n\\midrule\n" + "\n".join(lines[3:]) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(os.path.join(TAB, "v2_strict.tex"), "w").write(tab)

# per-subject figure
fig, ax = plt.subplots(figsize=(9, 3.2))
x = np.arange(len(SUBJ)); wdt = 0.27
for k, (lab, model, col) in enumerate([("StressMAE", "StressMAE-multi", "#1f4e9c"), ("No pre-training", "No pre-training (multi)", "#9aa5b1"),
                                       ("Features + RF", "Features + RF", "#d08c2f")]):
    ax.bar(x + (k - 1) * wdt, vals(model, "consensus", "f1"), wdt, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels(SUBJ); ax.set_ylabel("Macro F1 (3 classes)"); ax.set_ylim(0, 1.05)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
fig.savefig(os.path.join(FIG, "strict_per_subject.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ non-strict
NB = json.load(open(os.path.join(RES, "final_nonstrict_baselines.json")))


def ours_ns(d):
    S = [json.load(open(f))["StressMAE-multi + Ensemble"] for f in sorted(glob.glob(os.path.join(RES, d, "S*.json")))]
    assert len(S) == 15, (d, len(S))
    return np.array([s["acc"] for s in S]), np.array([s["f1"] for s in S])


ns_lines = []
cols = [("multi_cv3_v2", "cv3", "Three"), ("multi_cv4_v2", "cv4", "Four"), ("multi_cv3_chrono_v2", "chrono3", "Chrono")]
for label, src in [("StressMAE (ours)", None), ("Handcrafted features + LR", "Features + LR"), ("Handcrafted features + RF", "Features + RF")]:
    cells = []
    for d, key, tag in cols:
        if src is None:
            a, f = ours_ns(d); t = "N"
        else:
            S = NB[key][src]; a = np.array([S[s]["acc"] for s in SUBJ]); f = np.array([S[s]["f1"] for s in SUBJ])
            t = "NLR" if "LR" in src else "NRF"
        M(f"{t}{tag}Acc", f3(a.mean())); M(f"{t}{tag}F", f3(f.mean()))
        cells += [f3(a.mean()), f3(f.mean())]
    ns_lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
# single logistic-regression head on the frozen StressMAE embeddings (random 10-fold, 3 classes)
lr_head = [json.load(open(f))["StressMAE-multi + LR"] for f in sorted(glob.glob(os.path.join(RES, "multi_cv3_v2", "S*.json")))]
M("NHeadLRThreeAcc", f3(np.mean([r["acc"] for r in lr_head]))); M("NHeadLRThreeF", f3(np.mean([r["f1"] for r in lr_head])))
tab = r"""\begin{table}[H]
\centering\addcolor
\caption{\add{Same-person protocols (for comparison with the literature only; the person's own labelled data are used for training). StressMAE: frozen pre-trained encoder of the subject's group (pre-trained without that subject) with an ensemble of person-specific classifiers. Random 10-fold: stratified 10-fold cross-validation of the non-overlapping 5\,s windows of each subject. Chronological: first two thirds of each condition for training, last third for testing (as in Liu et al.~\cite{liu2021learning}). No temporal context is used: each window is classified from its own 5\,s of signal, but the embedding standardization and the label spreading use the unlabeled windows of the whole recording. Mean over 15 subjects.}}
\label{tab:nonstrict}
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{Random 10-fold, 3 classes}} & \multicolumn{2}{c}{\textbf{Random 10-fold, 4 classes}} & \multicolumn{2}{c}{\textbf{Chronological, 3 classes}} \\
\textbf{Model} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{Accuracy} & \textbf{Macro F1} \\
\midrule
""" + "\n".join(ns_lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(os.path.join(TAB, "v2_nonstrict.tex"), "w").write(tab)

# ------------------------------------------------------------------ label efficiency
LL = {s: json.load(open(os.path.join(RES, "multi_lowlabel", f"{s}.json"))) for s in SUBJ}
fr = ["0.01", "0.05", "0.1", "0.25"]; names = {"0.01": "One", "0.05": "Five", "0.1": "Ten", "0.25": "Quarter"}
ps = []
for q in fr:
    o = np.array([LL[s][f"StressMAE-multi|{q}"]["f1"] for s in SUBJ]); c = np.array([LL[s][f"No-pretrain-multi|{q}"]["f1"] for s in SUBJ])
    ps.append(wilcoxon(o - c).pvalue if np.any(o - c) else 1.0)
adj = holm(np.array(ps)); ll_lines = []
for q, a in zip(fr, adj):
    o = np.array([LL[s][f"StressMAE-multi|{q}"]["f1"] for s in SUBJ]); c = np.array([LL[s][f"No-pretrain-multi|{q}"]["f1"] for s in SUBJ])
    n = int(np.mean([LL[s][f"StressMAE-multi|{q}"]["n_train"] for s in SUBJ]))
    t = names[q]; M(f"L{t}O", f3(o.mean())); M(f"L{t}S", f3(c.mean())); M(f"L{t}P", pfmt(a)); M(f"L{t}W", str(int((o > c).sum())))
    M(f"L{t}N", str(n))
    ll_lines.append(f"{float(q) * 100:g}\\% & {n} & {o.mean():.3f} $\\pm$ {o.std(ddof=1):.3f} & {c.mean():.3f} $\\pm$ {c.std(ddof=1):.3f} & {int((o > c).sum())}/15 & {pfmt(a)} \\\\")
tab = r"""\begin{table}[H]
\centering\addcolor
\caption{\add{Label efficiency under the strict leave-one-subject-out protocol. Only the given fraction of the labelled windows of the training and validation subjects is used (stratified by subject and class, at least one window per class and subject). Both networks are fine-tuned with one seed, batch size 32, at most 40 epochs and early stopping on the subsampled validation subjects (patience 8). Macro F1 (three classes) of single 5\,s windows without post-processing, mean $\pm$ SD over 15 subjects. $p$: Wilcoxon signed-rank test, Holm-corrected over the four fractions.}}
\label{tab:lowlabel}
\resizebox{\linewidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Labels} & \textbf{Labelled windows} & \textbf{StressMAE} & \textbf{No pre-training} & \textbf{StressMAE better} & \textbf{$p$} \\
\midrule
""" + "\n".join(ll_lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(os.path.join(TAB, "v2_lowlabel.tex"), "w").write(tab)

# ------------------------------------------------------------------ data / pre-training facts
win = []
for g in range(5):
    log = "".join(open(f).read() for f in glob.glob(os.path.join(RES, "log_multi_pretrain*.txt")))
import re
win = sorted({int(m) for m in re.findall(r"\[multi pretrain group \d\] (\d+) windows", log)})
M("PreWinMin", f"{min(win):,}"); M("PreWinMax", f"{max(win):,}")
M("PreHoursMin", f"{min(win) * 5 / 3600:.1f}"); M("PreHoursMax", f"{max(win) * 5 / 3600:.1f}")
curves = [json.load(open(os.path.join(RES, "encoders_multi", f"multi_group{g}.json")))["curve"] for g in range(5)]
M("PreLossFirst", f"{np.mean([c[0] for c in curves]):.2f}"); M("PreLossLast", f"{np.mean([c[-1] for c in curves]):.2f}")

import run_multi as rm
mins, nlab, cls = [], [], np.zeros(3)
for s in SUBJ:
    z = np.load(os.path.join(rm.MDIR, f"{s}.npz")); mins.append(z["sig"].shape[1] / 100 / 60)
    d = rm.load_multi(s); nlab.append(len(d["y"])); cls += np.bincount(d["y"], minlength=3)
mins, nlab = np.array(mins), np.array(nlab); pct = 100 * cls / cls.sum()
M("RecMin", f"${mins.mean():.1f} \\pm {mins.std(ddof=1):.1f}$ ({mins.min():.1f}--{mins.max():.1f})")
M("LabWin", f"${nlab.mean():.0f} \\pm {nlab.std(ddof=1):.0f}$ ({nlab.min()}--{nlab.max()})")
# largest-remainder rounding to one decimal, so that the three percentages sum to exactly 100.0
tenths = np.floor(pct * 10).astype(int)
for j in np.argsort(-(pct * 10 - tenths))[: 1000 - tenths.sum()]:
    tenths[j] += 1
M("ClassPct", " / ".join(f"{t / 10:.1f}\\%" for t in tenths))
M("LabWinTotal", f"{int(nlab.sum()):,}")

with open(os.path.join(TAB, "v2_numbers.tex"), "w") as f:
    f.write("% written by make_numbers_v2.py -- do not edit by hand\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros written")
for k in ["SOursAcc", "SOursF", "SOursBacc", "SOursBf", "SFeatRFF", "SScratchF", "SEcgF", "NThreeAcc", "NFourAcc", "NChronoAcc"]:
    print(k, macros.get(k))
