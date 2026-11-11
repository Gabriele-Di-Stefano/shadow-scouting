"""Grafici dell'app (matplotlib, nessuna dipendenza extra)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def radar_compare(radar: pd.DataFrame, top_id: int, u22_id: int, position: str,
                  top_name: str, u22_name: str, clip: float = 3.0):
    """Radar Top vs giovane sulle feature a peso Lasso più alto (z-score rispetto ai Top)."""
    r = radar[radar["position"] == position]
    a = r[r["player_id"] == top_id].set_index("feature")["z"]
    b = r[r["player_id"] == u22_id].set_index("feature")["z"]
    feats = [f for f in a.index if f in b.index]
    if len(feats) < 3:
        return None
    a, b = a[feats].clip(-clip, clip), b[feats].clip(-clip, clip)
    ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"polar": True})
    for vals, name, color in ((a, top_name, "#d62728"), (b, u22_name, "#1f77b4")):
        v = np.concatenate([vals.values, vals.values[:1]])
        ax.plot(ang, v, color=color, linewidth=2, label=name)
        ax.fill(ang, v, color=color, alpha=0.15)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([f.replace("_", " ") for f in feats], fontsize=8)
    ax.set_ylim(-clip, clip)
    ax.set_yticks([-2, 0, 2])
    ax.set_yticklabels(["-2σ", "media Top", "+2σ"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=8)
    fig.tight_layout()
    return fig
