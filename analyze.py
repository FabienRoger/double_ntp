# %%
from collections import defaultdict
import math
from typing import Any

import numpy as np
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
api = wandb.Api()
runs = api.runs(f"fabien-roger/double_ntp", {"config.version": "v1.4"})

def get_stats(run, keys_of_interest):
    res = {k: [] for k in keys_of_interest}
    if not keys_of_interest:
        return res
    
    for h in tqdm(run.scan_history(keys=keys_of_interest)):
        for k in keys_of_interest:
            if k in h:
                res[k].append(h[k])
    return res


results = [
    {
        "name": run.name,
        "config": run.config,
        "summary": run.summary,
        "history": get_stats(run, ["left_text_pad", "left_code_pad"] if run.config["right_frac"] == 0 else []),
    }
    for run in tqdm(runs)
]
# %%
# plot baseline code & text ntp
baseline_run = [r for r in results if r["config"]["right_frac"] == 0][0]
baseline_code_pad = baseline_run["history"]["left_code_pad"]
baseline_text_pad = baseline_run["history"]["left_text_pad"]
plt.plot(baseline_code_pad, label="code")
plt.plot(baseline_text_pad, label="text")
plt.yscale("log")
plt.legend()
plt.show()
# %%
def loss_to_run_frac(loss, kind):
    losses = {
        "code": baseline_code_pad,
        "text": baseline_text_pad,
    }[kind]
    return min([
        i for i, l in enumerate(losses) if l < loss
    ]) / len(losses)
# %%
for r in results:
    right_frac = r["config"]["right_frac"]
    if right_frac == 0:
        continue
    for pad_kind in ["pad", "text", "code"]:
        right_pad_code = r["summary"][f"right_{pad_kind}_code"]
        right_pad_code_frac = loss_to_run_frac(right_pad_code, "code")
        right_pad_text = r["summary"][f"right_{pad_kind}_text"]
        right_pad_text_frac = loss_to_run_frac(right_pad_text, "text")
        print(f"secondary_frac={right_frac}, when main is {pad_kind:4} - code: {right_pad_code:.2f} ({right_pad_code_frac:.2f}% time), text: {right_pad_text:.2f} ({right_pad_text_frac:.2f}% time)")
# %%
