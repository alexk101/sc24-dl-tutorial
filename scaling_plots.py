import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import seaborn as sns
    from pathlib import Path
    import polars as pl
    from collections import defaultdict
    import json
    return (
        EventAccumulator,
        Path,
        defaultdict,
        glob,
        json,
        np,
        os,
        pd,
        pl,
        plt,
        sns,
    )


@app.cell
def _(Path, defaultdict, pl, plt, sns):
    from networks import vit
    from utils.YParams import YParams
    import torch

    def get_model_params(yaml_path, config, scale_dim, scale_depth, scale_heads):
        params = YParams(yaml_path, config)
        params.embed_dim = scale_dim
        params.depth = scale_depth
        params.num_heads = scale_heads
        model = vit.ViT(params)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return param_count

    def calc_iters(budget, scale_dim, scale_depth, scale_heads):
        param_count = get_model_params(Path('./config/ViT.yaml'), 'mp', scale_dim, scale_depth, scale_heads)
        data_shape = (360, 720)
        patch_size = 8
        seq_len = (data_shape[0] // patch_size) * (data_shape[1] // patch_size)
        global_batch_size = 64
        tokens_per_step = global_batch_size * seq_len
        max_steps = int(budget // (6 * param_count * tokens_per_step))
        num_iters = max_steps // tokens_per_step
        return num_iters, param_count

    budgets = [1e19, 1e20, 1e21, 1e22, 1e23]
    budget_iter = defaultdict(list)
    for layer in [12,16,20,24]:
        for budget in budgets:
            iters, p_count = calc_iters(budget, 384, layer, 8)
            budget_iter['layer'].append(layer)
            budget_iter['params'].append(p_count)
            budget_iter['iters'].append(iters)
            budget_iter['flops'].append(budget)
            budget_iter['embed'].append(384)
            budget_iter['exp'].append('layers')

    for embed in [128, 256, 512, 1024]:
        for budget in budgets:
            iters, p_count = calc_iters(budget, embed, 12, 8)
            budget_iter['layer'].append(12)
            budget_iter['params'].append(p_count)
            budget_iter['iters'].append(iters)
            budget_iter['flops'].append(budget)
            budget_iter['embed'].append(embed)
            budget_iter['exp'].append('embed')

    iter_data = pl.DataFrame(budget_iter)
    fig, ax = plt.subplots(1, 2, figsize=(20,8))
    sns.lineplot(iter_data.filter(pl.col('exp')=='layers'), x='flops', y='iters', hue='layer', ax=ax[0])
    sns.lineplot(iter_data.filter(pl.col('exp')=='embed'), x='flops', y='iters', hue='embed', ax=ax[1])
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    fig
    return (
        YParams,
        ax,
        budget,
        budget_iter,
        budgets,
        calc_iters,
        embed,
        fig,
        get_model_params,
        iter_data,
        iters,
        layer,
        p_count,
        torch,
        vit,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Flops vs Iterations

    In this chinchilla papers, one way in which plots describing computation vs accuracy are created is by limiting the number of iterations the model does as a function of the desired number of Flops.

    This is calculated as follows for a given budget $B$

    $\text{Sequence Length} = (x_{dim} // \text{Patch Size}) * (y_{dim} // \text{Patch Size})$

    $\text{Tokens Per Step} = \text{Global Batch Size} * \text{Sequence Length}$

    $\text{Max Steps} = floor(B // (6 * \text{Parameters} * \text{Tokens Per Step}))$

    $\text{Iterations} = \text{Max Steps} // \text{Tokens Per Step}$
    """)
    return


@app.cell
def _(mo):
    mo.md("## Accuracy vs Number of Attention Heads")
    return


@app.cell
def _(mo):
    mo.md("## Performance vs Number of Attention Heads")
    return


@app.cell
def _(mo):
    mo.md("## Accuracy/Performance vs Number of Attention Heads")
    return


@app.cell
def _(mo):
    mo.md("## Accuracy vs Depth")
    return


@app.cell
def _(mo):
    mo.md("## Performance vs Depth")
    return


@app.cell
def _(mo):
    mo.md("## Accuracy/Performance vs Depth")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Accuracy vs Embedding Dimension

    The embedding dimension in a ViT model determines how well it can learn complex relationships. Patches from the spatial domain are linearly projected into the embedding dimension and are directly proportional to the weights. In this experiment, we hold constant the number of layers and heads, 8 and 12 respectively, while progressively increasing the embedding dimension by mutiples of 2, starting at 128. This shows that accuracy of the model increases as a function of embedding dimension.
    """)
    return


@app.cell
def _(mo):
    mo.md("## Performance vs Embedding Dimension")
    return


@app.cell
def _(mo):
    mo.md("## Accuracy/Performance vs Embedding Dimension")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Accuracy vs Dataset Size

    This experiment was run by varying the amount of data used for training vs validation. Our data consists of a downsampled version of the ERA5 dataset, which consists of multiple 4D variables (ie ones which vary across space and time). The values have a temporal resolution of 6 hours and a spatial resolution of 360 by 720 sampled on a Gaussian Grid. This is across 20 atmospheric variables. Data is further organized by year, with our set consisting of years 1990-2017. 

    For this test, we incrementally reduce the number of years used in training while simultaneously increasing the number used for validation. We begin with 1 year for training and progressively increase this by powers of 2, with samples taken from the front of the validation dataset being moved to the training. 

    The results of this experiment show that accuracy increases as a function of the training number. This is expected, as more data should help the model learn the atmospheric conditions more effectively.
    """)
    return


@app.cell
def _(EventAccumulator, Path, defaultdict, json, np, pl):
    def add_trace(ea: EventAccumulator, ts: dict, name: str, key: str, run_num: str):
            if key not in ea.Tags()['scalars']:
                print(f'Experiment {run_num} has no data for {key}. skipping')
                return
            data = np.array([np.array([x.value, x.step]) for x in ea.Scalars(key)]).T

            ts['run'] += [run_num]*len(data[0])
            ts['series'] += [name] * len(data[0])
            ts['value'] += data[0].tolist()
            ts['epoch'] += range(1, 1+len(data[0]))
            ts['iter'] += data[1].tolist()


    def load_logs(target: Path):
        output_params = defaultdict(list)
        output_traces = defaultdict(list)

        for run_dir in target.iterdir():
            ea = EventAccumulator(str(run_dir/'logs'))
            ea.Reload()
            with open(run_dir/'hparams.json', 'r') as fp:
                params = json.load(fp)
            exp = run_dir.name

            if 'n_nodes' not in list(params.keys()):
                print(run_dir)

            for key, val in params.items():
                output_params[key].append(val)
            output_params['run'].append(exp)

            add_trace(ea, output_traces, 'rmse', 'RMSE(u10m)/valid', exp)
            add_trace(ea, output_traces, 'val_loss', 'Loss/valid', exp)
            add_trace(ea, output_traces, 'train_loss', 'Loss/train', exp)
            add_trace(ea, output_traces, 'samples/sec', 'Avg samples per sec', exp)
        return pl.DataFrame(output_traces).join(pl.DataFrame(output_params), on='run')
    return add_trace, load_logs


@app.cell
def _(Path, load_logs, pl):
    target_new_logs = Path('./scaling_logs')
    if Path('./results/exps.parquet').exists():
        exps = pl.read_parquet('./results/exps.parquet')
    else:
        exps = load_logs(target_new_logs)
        exps = exps.with_columns(pl.col('time_limit').str.to_time())
        exps = exps.cast({'run': pl.Int16})
        exps.write_parquet('./results/exps.parquet')
    exps
    return exps, target_new_logs


@app.cell
def _(exps, pl, sns):
    train_data = exps.filter(pl.col('series')=='rmse').sort('time_limit')
    g = sns.relplot(train_data.to_pandas(), x='epoch', y='value', col='time_limit', row='embed', kind='line')
    g.set(ylim=(0, 1))
    return g, train_data


@app.cell
def _(sns, train_data):
    g2 = sns.relplot(train_data.to_pandas(), x='iter', y='value', col='time_limit', row='embed', kind='line', hue='train_years')
    g2.set(ylim=(0, 1))
    return (g2,)


@app.cell
def _(defaultdict, np, target_new_logs):
    mem_usage = defaultdict(list)

    for x in target_new_logs.iterdir():
        with open(x/'out.log', 'r') as fp:
            run_log = fp.readlines()
        for line in run_log:
            if "Memory" in line:
                mem_usage[int(x.name)].append(float(line.split(' ')[-2]))

    for key, val in sorted(mem_usage.items(), key=lambda x: x[0]):
        print(f"{key}: mean {np.mean(val):.2f} | std {np.std(val):.2f} | samples {len(val)}")
    return fp, key, line, mem_usage, run_log, val, x


@app.cell
def _(mem_usage, np, plt):
    # M2 ≈ M1 * (E2/E1)²
    def plot_mem_usage(sample):
        # samp_func = np.mean
        samp_func = np.max
        measurement = samp_func(mem_usage[sample])

        x = np.linspace(0, 2048, 16)
        traces = {}
        traces['fp16_local_batch_8'] = measurement * (x/512)**2
        traces['fp16_local_batch_4'] = measurement * (x/512)**2 / 2
        traces['fp16_local_batch_2'] = measurement * (x/512)**2 / 4

        fig, ax = plt.subplots(1, 1)
        ax.scatter(512, measurement, label=f'measurement')
        for key, val in traces.items():
            ax.plot(x, val, label=key)
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Projected Memory Usage (GB)')
        ax.set_title('Projected Memory Usage vs Embedding Dimension')
        ax.axhline(y=40, color='r', linestyle='--', label='40GB GPU')
        ax.legend()

        # Calculate intercepts
        intercepts = {}
        for key, val in traces.items():
            intercepts[key] = np.interp(40, val, x)
            print(f'{key}: Max embedding dimension {intercepts[key]:.0f}')
        return fig

    plot_mem_usage(10)
    return (plot_mem_usage,)


@app.cell
def _(mo):
    mo.md("## Accuracy vs Dtype")
    return


@app.cell
def _(mo):
    mo.md("## Performance vs Dtype")
    return


@app.cell
def _(mo):
    mo.md("## Accuracy/Performance vs Dtype")
    return


if __name__ == "__main__":
    app.run()
