import marimo

__generated_with = "0.9.34"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
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
    return (
        EventAccumulator,
        Path,
        defaultdict,
        glob,
        np,
        os,
        pd,
        pl,
        plt,
        sns,
    )


@app.cell
def __(Path, defaultdict, pl, plt, sns):
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
def __(EventAccumulator, Path, defaultdict, glob, pl):
    def get_config_details(run_path):
        """Parse config string like 'dim128_depth6_heads4' into dict"""

        run_name = run_path.parent.stem
        if 'L' in run_name:
            emb = 384
            layers = int(run_name.split('_')[1][1:])
            heads = int(run_name.split('_')[2][1:])
            train_years = 27
        else:
            emb = int(run_name.split('_')[1][3:])
            layers = 12
            heads = 8
            train_years = 27
            if 'val' in run_name:
                train_years = int(run_name.split('_')[2][3:])

        return {
            'embed_dim': emb,
            'depth': layers,
            'num_heads': heads,
            'train_years': train_years
        }


    def count_parameters(config):
        """Calculate total trainable parameters for a given config"""
        embed_dim = config['embed_dim']
        depth = config['depth']
        num_heads = config['num_heads']

        # Transformer block parameters
        attn_params = 4 * embed_dim * embed_dim  # Q,K,V matrices and output projection
        mlp_params = 8 * embed_dim * embed_dim  # MLP with 4x hidden dim
        block_params = attn_params + mlp_params

        # Total parameters
        total_params = depth * block_params
        return total_params


    def add_trace(ea: EventAccumulator, ts: dict, name: str, key: str, run_num: str):
        if key not in ea.Tags()['scalars']:
            print(f'Experiment {run_num} has no data for {key}. skipping')
            return
        ts_data = [x.value for x in ea.Scalars(key)]

        ts['run'] += [run_num]*len(ts_data)
        ts['series'] += [name] * len(ts_data)
        ts['value'] += ts_data
        ts['epoch'] += range(1, 1+len(ts_data))


    def extract_metrics(logdir):
        """Extract metrics from tensorboard logs"""
        results = []
        ts = defaultdict(list)

        for run_dir in glob.glob(f"{logdir}/**/logs", recursive=True)[1:]:
            run_path = Path(run_dir)
            ea = EventAccumulator(run_dir)
            ea.Reload()

            run_num = run_dir.split('/')[-2]
            # Get config details
            config_details = get_config_details(run_path)
            param_count = count_parameters(config_details)

            add_trace(ea, ts, 'rmse', 'RMSE(u10m)/valid', run_num)
            add_trace(ea, ts, 'val_loss', 'Loss/valid', run_num)
            add_trace(ea, ts, 'train_loss', 'Loss/train', run_num)
            add_trace(ea, ts, 'samples/sec', 'Avg samples per sec', run_num)

            results.append({
                'run': run_num,
                'parameters': param_count,
                'embed_dim': config_details['embed_dim'],
                'depth': config_details['depth'],
                'num_heads': config_details['num_heads'],
                'train_years': config_details['train_years']
            })

        return pl.DataFrame(results), pl.DataFrame(ts)
    return add_trace, count_parameters, extract_metrics, get_config_details


@app.cell
def __(extract_metrics):
    # target = '/Users/alexdev/Documents/research/phd/projects/sc24-dl-tutorial'
    target = '/home/alexk101/sc24-dl-tutorial'
    results_df, ts_df = extract_metrics(target)
    results_df
    return results_df, target, ts_df


@app.cell
def __(ts_df):
    ts_df
    return


@app.cell
def __(results_df, ts_df):
    all_data = ts_df.join(results_df, on='run')
    all_data
    return (all_data,)


@app.cell
def __():
    acc_vars = ['rmse', 'val_loss', 'train_loss']
    perf_vars = ['samples/sec']
    return acc_vars, perf_vars


@app.cell
def __(all_data, pl):
    layer_exps = all_data.filter(pl.col('run').str.contains('L'))
    print(layer_exps['run'].unique().to_list())
    return (layer_exps,)


@app.cell
def __(acc_vars, layer_exps, pl, sns):
    layer_acc_dat = layer_exps.filter(pl.col('series').is_in(acc_vars))
    sns.relplot(layer_acc_dat, x='epoch', y='value', col='series', hue='num_heads', kind='line')
    return (layer_acc_dat,)


@app.cell
def __(layer_exps, perf_vars, pl, sns):
    layer_perf_dat = layer_exps.filter(pl.col('series').is_in(perf_vars))
    sns.relplot(layer_perf_dat, x='epoch', y='value', col='depth', hue='num_heads', kind='line')
    return (layer_perf_dat,)


@app.cell
def __(layer_acc_dat, sns):
    sns.relplot(layer_acc_dat, x='epoch', y='value', col='series', hue='depth', kind='line')
    return


@app.cell
def __(acc_vars, all_data, pl, sns):
    all_embedding = all_data.filter(pl.col('run').str.contains('emb'))
    all_embedding = all_data.filter(~pl.col('run').str.contains('val'))
    all_embedding = all_embedding.filter(~pl.col('embed_dim').is_in([384, 768, 576]))
    emb_acc_dat = all_embedding.filter(pl.col('series').is_in(acc_vars))
    sns.relplot(emb_acc_dat, x='epoch', y='value', col='series', hue='embed_dim', kind='line')
    return all_embedding, emb_acc_dat


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Accuracy vs Embedding Dimension

        The embedding dimension in a ViT model determines how well it can learn complex relationships. Patches from the spatial domain are linearly projected into the embedding dimension and are directly proportional to the weights. In this experiment, we hold constant the number of layers and heads, 8 and 12 respectively, while progressively increasing the embedding dimension by mutiples of 2, starting at 128. This shows that accuracy of the model increases as a function of embedding dimension.
        """
    )
    return


@app.cell
def __(acc_vars, all_data, pl, sns):
    dataset_scaling = all_data.filter(pl.col('run').str.contains('val'))
    dataset_acc_dat = dataset_scaling.filter(pl.col('series').is_in(acc_vars))
    sns.relplot(dataset_acc_dat, x='epoch', y='value', col='series', hue='train_years', kind='line')
    return dataset_acc_dat, dataset_scaling


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Accuracy vs Dataset Size

        This experiment was run by varying the amount of data used for training vs validation. Our data consists of a downsampled version of the ERA5 dataset, which consists of multiple 4D variables (ie ones which vary across space and time). The values have a temporal resolution of 6 hours and a spatial resolution of 360 by 720 sampled on a Gaussian Grid. This is across 20 atmospheric variables. Data is further organized by year, with our set consisting of years 1990-2017. 

        For this test, we incrementally reduce the number of years used in training while simultaneously increasing the number used for validation. We begin with 1 year for validation and progressively increase this by powers of 2, with samples taken the tail end of the training dataset being moved to the validation. 

        | Training Number      | Validation  | Training    |
        | -------------------- | ----------- | ----------- |
        | 1                    | 1990 - 2016 | 2016 - 2017 |
        | 2                    | 1990 - 2015 | 2015 - 2017 |
        | 4                    | 1990 - 2013 | 2013 - 2017 |
        | 8                    | 1990 - 2009 | 2009 - 2017 |
        | 16                   | 1990 - 2001 | 2001 - 2017 |

        The results of this experiment show that accuracy increases as a function of the training number. This is expected, as more data should help the model learn the atmospheric conditions more effectively.
        """
    )
    return


if __name__ == "__main__":
    app.run()
