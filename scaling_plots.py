import marimo

__generated_with = "0.9.34"
app = marimo.App(width="medium")


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
def __(EventAccumulator, Path, defaultdict, glob, pl, plt, sns):
    def get_config_details(run_path):
        """Parse config string like 'dim128_depth6_heads4' into dict"""

        run_name = run_path.parent.stem
        if 'L' in run_name:
            emb = 384
            layers = int(run_name.split('_')[1][1:])
            heads = int(run_name.split('_')[2][1:])
        else:
            emb = int(run_name.split('_')[1][3:])
            layers = 12
            heads = 8

        return {
            'embed_dim': emb,
            'depth': layers,
            'num_heads': heads
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

    def extract_metrics(logdir):
        """Extract metrics from tensorboard logs"""
        results = []
        ts = defaultdict(list)

        for run_dir in glob.glob(f"{logdir}/**/logs", recursive=True)[1:]:
            run_path = Path(run_dir)
            ea = EventAccumulator(run_dir)
            ea.Reload()

            run_num = run_dir.split('/')[-2]

            # Get final metrics
            final_val_loss = ea.Scalars('Loss/valid')[-1].value
            final_rmse = ea.Scalars('RMSE(u10m)/valid')[-1].value
            if 'Avg samples per sec' not in ea.scalars.Keys():
                throughput = 0
            else:
                throughput = ea.Scalars('Avg samples per sec')[-1].value

            # Get config details
            config_details = get_config_details(run_path)
            param_count = count_parameters(config_details)

            rmse = [x.value for x in ea.Scalars('RMSE(u10m)/valid')]
            val_loss = [x.value for x in ea.Scalars('Loss/valid')]
            train_loss = [x.value for x in ea.Scalars('Loss/train')]
            
            ts['run'] += [run_num]*len(rmse)
            ts['rmse'] += rmse
            ts['epoch'] += range(1,len(rmse)+1)
            ts['train_loss'] += val_loss
            ts['val_loss'] += train_loss

            results.append({
                'run': run_num,
                'parameters': param_count,
                'val_loss': final_val_loss,
                'final_rmse': final_rmse,
                'throughput': throughput,
                'embed_dim': config_details['embed_dim'],
                'depth': config_details['depth'],
                'num_heads': config_details['num_heads']
            })

        return pl.DataFrame(results), pl.DataFrame(ts)

    def plot_analysis(df):
        embedding = df.filter(pl.col('run').str.contains('emb'))

        fig, axes = plt.subplots(1,3, figsize=(15,5))
        sns.barplot(embedding, x='embed_dim', y='throughput', ax=axes[0])
        axes[0].set_title('embed_dim vs throughput')
        sns.barplot(embedding, x='embed_dim', y='val_loss', ax=axes[1])
        axes[1].set_title('embed_dim vs val_loss')
        sns.barplot(embedding, x='embed_dim', y='final_rmse', ax=axes[2])
        axes[2].set_title('embed_dim vs rmse')

        params = df.filter(~pl.col('run').str.contains('emb'))
        sns.catplot(params, x='num_heads', y='final_rmse', col='depth', kind='bar')
        plt.show()
    return (
        count_parameters,
        extract_metrics,
        get_config_details,
        plot_analysis,
    )


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
def __(plot_analysis, results_df):
    plot_analysis(results_df)
    return


@app.cell
def __(results_df, ts_df):
    all_data = ts_df.join(results_df, on='run')
    all_data
    return (all_data,)


@app.cell
def __(all_data, pl, sns):
    all_embedding = all_data.filter(pl.col('run').str.contains('emb'))
    sns.relplot(all_embedding, x='epoch', y='rmse', hue='embed_dim', kind='line')
    return (all_embedding,)


@app.cell
def __(all_embedding, sns):
    sns.relplot(all_embedding, x='epoch', y='train_loss', hue='embed_dim', kind='line')
    return


@app.cell
def __(all_embedding, sns):
    sns.relplot(all_embedding, x='epoch', y='val_loss', hue='embed_dim', kind='line')
    return


@app.cell
def __(all_data, pl, sns):
    scaling = all_data.filter(~pl.col('run').str.contains('emb'))
    sns.relplot(scaling, x='epoch', y='rmse', hue='depth', kind='line')
    return (scaling,)


@app.cell
def __(scaling, sns):
    sns.relplot(scaling, x='epoch', y='train_loss', hue='depth', kind='line')
    return


@app.cell
def __(scaling, sns):
    sns.relplot(scaling, x='epoch', y='val_loss', hue='depth', kind='line')
    return


if __name__ == "__main__":
    app.run()
