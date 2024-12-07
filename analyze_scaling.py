import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

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
    
    for run_dir in glob.glob(f"{logdir}/**/logs", recursive=True):
        ea = EventAccumulator(run_dir)
        ea.Reload()
        
        config = run_dir.split('/')[-3]
        run_num = run_dir.split('/')[-2]
        
        try:
            # Get final metrics
            final_val_loss = ea.Scalars('Loss/valid')[-1].value
            final_rmse = ea.Scalars('RMSE(u10m)/valid')[-1].value
            throughput = ea.Scalars('Avg samples per sec')[-1].value
            
            # Get config details
            config_details = get_config_details(config)
            param_count = count_parameters(config_details)
            
            results.append({
                'config': config,
                'run': run_num,
                'parameters': param_count,
                'val_loss': final_val_loss,
                'rmse': final_rmse,
                'throughput': throughput,
                'embed_dim': config_details['embed_dim'],
                'depth': config_details['depth'],
                'num_heads': config_details['num_heads']
            })
        except:
            print(f"Skipping incomplete run: {run_dir}")
    
    return pd.DataFrame(results)

def plot_scaling_analysis(df):
    """Generate comprehensive scaling analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 1. Loss vs Parameters
    sns.scatterplot(data=df, x='parameters', y='val_loss', 
                   ax=axes[0,0], alpha=0.6)
    axes[0,0].set_xscale('log')
    axes[0,0].set_title('Validation Loss vs Model Size')
    axes[0,0].set_xlabel('Number of Parameters')
    axes[0,0].set_ylabel('Validation Loss')
    
    # 2. RMSE vs Parameters
    sns.scatterplot(data=df, x='parameters', y='rmse',
                   ax=axes[0,1], alpha=0.6)
    axes[0,1].set_xscale('log')
    axes[0,1].set_title('RMSE vs Model Size')
    axes[0,1].set_xlabel('Number of Parameters')
    axes[0,1].set_ylabel('RMSE')
    
    # 3. Throughput vs Parameters
    sns.scatterplot(data=df, x='parameters', y='throughput',
                   ax=axes[1,0], alpha=0.6)
    axes[1,0].set_xscale('log')
    axes[1,0].set_title('Training Throughput vs Model Size')
    axes[1,0].set_xlabel('Number of Parameters')
    axes[1,0].set_ylabel('Samples/second')
    
    # 4. Compute Efficiency (RMSE reduction per parameter)
    df['efficiency'] = (1.0 - df['rmse']) / df['parameters']
    sns.scatterplot(data=df, x='parameters', y='efficiency',
                   ax=axes[1,1], alpha=0.6)
    axes[1,1].set_xscale('log')
    axes[1,1].set_title('Compute Efficiency vs Model Size')
    axes[1,1].set_xlabel('Number of Parameters')
    axes[1,1].set_ylabel('RMSE Reduction per Parameter')
    
    plt.tight_layout()
    plt.savefig('scaling_analysis.png')
    plt.close()

def fit_scaling_laws(df):
    """Fit and analyze scaling relationships"""
    # Log-log regression for key metrics
    log_params = np.log(df['parameters'])
    
    # Loss scaling
    loss_fit = np.polyfit(log_params, np.log(df['val_loss']), 1)
    loss_exp = loss_fit[0]
    
    # RMSE scaling
    rmse_fit = np.polyfit(log_params, np.log(df['rmse']), 1)
    rmse_exp = rmse_fit[0]
    
    # Throughput scaling
    thru_fit = np.polyfit(log_params, np.log(df['throughput']), 1)
    thru_exp = thru_fit[0]
    
    print("Scaling Law Analysis:")
    print(f"Loss scales with N^{loss_exp:.3f}")
    print(f"RMSE scales with N^{rmse_exp:.3f}")
    print(f"Throughput scales with N^{thru_exp:.3f}")

if __name__ == '__main__':
    results_df = extract_metrics('/path/to/logs')
    plot_scaling_analysis(results_df)
    fit_scaling_laws(results_df)