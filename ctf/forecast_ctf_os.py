# Run with:
# python forecast_ctf.py --dataset ODE_Lorenz --pair_id 1 --validation
#  apptainer run --nv --cwd "/app/code" --bind "/mmfs1/home/alexeyy/storage/CTF-for-Science/models/moirai":"/app/code" /mmfs1/home/alexeyy/storage/CTF-for-Science/models/moirai/apptainer/gpu.sif python -u /app/code/ctf/forecast_ctf.py --dataset ODE_Lorenz --pair_id 1 --validation 0 --identifier lorenz_1

# ## Imports

import os
import sys
import time
import torch
import pickle
import argparse
import numpy as np
import pprint as pp
import pandas as pd  # requires: pip install pandas
from pathlib import Path
from chronos import BaseChronosPipeline

from ctf4science.data_module import load_validation_dataset, load_dataset, get_prediction_timesteps, get_validation_prediction_timesteps, get_validation_training_timesteps, get_metadata, get_config

top_dir = Path(__file__).parent.parent
pickle_dir = top_dir / 'pickles'
ckpt_dir = top_dir / 'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)
pickle_dir.mkdir(parents=True, exist_ok=True)

CKPT_INTERVAL_S = 600  # write checkpoints at most every 10 minutes (large pickles on GPFS)


def chronos2_median(pred_df, target_cols):
    """Return the 0.5-quantile forecast as an array of shape (n_steps, len(target_cols)).

    Handles both the long output format of chronos-forecasting>=2.0
    (columns 'target_name' and '0.5') and the older wide format ('<target>_0.5').
    """
    if 'target_name' in pred_df.columns:
        q_col = '0.5' if '0.5' in pred_df.columns else 0.5
        wide = pred_df.pivot(index='timestamp', columns='target_name', values=q_col)
        return wide[target_cols].to_numpy()
    return pred_df[[f"{col}_0.5" for col in target_cols]].values


def expected_test_shape(dataset, pair_id, md):
    """Look up the test matrix shape for a pair from the dataset YAML (works for .mat and .npz)."""
    pair = next(p for p in get_config(dataset)['pairs'] if p['id'] == pair_id)
    return md['matrix_shapes'].get(pair['test'])

def main(args=None):
    # Start timing from the beginning of main
    main_start_time = time.time()
    
    # ## Model Parameters

    print("> Setting up model parameters")

    if args.model == 'chronos-2':
        from chronos import Chronos2Pipeline  # requires chronos-forecasting>=2.0
        pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=args.device,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map=args.device,
            torch_dtype=torch.bfloat16,
        )

    # ## Data

    # Pair ids 2, 4: reconstruction
    # Pair ids 1, 3, 5-7: forecast
    # Pair ids 8, 9: burn-in
    pair_id = args.pair_id
    dataset = args.dataset
    validation = args.validation
    recon_ctx = args.recon_ctx # Context length for reconstruction

    print("> Setting up training data")

    md = get_metadata(dataset)

    if validation:
        train_data, val_data, init_data = load_validation_dataset(dataset, pair_id=pair_id)
        forecast_length = get_validation_prediction_timesteps(dataset, pair_id).shape[0]
    else:
        train_data, init_data = load_dataset(dataset, pair_id=pair_id)
        forecast_length = get_prediction_timesteps(dataset, pair_id).shape[0]

    print(f"> Predicting {dataset} for pair {pair_id} with forecast length {forecast_length}")

    delta_t = md['delta_t']

    # Perform pair_id specific operations
    if pair_id in [2, 4]:
        # Reconstruction
        print(f"> Reconstruction task, using {recon_ctx} context length")
        train_mat = train_data[0]
        train_mat = train_mat[0:recon_ctx,:]
        forecast_length = forecast_length - recon_ctx
    elif pair_id in [1, 3, 5, 6, 7]:
        # Forecast
        print(f"> Forecasting task, using {forecast_length} forecast length")
        train_mat = train_data[0]
    elif pair_id in [8, 9]:
        # Burn-in
        print(f"> Burn-in matrix of size {init_data.shape[0]}, using {forecast_length} forecast length")
        train_mat = init_data
    else:
        raise ValueError(f"Pair id {pair_id} not supported")

    print("> Predicting...")
    if args.model == 'chronos-2':
        # Multivariate prediction: all dimensions at once
        n_timesteps, n_dims = train_mat.shape
        target_cols = [f"dim_{i}" for i in range(n_dims)]
        context_df = pd.DataFrame({
            'id': 'series_0',
            'timestamp': pd.date_range(start='2000-01-01', periods=n_timesteps, freq='s'),
            **{col: train_mat[:, i] for i, col in enumerate(target_cols)},
        })

        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=forecast_length,
            quantile_levels=[0.5],
            id_column='id',
            timestamp_column='timestamp',
            target=target_cols,
        )

        # Extract median (0.5 quantile) for each dimension
        raw_pred = chronos2_median(pred_df, target_cols)  # shape: (forecast_length, n_dims)
        print(f"  Predicted all {n_dims} dims at once")
    else:
        # Univariate column-by-column prediction with checkpointing
        ckpt_path = ckpt_dir / f"{args.identifier}.pkl"
        if ckpt_path.exists():
            print(f"> Loading saved predictions from {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                pred_l = pickle.load(f)
            print(f"  Starting with {sum(p.shape[0] for p in pred_l)} predicted dims")
        else:
            print(f"> No prediction found at {ckpt_path}, initializing pred_l as empty list ([])")
            pred_l = []

        def save_ckpt():
            tmp_path = ckpt_path.with_suffix('.tmp')
            with open(tmp_path, "wb") as f:
                pickle.dump(pred_l, f)
            os.replace(tmp_path, ckpt_path)

        # Each entry of pred_l has shape (n_cols_in_batch, forecast_length)
        n_dims = train_mat.shape[1]
        i = sum(p.shape[0] for p in pred_l)
        last_ckpt_time = time.time()
        while i < n_dims:
            if args.max_time_hours is not None:
                elapsed_hours = (time.time() - main_start_time) / 3600.0
                if elapsed_hours > args.max_time_hours:
                    save_ckpt()
                    print(f"> Maximum time of {args.max_time_hours} hours exceeded ({elapsed_hours:.2f} hours elapsed)")
                    print(f"> Exiting after {i} dimensions (out of {n_dims})")
                    sys.exit(1)

            j = min(i + args.batch_size, n_dims)
            quantiles, mean = pipeline.predict_quantiles(
                torch.tensor(np.ascontiguousarray(train_mat[:, i:j].T)),
                prediction_length=forecast_length,
                quantile_levels=[0.5],
            )

            pred_l.append(mean)

            if time.time() - last_ckpt_time > CKPT_INTERVAL_S:
                save_ckpt()
                last_ckpt_time = time.time()

            print(f"  Predicted dims {i+1}-{j} of {n_dims}")
            i = j

        save_ckpt()

        raw_pred = torch.concatenate(pred_l, dim=0).T.numpy()

    print("> Raw Prediction Shape", raw_pred.shape)

    print("> Creating prediction matrix")

    # Perform pair_id specific operations
    if pair_id in [2, 4]:
        # Reconstruction
        pred = np.vstack([train_mat, raw_pred])
    elif pair_id in [1, 3, 5, 6, 7]:
        # Forecast
        #pred = np.vstack([train_mat, raw_pred])
        pred = raw_pred
    elif pair_id in [8, 9]:
        # Burn-in
        pred = raw_pred
    else:
        raise ValueError(f"Pair id {pair_id} not supported") 

    print("> Predicted Matrix Shape:", pred.shape)
    
    if args.validation:
        print("> Expected Shape: ", val_data.shape)
    else:
        print("> Expected Shape: ", expected_test_shape(args.dataset, pair_id, md))

    # ## Save prediction matrix
    with open(pickle_dir / f"{args.identifier}.pkl", "wb") as f:
        pickle.dump(pred, f)

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, default=None, required=True, help="Identifier for the run")
    parser.add_argument('--model', type=str, default='chronos-t5-base', choices=['chronos-t5-base', 'chronos-2'], help="Model to use (chronos-t5-base or chronos-2)")
    parser.add_argument('--dataset', type=str, default=None, required=True, help="Dataset to run (ODE_Lorenz or PDE_KS)")
    parser.add_argument('--pair_id', type=int, default=1, help="Pair_id to run (1-9)")
    parser.add_argument('--recon_ctx', type=int, default=20, help="Context length for reconstruction")
    parser.add_argument('--validation', type=int, default=0, help="Generate and use validation set")
    parser.add_argument('--device', type=str, default=None, required=True, help="Device to run on")
    parser.add_argument('--max_time_hours', type=float, default=None, help="Maximum time in hours for the forecast loop")
    parser.add_argument('--batch_size', type=int, default=1, help="Number of columns predicted per call for chronos-t5 (univariate, per-column model)")
    args = parser.parse_args()

    # Args
    print("> Args:")
    pp.pprint(vars(args), indent=2)

    # Start timing
    start_time = time.time()
    
    main(args)
    
    # End timing and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Convert to HH:MM:SS format
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"> Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
