# Run with:
# python forecast_ctf.py --dataset ODE_Lorenz --pair_id 1 --validation
#  apptainer run --nv --cwd "/app/code" --bind "/mmfs1/home/alexeyy/storage/CTF-for-Science/models/moirai":"/app/code" /mmfs1/home/alexeyy/storage/CTF-for-Science/models/moirai/apptainer/gpu.sif python -u /app/code/ctf/forecast_ctf.py --dataset ODE_Lorenz --pair_id 1 --validation 0 --identifier lorenz_1

# ## Imports

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

from ctf4science.data_module import load_validation_dataset, load_dataset, get_prediction_timesteps, get_validation_prediction_timesteps, get_validation_training_timesteps, get_metadata

top_dir = Path(__file__).parent.parent
pickle_dir = top_dir / 'pickles'
ckpt_dir = top_dir / 'checkpoints'
pickle_dir.mkdir(parents=True, exist_ok=True)

def main(args=None):
    # Start timing from the beginning of main
    main_start_time = time.time()
    
    # ## Model Parameters

    print("> Setting up model parameters")

    MODEL = "amazon/chronos-t5-base"  # model name
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL,  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
        device_map=args.device,  # use "cpu" for CPU inference
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
        forecast_length = forecast_length - init_data.shape[0]
    else:
        raise ValueError(f"Pair id {pair_id} not supported")

    # Initialize list containing predictions for each column
    #   First check if pickle exists
    ckpt_path = ckpt_dir / f"{args.identifier}.pkl"
    if ckpt_path.exists():
        print(f"> Loading saved predictions from {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            pred_l = pickle.load(f)
        print(f"  Starting with {len(pred_l)} predictions")
    else:
        print(f"> No prediction found at {ckpt_path}, initializing pred_l as empty list ([])")
        pred_l = []

    # Forecast each column
    print("> Predicting...")
    for i in range(len(pred_l), train_mat.shape[1]):
        # Check if max time has been exceeded
        if args.max_time_hours is not None:
            elapsed_hours = (time.time() - main_start_time) / 3600.0
            if elapsed_hours > args.max_time_hours:
                print(f"> Maximum time of {args.max_time_hours} hours exceeded ({elapsed_hours:.2f} hours elapsed)")
                print(f"> Exiting after {i} dimensions (out of {train_mat.shape[1]})")
                sys.exit(1)
        
        quantiles, mean = pipeline.predict_quantiles(
            context=torch.tensor(train_mat[:,i]),
            prediction_length=forecast_length,
            quantile_levels=[0.5],
        )

        pred_l.append(mean)

        # Save prediction
        with open(ckpt_path, "wb") as f:
            pickle.dump(pred_l, f)

        print(f"  Predicted dim {i+1} of {train_mat.shape[1]}")

    # Save prediction
    raw_pred = torch.concatenate(pred_l, dim=0).T

    print("> Concatenated Shape", raw_pred.shape)

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
        pred = np.vstack([train_mat, raw_pred])
    else:
        raise ValueError(f"Pair id {pair_id} not supported") 

    print("> Predicted Matrix Shape:", pred.shape)
    
    if args.validation:
        print("> Expected Shape: ", val_data.shape)
    else:
        if args.dataset in ['seismo', 'ocean_das']:
            print("> Expected Shape: ", md['matrix_shapes'][f'X{pair_id}test.npz'])
        else:
            print("> Expected Shape: ", md['matrix_shapes'][f'X{pair_id}test.mat'])

    # ## Save prediction matrix
    with open(pickle_dir / f"{args.identifier}.pkl", "wb") as f:
        pickle.dump(pred, f)

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, default=None, required=True, help="Identifier for the run")
    parser.add_argument('--dataset', type=str, default=None, required=True, help="Dataset to run (ODE_Lorenz or PDE_KS)")
    parser.add_argument('--pair_id', type=int, default=1, help="Pair_id to run (1-9)")
    parser.add_argument('--recon_ctx', type=int, default=20, help="Context length for reconstruction")
    parser.add_argument('--validation', type=int, default=0, help="Generate and use validation set")
    parser.add_argument('--device', type=str, default=None, required=True, help="Device to run on")
    parser.add_argument('--max_time_hours', type=float, default=None, help="Maximum time in hours for the forecast loop")
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
        
