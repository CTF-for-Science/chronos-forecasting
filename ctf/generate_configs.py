# Write python code which generates config files for pair_ids 1-9 and for datasets KS_Official and Lorenz_Official

from pathlib import Path

config_template = \
"""dataset:
  name: {dataset}
  pair_id:
  - {pair_id}
model:
  name: chronos
  recon_ctx: 20
  validation: 0
"""

# Define the datasets and pair_ids
datasets = ["KS_Official", "Lorenz_Official"]
pair_ids = list(range(1, 10))  # 1-9

# Get the directory where this script is located
config_dir = Path(__file__).parent / 'config'
config_dir.mkdir(parents=True, exist_ok=True)

# Generate config files
for dataset in datasets:
    for pair_id in pair_ids:
        # Format the template with current values
        config_content = config_template.format(
            dataset=dataset,
            pair_id=pair_id
        )
        
        # Create filename
        filename = f"config_{dataset}_p{pair_id}.yaml"
        filepath = config_dir / filename
        
        # Write the config file
        with open(filepath, 'w') as f:
            f.write(config_content)
        
        print(f"Generated: {filename}")

print(f"Generated {len(datasets) * len(pair_ids)} config files")

# Configs for the newer .npz datasets, one file per (dataset, variant, pair_id).
# Chronos-2 is multivariate (all columns in one group): use the autoregressive forecaster
# (CTX=10, PDT=10) on the very high-dimensional datasets, as was done for msfr.
variant_config_template = \
"""dataset:
  name: {dataset}
  pair_id:
  - {pair_id}
model:
  name: {variant}
  variant: {variant}
  forecaster: {forecaster}
  batch_size: {batch_size}
  recon_ctx: 50
  validation: 0
"""

variant_datasets = ["pke-hpmr", "dynasty", "msfr", "mhd-2cyls"]
variants = {
    # variant: (forecaster per dataset, batch_size per dataset)
    "chronos-t5-base": ({ds: "os" for ds in variant_datasets}, {ds: 16 if ds == "dynasty" else 256 for ds in variant_datasets}),
    "chronos-2": ({"pke-hpmr": "os", "dynasty": "os", "msfr": "ar", "mhd-2cyls": "ar"}, {ds: 256 for ds in variant_datasets}),
}

n_generated = 0
for dataset in variant_datasets:
    for variant, (forecasters, batch_sizes) in variants.items():
        for pair_id in pair_ids:
            filename = f"config_{dataset}_{variant}_p{pair_id}.yaml"
            with open(config_dir / filename, 'w') as f:
                f.write(variant_config_template.format(
                    dataset=dataset,
                    pair_id=pair_id,
                    variant=variant,
                    forecaster=forecasters[dataset],
                    batch_size=batch_sizes[dataset],
                ))
            n_generated += 1
            print(f"Generated: {filename}")

print(f"Generated {n_generated} variant config files")
