import os
import pathlib

import numpy as np
import torch
import wandb

from sdc.config import build_parser
from sdc.trainer import train


def main():
    # Init
    parser = build_parser()
    args = parser.parse_args()

    args.bc_generation_mode = 'sampling'
    args.model_name = 'bc'
    args.dir_data = "../../dataset/"
    args.wandb_project = "ydx_prediction"
    args.data_use_prerendered = True
    args.exp_batch_size = 128

    args.debug_eval_mode = True
    args.debug_overfit_eval = False # True

    args.rip_per_plan_algorithm = 'BCM'
    args.rip_per_scene_algorithm = 'BCM'

    args.dir_checkpoint = "../../baseline-models/"
    args.rip_cache_all_preds = True

    #args.debug_overfit_eval = True
    args.debug_collect_dataset_stats = True

    args.rip_eval_subgroup = 'eval'

    if args.exp_sweep:
        print('Removing old logs.')
        os.system('rm -r wandb')

    if args.np_seed == -1:
        args.np_seed = np.random.randint(0, 1000)
    if args.torch_seed == -1:
        args.torch_seed = np.random.randint(0, 1000)
    if args.exp_name is None:
        args.exp_name = f'{wandb.util.generate_id()}'

    pathlib.Path(args.dir_wandb).mkdir(parents=True, exist_ok=True)

    wandb_args = dict(
        project=args.wandb_project,
        entity="techtoker",
        dir=args.dir_wandb,
        reinit=True,
        name=args.exp_name,
        group=args.exp_group)

    # Set seeds
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)

    # Resolve CUDA device(s)
    if args.exp_use_cuda and torch.cuda.is_available():
        print('Running model with CUDA.')
        exp_device = 'cuda:0'
    else:
        print('Running model on CPU.')
        exp_device = 'cpu'

    args.exp_device = exp_device

    # Initialize wandb
    wandb_run = wandb.init(**wandb_args)
    wandb.config.update(args, allow_val_change=True)
    c = wandb.config

    # Initialize torch hub dir
    torch.hub.set_dir(f'{c.dir_data}/torch_hub')

    print(f'Using {c.data_num_workers} workers in PyTorch dataloading.')

    # Run our script
    train(c)
    wandb_run.finish()


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
