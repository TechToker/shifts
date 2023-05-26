import copy
import os
import pathlib
from matplotlib import collections as mc

import numpy as np
import torch
import yaml
import cv2

from functools import partial

from sdc.config import build_parser
from sdc.dataset import load_datasets

from sdc.oatomobile.torch.baselines import (ImitativeModel, BehaviouralModel, batch_transform, init_model)
from sdc.oatomobile.torch.baselines.robust_imitative_planning import (load_rip_checkpoints)

from ysdc_dataset_api.utils import transform_2d_points

from ysdc_dataset_api.features import FeatureRenderer

import matplotlib.pyplot as plt


class Test():
    def __init__(self):

        # Init
        parser = build_parser()
        args = parser.parse_args()

        args.bc_generation_mode = 'sampling'
        args.model_name = 'bc'
        args.dir_data = "../dataset/"

        args.data_use_prerendered = True
        args.exp_batch_size = 128

        args.debug_eval_mode = True
        args.debug_overfit_eval = False  # True

        args.rip_per_plan_algorithm = 'BCM'
        args.rip_per_scene_algorithm = 'BCM'

        args.dir_checkpoint = "../baseline-models/"
        args.rip_cache_all_preds = False

        args.debug_collect_dataset_stats = False

        args.rip_eval_subgroup = 'eval'

        if args.np_seed == -1:
            args.np_seed = np.random.randint(0, 1000)
        if args.torch_seed == -1:
            args.torch_seed = np.random.randint(0, 1000)

        pathlib.Path(args.dir_wandb).mkdir(parents=True, exist_ok=True)

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

        device = args.exp_device

        data_dtype = args.data_dtype  # float32
        output_shape = args.model_output_shape  # (25, 2)
        num_timesteps_to_keep, _ = output_shape
        downsample_hw = None

        self.downsize_cast_batch_transform = partial(
            batch_transform, device=device, downsample_hw=downsample_hw,
            dtype=data_dtype, num_timesteps_to_keep=num_timesteps_to_keep,
            data_use_prerendered=args.data_use_prerendered)

        self.model, full_model_name, train_step, self.evaluate_step = init_model(args)

        # Create checkpoint dir, if necessary; init Checkpointer.
        checkpoint_dir = f'{args.dir_checkpoint}/{full_model_name}'

        self.model = load_rip_checkpoints(model=self.model, device=device, k=args.rip_k, checkpoint_dir=checkpoint_dir)
        self.model.eval()

        # Get data item
        datasets = load_datasets(args, splits=None)
        data = datasets['development']['ood__development']

        dataset_iter = iter(data)
        self.data_item = next(dataset_iter)

        # if override_fmap is not None:
        #     data_item['prerendered_feature_map'] = override_fmap

    def inference(self, data_item):
        feature_map = data_item['prerendered_feature_map']

        data_item['scene_id'] = [data_item['scene_id']]
        data_item['track_id'] = torch.tensor([data_item['track_id']])

        for key in data_item['scene_tags']:
            data_item['scene_tags'][key] = [data_item['scene_tags'][key]]

        data_item['ground_truth_trajectory'] = torch.tensor([data_item['ground_truth_trajectory']])
        data_item['prerendered_feature_map'] = torch.tensor([data_item['prerendered_feature_map']])

        # Inference
        evaluate_args = {
            'model': self.model,
            'metadata_cache': None,
            'sdc_loss': None
        }

        batch = self.downsize_cast_batch_transform(data_item)
        evaluate_args['batch'] = batch

        with torch.no_grad():
            predictions, plan_confidence_scores, pred_request_confidence_scores = self.evaluate_step(**evaluate_args)

        predictions = predictions[0]
        plan_confidence_scores = plan_confidence_scores[0]
        pred_request_confidence_scores = pred_request_confidence_scores[0]
        data_item['ground_truth_trajectory'] = data_item['ground_truth_trajectory'][0]

        # Draw predictions

        with open('./sdc/renderer_config.yaml') as f:
            renderer_config = yaml.safe_load(f)

        renderer = FeatureRenderer(renderer_config)

        # Plot vehicles occupancy, pedestrian occupancy, lane occupancy and road polygon
        plt.figure(figsize=(10, 10))
        plt.imshow(feature_map[0], origin='lower', cmap='binary', alpha=0.7)
        plt.imshow(feature_map[6], origin='lower', cmap='binary', alpha=0.5)
        plt.imshow(feature_map[13], origin='lower', cmap='binary', alpha=0.2)
        plt.imshow(feature_map[16], origin='lower', cmap='binary', alpha=0.1)

        ax = plt.gca()

        test_predictions = copy.deepcopy(predictions)

        for i in range(len(test_predictions)):
            test_predictions[i] = transform_2d_points(test_predictions[i], renderer.to_feature_map_tf)
            test_predictions[i] = np.round(test_predictions[i] - 0.5).astype(np.int32)
            ax.add_collection(mc.LineCollection([test_predictions[i]], color='red', linewidths=5))

        # ax.add_collection(mc.LineCollection([transformed_gt], color='green', linewidths=1))

        plt.show()

        np_map = np.zeros_like(feature_map[0])

        np_map = cv2.addWeighted(np_map, 1, feature_map[0], 0.2, 0)
        np_map = cv2.addWeighted(np_map, 1, feature_map[6], 0.5, 0)
        np_map = cv2.addWeighted(np_map, 1, feature_map[13], 0.2, 0)
        np_map = cv2.addWeighted(np_map, 1, feature_map[16], 0.1, 0)

        np_map = (np_map * 255).astype(np.uint8)
        np_map = (255 - np_map)

        np_map = cv2.cvtColor(np_map, cv2.COLOR_GRAY2RGB)

        for i in range(len(predictions)):
            predictions[i] = transform_2d_points(predictions[i], renderer.to_feature_map_tf)
            predictions[i] = np.round(predictions[i] - 0.5).astype(np.int32)

            np_map = cv2.polylines(np_map, np.int32([predictions[i]]), color=(255, 0, 0), isClosed=False, thickness=1)

        np_map = cv2.flip(np_map, 0)
        np_map = cv2.resize(np_map, (256, 256))

        return np_map


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    inference_class = Test()
    inference_class.inference(inference_class.data_item)

    #main()
