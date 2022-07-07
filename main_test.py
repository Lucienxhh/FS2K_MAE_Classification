import argparse
import torch

from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed

import models_vit

from engine_finetune import evaluate_1 as evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('test for image classification', add_help=False)

    parser.add_argument('--batch_size', default=523, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')


    parser.add_argument('--drop_path', default=0.1, type=float, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
 
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_workers', default=10, type=int)


    # change it for different classification tasks
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classfication types')
    parser.add_argument('--predict', default='output_dir/checkpoint-39.pth',
                        help='predict from checkpoint')
    parser.add_argument('--data_path', default='data_transform/hair/', type=str,
                        help='dataset path')
    
    return parser
 

def initMae(args):
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    checkpoint = torch.load(args.predict, map_location='cpu')
    checkpoint_model = checkpoint['model']
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    return model


def main(args):
    device = torch.device(args.device)
    model = initMae(args).to(device)

    dataset_test = build_dataset(is_train=False, is_test=True, args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    evaluate(data_loader_test, model, device, cls=args.nb_classes)
    # print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    


