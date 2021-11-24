# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from modelingv2 import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_nni import get_loader
from utils.dist_util import get_world_size
from utils.common_nni import file_write_log, file_write_log_ori
import nni
from nni.utils import merge_parameter

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args['output_dir'], "%s_checkpoint.out_bin" % args['name'])
    if args['fp16']:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args['output_dir'])
    file_write_log(args, "Saved model checkpoint to [DIR: %s]", args['output_dir'])


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == "butterfly200":
        num_classes = 200

    model = VisionTransformer(config, args.img_size, args.output_dir, zero_head=True, num_classes=num_classes,
                              smoothing_value=args.smoothing_value)

    model.load_from(torch.load(args.pretrained_dir))

    # if args.pretrained_model is not None:
    #     pretrained_model = torch.load(args.pretrained_model)['model']
    #     model.load_state_dict(pretrained_model)
    #
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    file_write_log_ori(args, "\n{}".format(config), "\nTraining parameters %s" % args,
                       "\nTotal Parameter: \t%2.1fM" % num_params)

    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed_ori(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_seed(args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['n_gpu'] > 0:
        torch.cuda.manual_seed_all(args['seed'])


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("\n\n\n***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args['eval_batch_size'])

    file_write_log(args, "\n\n***** Running Validation *****", "\n  Num steps = %d" % len(test_loader)
                   , "\n  Batch size = %d" % args['eval_batch_size'])

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args['local_rank'] not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args['device']) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args['device'])

    val_accuracy = accuracy
    if args['local_rank'] != -1:
        dist.barrier()
        val_accuracy = reduce_mean(accuracy, args['nprocs'])

    val_accuracy = val_accuracy.detach().cpu().numpy()

    logger.info("\n\nValidation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    file_write_log(args, "\n\nValidation Results", "\nGlobal Steps: %d" % global_step,
                   "\nValid Loss: %2.5f" % eval_losses.avg, "\nValid Accuracy: %2.5f" % val_accuracy)

    if args['local_rank'] in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)

    return val_accuracy


def train(args, model):
    global optimizer, scheduler

    writer = SummaryWriter(log_dir=os.path.join("logs", os.environ['NNI_OUTPUT_DIR'], "tensorboard"))

    args['train_batch_size'] = args['train_batch_size'] // args['gradient_accumulation_steps']

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    t_total = args['num_steps']

    if args['optimizer'][0] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args['optimizer'][1],
                                    weight_decay=args['optimizer'][2],
                                    momentum=args['optimizer'][3])
    elif args['optimizer'][0] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args['optimizer'][1],
                                      weight_decay=args['optimizer'][2])
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.1)

    print(optimizer)

    if args['decay_type'] == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)

    if args['fp16']:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args['fp16_opt_level'])
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Distributed training
    if args['local_rank'] != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!

    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args['num_steps'])
    logger.info("  Instantaneous batch size per GPU = %d", args['train_batch_size'])
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args['train_batch_size'] * args['gradient_accumulation_steps'] * (
                    torch.distributed.get_world_size() if args['local_rank'] != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])

    file_write_log(args, "\n\n\n***** Running training *****", "\n  Total optimization steps = %d" % args['num_steps']
                   , "\n  Instantaneous batch size per GPU = %d" % args['train_batch_size']
                   , "\n  Total train batch size (w. parallel, distributed & accumulation) = %d" %
                   args['train_batch_size'] * args['gradient_accumulation_steps'] * (
                       torch.distributed.get_world_size() if args['local_rank'] != -1 else 1)
                   , "\n  Gradient Accumulation steps = %d" % args['gradient_accumulation_steps'])

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()

    global_step, best_acc = 0, 0
    start_time = time.time()

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args['local_rank'] not in [-1, 0])
        all_preds, all_label = [], []

        # 一轮图片集
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()

            batch = tuple(t.to(args['device']) for t in batch)
            x, y = batch

            loss, logits = model(x, y)
            loss = loss.mean()
            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                losses.update(loss.item() * args['gradient_accumulation_steps'])
                if args['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                optimizer.step()
                scheduler.step()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args['local_rank'] in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                # Run prediction on validation
                if global_step % args['eval_every'] == 0:
                    with torch.no_grad():
                        accuracy = valid(args, model, writer, test_loader, global_step)
                        if args['local_rank'] != 1:
                            a = int(accuracy * 10000) / 100
                            nni.report_intermediate_result(a)

                    if args['local_rank'] in [-1, 0]:
                        if best_acc < accuracy:
                            save_model(args, model)
                            best_acc = accuracy
                        logger.info("best accuracy so far: %f" % best_acc)
                        file_write_log(args, "\nbest accuracy so far: %f" % best_acc)

                    model.train()

                if global_step % t_total == 0:
                    break

        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args['device'])

        train_accuracy = accuracy

        if args['local_rank'] != -1:
            dist.barrier()
            train_accuracy = reduce_mean(accuracy, args['nprocs'])

        train_accuracy = train_accuracy.detach().cpu().numpy()
        writer.add_scalar("train/accuracy", scalar_value=train_accuracy, global_step=global_step)

        logger.info("train accuracy so far: %f" % train_accuracy)
        file_write_log(args, "\n train accuracy so far: %f" % train_accuracy)
        losses.reset()
        # 总步数达到最大 即退出训练
        if global_step % t_total == 0:
            break

    if args['local_rank'] != 1:
        a = int(best_acc * 10000) / 100
        nni.report_final_result(a)

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))
    file_write_log(args, "\nBest Accuracy: \t%f" % best_acc,
                   "\nEnd Training!",
                   "\nTotal Training Time: \t%f" % ((end_time - start_time) / 3600)
                   , "\n\n")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", "butterfly200"],
                        default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/home/ubuntu/Datas/CUB')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14",
                                                 "CCT-7/3x1", "CCT-7/7x2", "CCT-14/7x2"],
                        default="CCT-14/7x2",
                        help="Which variant to use.")
    parser.add_argument("--pretrain", type=bool, default=True,
                        help="Whether to use pretrain.")
    parser.add_argument("--pretrained_dir", type=str, default="/home/ubuntu/Datas/cct_14_7x2_384_imagenet.pth",
                        help="Where to search for pretrained ViT models.")

    parser.add_argument("--pretrained_model", type=str, default="cct_14_7x2_384",
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--train_log_name", default="log_train_fix_cub_nni_local.txt", type=str,
                        help="train_log_name")

    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--optimizer", default='SGD', type=str,
                        help="The initial learning rate for SGD.")

    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--momentum", default=0, type=float,
                        help="Weight decay if we apply some.")

    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")

    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # 梯度累计 解决显存不足
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # 在执行向后/更新传递之前累积的更新步骤数
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    parser.add_argument('--seq_pool', type=bool, default=True,
                        help="Whether to use seq_pool")

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    # args.pretrained_dir = os.path.join(args.data_root, args.pretrained_dir)

    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training

    if args.local_rank == -1:
        # 单卡训练
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # 分布式
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1

    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    file_write_log_ori(args,
                       "\n\nProcess rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                       (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    set_seed_ori(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    data = nni.get_next_parameter()
    RCV_PARAMS = parse_init_json(data)

    args = vars(args)
    args.update(RCV_PARAMS)

    print(args)

    if args['train_batch_size'] == 32:
        args['eval_batch_size'] = 16
    elif args['train_batch_size'] == 16:
        args['eval_batch_size'] = 8
    else:
        args['eval_batch_size'] = 8

    train(args, model)


def parse_init_json(data):
    params = {}

    value = data['optimizer']
    op_name = value["_name"]
    print(value)

    if op_name == 'SGD':
        params['optimizer'] = [op_name, value['learning_rate'], value['weight_decay'], value['momentum']]
        smoothing_value = value['smoothing_value']
        params['smoothing_value'] = smoothing_value
        train_batch_size = value['train_batch_size']
        params['train_batch_size'] = train_batch_size
    # elif op_name == 'AdamW':
    #     params['optimizer'] = [op_name, value['learning_rate'], value['weight_decay']]

    return params


if __name__ == "__main__":
    main()
