import os
import time
import datetime
import random
import numpy as np 
import argparse
from pathlib import Path
from tqdm import tqdm
import utils
import torch 
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoTokenizer
from model import proteinXVL
from dataset import KIBADataset
from scheduler import create_scheduler


def train(model, data_loader, optimizer, protein_tokenizer, smiles_tokenizer, epoch, warmup_steps, device, scheduler, config):
    model.train()

    header = 'Train Epoch: [{}] '.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, (smiles, proteinSeq, affinity_target) in enumerate(tqdm_data_loader):
        optimizer.zero_grad()
        protein_token = protein_tokenizer(proteinSeq, padding='longest', truncation=True, return_tensors="pt").to(device)
        proteinIds = protein_token['input_ids'].to(device)
        protAttentionMask = protein_token['attention_mask'].to(device)

        smiles_token = smiles_tokenizer(smiles, padding='longest', truncation=True, return_tensors="pt").to(device)
        smilesIds = smiles_token['input_ids'].to(device)
        smilesAttentionMask = smiles_token['attention_mask'].to(device)

        affinity_target = affinity_target.unsqueeze(1).to(device)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1., i/len(data_loader))
        
        contrastive_loss, regression_loss = model(proteinIds, smilesIds, protAttentionMask, smilesAttentionMask, affinity_target, alpha)
        loss = contrastive_loss + regression_loss * 5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

        print('\n')
        tqdm_data_loader.set_description(f'contrastive_loss={contrastive_loss.item():.4f}, regression_loss={regression_loss.item():.4f}')
        if epoch == 0 and 1 % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)


def main(args, config):

    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0 
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    temp = config['temp']

    # Dataset
    print("Creating Dataset")
    dataset = KIBADataset('./data/KIBA.csv', shuffle=False)

    data_loader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, drop_last=True)
    
    protein_tokenizer = AutoTokenizer.from_pretrained(train_config['esm_checkpoint'])
    smiles_tokenizer = BertTokenizer(vocab_file="./vocab_bpe_300.txt", lowercase=False, do_basic_tokenize=False)
    
    print("Build model")
    model = proteinXVL(config=train_config, temp=temp)
    model = model.to(device)

    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler,_ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(state_dict)
        print('load checkpoint form %s' % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    print("Start Training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)
        
        if args.checkpoint:
            stopped_epoch = int(args.checkpoint[-6:-4])
            if epoch <= stopped_epoch:
                continue
    
        train_stats = train(model, data_loader, optimizer, protein_tokenizer, smiles_tokenizer, epoch, warmup_steps, device, lr_scheduler, config)
        
        if True:
            print('SAVE START')
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))
            print('SAVE DONE','checkpoint_%02d.pth' % epoch)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training Time {}'.format(total_time_str))

 
if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='./train/')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    train_config = {
        'embed_dim': 256,
        'batch_size': 2,
        'temp': 0.07,
        'queue_size': 2048,
        'momentum': 0.995,
        'alpha': 0.4,
        'schedular': {'sched': 'cosine', 'lr': 1e-4, 'epochs':5, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 1e-4, 'weight_decay': 0.02},
        'esm_checkpoint': "facebook/esm2_t12_35M_UR50D"
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, train_config)