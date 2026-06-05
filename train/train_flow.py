import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
import sys
sys.path.append('../modules')
import dataset_inversion_spin4d as dataset
import model
try:
    from nvitop import Device
    NVITOP = True
except:
    NVITOP = False
from collections import OrderedDict
import pathlib
import logging
import argparse
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from einops import rearrange
import matplotlib.pyplot as pl

def check_nan_gradients(model):
    """
    Iterates through all parameters of a model and checks if any gradient is NaN.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"!!! NaN gradient found in parameter: {name}")
                return True
    return False
           
class Training(object):
    def __init__(self, config, parallel=False, restart=None):
        
        #**********************
        # Read configuration file
        #**********************
        self.hyperparameters = self.read_config_file(config)

        #**********************
        # Distributed training
        #**********************
        self.parallel = parallel
        self.restart = restart
        self.gpu = self.hyperparameters['training']['gpu']

        # torch.autograd.set_detect_anomaly(True)

        self.cuda = torch.cuda.is_available()

        if self.parallel:            
            self.rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
            self.device = torch.device(f"cuda:{self.rank}")  # Set device to current GPU
            torch.cuda.set_device(self.rank)
        else:
            self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
            self.rank = 0
            self.world_size = 1        

        #**********************
        # Logger
        #**********************
        self.logger = logging.getLogger(f"flow-{self.rank}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if self.parallel:                        
            self.logger.info(f"Hello from rank {self.rank} of {self.world_size}!")            
        else:
            self.logger.info(f"Running on single device {self.device}")
        
        self.smooth = self.hyperparameters['training']['smooth']        

        if (NVITOP):            
            self.handle = Device.all()[0]
            memory_usage = f'{self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
            self.logger.info(f"Computing in {self.device} : {self.handle.name()} - Memory : {memory_usage}")

        self.batch_size = self.hyperparameters['training']['batch_size']        
                
        kwargs = {'num_workers': 8, 'pin_memory': True} if self.cuda else {}

        #**********************
        # Restart from a checkpoint
        #**********************
        if self.restart is not None:
            self.logger.info(f"Restarting from checkpoint: {restart}")
            self.checkpoint = torch.load(self.restart, map_location=lambda storage, loc: storage)

        #**********************
        # UNet for the Stokes conditioning + Flow matching velocity model
        #**********************
        self.model_args_stokes = self.hyperparameters['unet_stokes']
        self.model_args = self.hyperparameters['unet_flow']
                
        self.model = model.ModelSimple(self.model_args, 
                           self.model_args_stokes, 
                           rank=self.rank, 
                           handle=self.handle,
                           logger=self.logger).to(self.device)

                                    
        # If we are restarting from a checkpoint, load the model state
        if self.restart is not None:
            self.logger.info(f"Loading model state from {self.restart}")
            self.model.load_state_dict(self.checkpoint['state_dict'], strict=False)
            
        if self.parallel:
            self.model = DDP(self.model, 
                             device_ids=[self.rank],
                             output_device=self.rank,
                             gradient_as_bucket_view=True)

        if self.rank == 0:            
            self.logger.info('Initializing training and validation datasets...')

        #**********************
        # Training and validation datasets
        #**********************
        self.logger.info('Creating training dataset')
        self.train_dataset = dataset.DatasetInversion(n_samples=self.hyperparameters['training']['n_training'],
                                             n_pixels=self.hyperparameters['training']['n_pixel'],
                                             logger=self.logger if self.rank == 0 else None,
                                             training=True)
        self.logger.info('Creating validation dataset')
        self.validation_dataset = dataset.DatasetInversion(n_samples=self.hyperparameters['training']['n_validation'],
                                                  n_pixels=self.hyperparameters['training']['n_pixel'],
                                                  logger=self.logger if self.rank == 0 else None,
                                                  normalization=self.train_dataset.normalization,
                                                  training=False)

        if self.parallel:
            # Distributed Samplers for training and validation datasets
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            self.validation_sampler = torch.utils.data.distributed.DistributedSampler(self.validation_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                
            # Data loaders that will inject data during training
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, **kwargs)
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, sampler=self.validation_sampler, **kwargs)
        else:
            # Data loaders that will inject data during training
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)


        # Set the precision for training
        if self.hyperparameters['training']['precision'] == 'fp16':
            self.use_amp = True
            self.precision = 'fp16'
            if self.rank == 0:
                self.logger.info('Using fp16 precision')
        else:
            self.use_amp = False
            self.precision = 'fp32'
            if self.rank == 0:
                self.logger.info('Using fp32 precision')

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.stokes_weight = torch.tensor(self.train_dataset.std_stokes.astype('float32')).to(self.device)
        self.mean_stokes = torch.tensor(self.train_dataset.mn_stokes.astype('float32')).to(self.device)

    def read_config_file(self, filename):
        """
        Read a configuration file in YAML format.

        Parameters:
        -----------
        filename : str
            The name of the configuration file.

        Returns:
        --------
        dict
            A dictionary containing the configuration parameters.
        """

        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        return config

    def init_optimize(self):
                
        self.lr = float(self.hyperparameters['training']['lr'])
        self.wd = float(self.hyperparameters['training']['wd'])
        self.n_epochs = self.hyperparameters['training']['n_epochs']
        
        if self.rank == 0:
            self.logger.info(f'Learning rate : {self.lr}')
        
            p = pathlib.Path(f'weights_model_{self.precision}/')
            p.mkdir(parents=True, exist_ok=True)

            current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
            self.out_name = f'weights_model_{self.precision}/{current_time}'
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.restart is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.lr = self.optimizer.param_groups[0]['lr']
            if self.rank == 0:
                self.logger.info(f"Restarting with learning rate: {self.lr}")
        
        self.n_batches = len(self.train_loader)
        
        if self.rank == 0:
            self.logger.info(f"N. batches : {self.n_batches}")
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_batches * self.n_epochs, eta_min=0.3*self.lr)        

        if self.restart is not None:
            self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
            if self.rank == 0:
                self.logger.info(f"Restarting scheduler")


        self.path = AffineProbPath(scheduler=CondOTScheduler())
        
    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e100
        
        if self.rank == 0:
            self.logger.info('Model : {0}'.format(self.out_name))

        if self.restart is not None:
            if self.rank == 0:
                self.logger.info(f"Restarting from epoch {self.checkpoint['epoch']}")
            start_epoch = self.checkpoint['epoch']
        else:
            start_epoch = 1

        for epoch in range(start_epoch, self.n_epochs + 1):
            loss = self.train(epoch)
            loss_val = self.test(epoch)

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict() if not self.parallel else self.model.module.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'hyperparameters': self.hyperparameters,
                'loss': self.loss,
                'loss_val': self.loss_val,
                'normalization': self.train_dataset.normalization,
            }

            if self.rank == 0:
                torch.save(checkpoint, f'{self.out_name}.pth')

                if (loss_val < best_loss):
                    print(f"Saving model {self.out_name}.best.pth")                
                    best_loss = loss_val
                    torch.save(checkpoint, f'{self.out_name}.best.pth')

                if (self.hyperparameters['training']['save_all_epochs']):
                    torch.save(checkpoint, f'{self.out_name}.ep_{epoch}.pth')

        # Cleanup
        if self.parallel:
            dist.destroy_process_group()

    def train(self, epoch):
        self.model.train()
        
        if self.parallel:
            self.train_loader.sampler.set_epoch(epoch)

        if self.rank == 0:
            self.logger.info(f"Epoch {epoch}/{self.n_epochs}")

        if self.rank == 0:        
            tr = tqdm(self.train_loader)
            postfix = OrderedDict()
        else:
            tr = self.train_loader

        loss_avg = 0.0
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (patches, stokes) in enumerate(tr):
            
            self.optimizer.zero_grad()

            x_1 = patches.to(self.device)
            stokes = stokes.to(self.device)

            # Add noise to the Stokes parameters            
            # Noise is added with standard deviation between 5e-4 and 2e-3, different for each epoch
            sigma = np.random.uniform(low=5e-4, high=2e-3)
            noise = sigma * torch.randn_like(stokes)

            stokes += noise
            stokes -= self.mean_stokes[None, :, None, None, None]
            stokes *= self.stokes_weight[None, :, None, None, None]
            stokes = rearrange(stokes, 'b s w x y -> b (s w) x y')
                                    
            x_0 = torch.randn_like(x_1).to(self.device)

            # sample time (user's responsibility)
            t = torch.rand(x_1.shape[0], device=x_1.device)
            
            # Cast operations to mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                            
                # sample probability path                
                path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
                
                # flow matching l2 loss
                loss = (self.model(path_sample.x_t, path_sample.t, stokes) - path_sample.dx_t).square().mean()
                
            # Scale the loss for mixed precision
            self.scaler.scale(loss).backward()

            nan_found = check_nan_gradients(self.model)

            if nan_found:
                print("NaN gradients found. Skipping optimizer step.")
                continue

            # UN-SCALE the gradients before clipping
            # This is the crucial step for mixed precision and gradient clipping
            self.scaler.unscale_(self.optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
            # Unscales gradients and calls the optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()


            if (batch_idx == 0):
                loss_avg = loss.item()                
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg                

            if self.rank == 0:

                if (NVITOP):                
                    gpu_usage = f'{self.handle.gpu_utilization()}'
                    memory_usage = f'{self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
                else:
                    gpu_usage = 'NA'
                    memory_usage = 'NA'
                
                postfix['gpu'] = gpu_usage
                postfix['mem'] = memory_usage
                postfix['lr'] = current_lr
                postfix['L'] = loss_avg                
                tr.set_postfix(ordered_dict=postfix)
                
        self.loss.append(loss_avg)
        
        return loss_avg

    def test(self, epoch):
        self.model.eval()

        if self.parallel:
            self.validation_loader.sampler.set_epoch(epoch)

        if self.rank == 0:
            tr = tqdm(self.validation_loader)
            postfix = OrderedDict()
        else:
            tr = self.validation_loader

        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (patches, stokes) in enumerate(tr):

                x_1 = patches.to(self.device)
                stokes = stokes.to(self.device)

                # Add noise to the Stokes parameters            
                # Noise is added with standard deviation between 5e-4 and 2e-3
                sigma = np.random.uniform(low=5e-4, high=2e-3)
                noise = sigma * torch.randn_like(stokes)
                
                stokes += noise
                stokes -= self.mean_stokes[None, :, None, None, None]
                stokes *= self.stokes_weight[None, :, None, None, None]
                stokes = rearrange(stokes, 'b s w x y -> b (s w) x y')
                
                x_0 = torch.randn_like(x_1).to(self.device)

                # sample time (user's responsibility)
                t = torch.rand(x_1.shape[0], device=x_1.device)
            
                # Cast operations to mixed precision
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):

                    # sample probability path                
                    path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
                
                    # flow matching l2 loss
                    loss = (self.model(path_sample.x_t, path_sample.t, stokes) - path_sample.dx_t).square().mean()
                                                                        
                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

                if self.rank == 0:
                    if (NVITOP):
                        gpu_usage = f'{self.handle.gpu_utilization()}'
                        memory_usage = f'{self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
                    else:
                        gpu_usage = 'NA'
                        memory_usage = 'NA'
                    
                    postfix['gpu'] = gpu_usage
                    postfix['mem'] = memory_usage                        
                    postfix['L'] = loss_avg
                    tr.set_postfix(ordered_dict=postfix)
                        
        self.loss_val.append(loss_avg)
            
        return loss_avg

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser("parallel")
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable Parallel calculations.'
    )
    parser.add_argument(
        '--restart',
        type=str,
        default=None,
        help='Path to checkpoint file to restart from.'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='conf.yaml',
        help='Path to checkpoint file to restart from.'
    )

    args = parser.parse_args()
        
    deepnet = Training(args.config, parallel=args.parallel, restart=args.restart)
    deepnet.init_optimize()
    deepnet.optimize()
