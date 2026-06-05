import sys
sys.path.append('../../modules')
import unet_meta
import unet
import unet_simple
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model_args, model_stokes_args, rank=0, handle=None, logger=None):
        super().__init__()
                
        self.stokes_unet = unet.UNet(input_channels=model_stokes_args['in_channels'],
                                     output_channels=model_stokes_args['out_channels'],
                                     n_channels=model_stokes_args['model_channels'],
                                     ch_mults=model_stokes_args['channel_mult'],
                                     is_attn=model_stokes_args['is_attn'],
                                     n_blocks=model_stokes_args['n_blocks'])

        
        memory_usage = f'{handle.memory_used_human()}/{handle.memory_total_human()}'
        if rank == 0:
            logger.info(f"Created UNet Stokes - Memory : {memory_usage}")        
        
        self.flow_unet = unet_meta.UNetModel(in_channels=model_args['in_channels'] + model_stokes_args['out_channels'],
                                         model_channels=model_args['model_channels'],
                                         out_channels=model_args['out_channels'],
                                         num_res_blocks=model_args['num_res_blocks'],
                                         attention_resolutions=model_args['attention_resolutions'],
                                         dropout=model_args['dropout'],
                                         channel_mult=model_args['channel_mult'],
                                         conv_resample=model_args['conv_resample'],
                                         dims=model_args['dims'],
                                         num_classes=None,
                                         use_checkpoint=model_args['use_checkpoint'],
                                         num_heads=model_args['num_heads'],
                                         num_head_channels=model_args['num_head_channels'],
                                         num_heads_upsample=model_args['num_heads_upsample'],
                                         use_scale_shift_norm=model_args['use_scale_shift_norm'],
                                         resblock_updown=model_args['resblock_updown'],
                                         use_new_attention_order=model_args['use_new_attention_order'])
        
        memory_usage = f'{handle.memory_used_human()}/{handle.memory_total_human()}'
        if rank == 0:
            logger.info(f"Creating flow matching velocity model - Memory : {memory_usage}")    

        if rank == 0:
            logger.info('N. total parameters UNet Stokes conditioning model : {0}'.format(sum(p.numel() for p in self.stokes_unet.parameters() if p.requires_grad)))
            logger.info('N. total parameters flow matching velocity model : {0}'.format(sum(p.numel() for p in self.flow_unet.parameters() if p.requires_grad)))
        
    def forward(self, x, t, y=None):
        
        stokes_conditioning = self.stokes_unet(y)
        x_and_stokes = torch.cat([x, stokes_conditioning], dim=1)
        out = self.flow_unet(x_and_stokes, t, {})
        
        return out
    
class ModelSimple(nn.Module):
    def __init__(self, model_args, model_stokes_args, rank=0, handle=None, logger=None):
        super().__init__()
                
        self.stokes_unet = unet_simple.UNet(n_channels=model_stokes_args['in_channels'],
                                     n_classes=model_stokes_args['out_channels'],
                                     channels_latent=model_stokes_args['model_channels'])

        
        memory_usage = f'{handle.memory_used_human()}/{handle.memory_total_human()}'
        if rank == 0:
            logger.info(f"Created UNet Stokes - Memory : {memory_usage}")        
        
        self.flow_unet = unet_meta.UNetModel(in_channels=model_args['in_channels'] + model_stokes_args['out_channels'],
                                         model_channels=model_args['model_channels'],
                                         out_channels=model_args['out_channels'],
                                         num_res_blocks=model_args['num_res_blocks'],
                                         attention_resolutions=model_args['attention_resolutions'],
                                         dropout=model_args['dropout'],
                                         channel_mult=model_args['channel_mult'],
                                         conv_resample=model_args['conv_resample'],
                                         dims=model_args['dims'],
                                         num_classes=None,
                                         use_checkpoint=model_args['use_checkpoint'],
                                         num_heads=model_args['num_heads'],
                                         num_head_channels=model_args['num_head_channels'],
                                         num_heads_upsample=model_args['num_heads_upsample'],
                                         use_scale_shift_norm=model_args['use_scale_shift_norm'],
                                         resblock_updown=model_args['resblock_updown'],
                                         use_new_attention_order=model_args['use_new_attention_order'])
        
        memory_usage = f'{handle.memory_used_human()}/{handle.memory_total_human()}'
        if rank == 0:
            logger.info(f"Creating flow matching velocity model - Memory : {memory_usage}")    

        if rank == 0:
            logger.info('N. total parameters UNet Stokes conditioning model : {0}'.format(sum(p.numel() for p in self.stokes_unet.parameters() if p.requires_grad)))
            logger.info('N. total parameters flow matching velocity model : {0}'.format(sum(p.numel() for p in self.flow_unet.parameters() if p.requires_grad)))
        
    def forward(self, x, t, y=None):
        
        stokes_conditioning = self.stokes_unet(y)
        x_and_stokes = torch.cat([x, stokes_conditioning], dim=1)
        out = self.flow_unet(x_and_stokes, t, {})
        
        return out


class ModelGenerative(nn.Module):
    def __init__(self, model_args, rank=0, handle=None, logger=None):
        super().__init__()

        self.conditioning = nn.Conv2d(3, model_args['in_channels'], kernel_size=3, padding=1)
                                
        self.flow_unet = unet_meta.UNetModel(in_channels=model_args['in_channels'] + model_args['in_channels'],
                                         model_channels=model_args['model_channels'],
                                         out_channels=model_args['out_channels'],
                                         num_res_blocks=model_args['num_res_blocks'],
                                         attention_resolutions=model_args['attention_resolutions'],
                                         dropout=model_args['dropout'],
                                         channel_mult=model_args['channel_mult'],
                                         conv_resample=model_args['conv_resample'],
                                         dims=model_args['dims'],
                                         num_classes=None,
                                         use_checkpoint=model_args['use_checkpoint'],
                                         num_heads=model_args['num_heads'],
                                         num_head_channels=model_args['num_head_channels'],
                                         num_heads_upsample=model_args['num_heads_upsample'],
                                         use_scale_shift_norm=model_args['use_scale_shift_norm'],
                                         resblock_updown=model_args['resblock_updown'],
                                         use_new_attention_order=model_args['use_new_attention_order'])
        
        memory_usage = f'{handle.memory_used_human()}/{handle.memory_total_human()}'
        if rank == 0:
            logger.info(f"Creating flow matching velocity model - Memory : {memory_usage}")    

        if rank == 0:            
            logger.info('N. total parameters flow matching velocity model : {0}'.format(sum(p.numel() for p in self.flow_unet.parameters() if p.requires_grad)))
        
    def forward(self, x, t, y=None):

        ycond = self.conditioning(y)
                
        x_and_stokes = torch.cat([x, ycond], dim=1)
        out = self.flow_unet(x_and_stokes, t, {})
        
        return out
