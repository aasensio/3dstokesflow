import torch
import torch.utils.data
import h5py
import numpy as np
import platform
import glob

def normalize_input(x, xmin, xmax):
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

def denormalize_output(x, xmin, xmax):
    return 0.5 * (x + 1.0) * (xmax - xmin) + xmin

class DatasetInversion(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, 
                 n_samples=10000, 
                 n_pixels=32, 
                 logger=None, 
                 normalization=None, 
                 training=True):
        """
        Read 3D models and prepare dataset for training
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super().__init__()

        if 'linux' in platform.node():
            self.root = '/scratch/aasensio/datasets/hinode_sims'
        elif 'vena' in platform.node():
            self.root = '/scratch1/datasets/spin4d_hinode/Hinode'
        elif 'doner' in platform.node() or 'durum' in platform.node():
            self.root = '/net/vena2/scratch/datasets/hinode_sims'
        else:
            self.root = '/home/aasensio/data/datasets'

        #######################
        # Get filenames for all directories
        #######################

        if training:
            dirs = [f'SPIN4D_SSD', 
                    f'SPIN4D_SSD_50G', 
                    f'SPIN4D_SSD_50G_V', 
                    f'SPIN4D_SSD_100G', 
                    f'SPIN4D_SSD_200G',]
            self.nx = 212
        else:
            dirs = [f'SPIN4D_SSD_Large']
            self.nx = 212 * 2

        self.files = []
        for d in dirs:            
            files = glob.glob(f'{self.root}/MURaM/{d}/*.h5')
            self.files.extend(files)

            logger.info(f'Found {len(files)} files in {self.root}/MURaM{d}')
                    
        self.n_cubes = len(self.files)

        logger.info(f'Total number of snapshots: {self.n_cubes}')

        #######################
        # Models
        #######################

        # tau is at [-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4.])
        self.variables = ['T', 'vz', 'tau', 'Bx', 'By', 'Bz', 'logP']
        self.new_variables = ['T', 'vz', 'tau', 'Bp1', 'Bp2', 'Bz', 'logP']
        
        self.n_variables = len(self.variables)
        self.multiplication_factor = [1.0, 1e-5, 1.0, 1.0, 1.0, 1.0, 1.0]  # to convert to cgs where needed
        
        # Get all handlers
        logger.info('Getting model handlers and computing global statistics')
        self.models = []
        mn_models = np.zeros((self.n_cubes, self.n_variables, 11))
        std_models = np.zeros((self.n_cubes, self.n_variables, 11))
        for i, f in enumerate(self.files):
            fin = h5py.File(f, 'r')
            self.models.append(fin)

            if normalization is None:
                q = []
                for k in range(self.n_variables):
                    q.append(self.multiplication_factor[k] * fin[self.variables[k]][0:20, 0:20, :])
                
                # Now transform Bx and By to deal with ambiguity-independent variables
                Bx = q[3]
                By = q[4]                
                Bt = np.sqrt(Bx**2 + By**2)  # horizontal field strength
                phi = np.atan2(By, Bx)  # azimuthal angle
                q[3] = Bt * np.cos(2 * phi)  # Bp1
                q[4] = Bt * np.sin(2 * phi)  # Bp2
                
                for k in range(self.n_variables):
                    med = np.median(q[k], axis=(0, 1))
                    std = np.std(q[k], axis=(0, 1))
                    
                    mn_models[i, k, :] = med
                    std_models[i, k, :] = 3 * std

        # Copute average medians and stds over all cubesif training set
        if normalization is None:
            self.mn_models = np.mean(mn_models, axis=0)
            self.std_models = np.mean(std_models, axis=0)
            logger.info(f'Computed global statistics for models for training set')        
            for k in range(self.n_variables):
                logger.info(f'Variable {self.new_variables[k]}')
                logger.info(f' median : {np.array2string(self.mn_models[k, :], formatter={'float': lambda x: f'{x:.1f}'})}')
                logger.info(f' 3*std  : {np.array2string(self.std_models[k, :], formatter={'float': lambda x: f'{x:.1f}'})}')
        else:
            self.mn_models = normalization['mn_models']
            self.std_models = normalization['std_models']
        
        # Height of tau=1 on average
        self.tau_reference = self.mn_models[2, 2]  # tau=1
            
        #######################
        # Stokes
        #######################
                
        # Get all handlers
        logger.info('Getting stokes handlers')
        self.stokes = []
        mn_stokes = np.zeros((self.n_cubes, 4))
        std_stokes = np.zeros((self.n_cubes, 4))
        for i, f in enumerate(self.files):
            # The filename is very similar, just change MURaM to SIR, interchange "flip" and "ssd" and add the rest
            tmp = f.replace('MURaM', 'SIR').split('-')
            if len(tmp) == 5:
                tmp[1], tmp[2] = tmp[2], tmp[1]
            f = f"{'-'.join(tmp[:-1])}-6302-stokes-Hinode.h5"            
            fin = h5py.File(f, 'r')            
            self.stokes.append(fin)
            
            if normalization is None:

                med = np.median(fin['stokes'][0:20, 0:20, 0, 0])
                std = np.std(fin['stokes'][0:20, 0:20, 0, 0])

                # Stokes I
                mn_stokes[i, 0] = 0.5
                std_stokes[i, 0] = 0.5 #3 * std
                # Stokes Q (make it 10 times larger than Stokes I)
                mn_stokes[i, 1] = 0.0
                std_stokes[i, 1] = 100.0 #3 * std * 0.1
                # Stokes U (make it 10 times larger than Stokes I)
                mn_stokes[i, 2] = 0.0
                std_stokes[i, 2] = 100.0 #3 * std * 0.1
                # Stokes V (make it 10 times larger than Stokes I)
                mn_stokes[i, 3] = 0.0
                std_stokes[i, 3] = 10.0 #3 * std * 0.1
                
        # Copute average medians and stds over all cubes
        if normalization is None:
            self.mn_stokes = np.mean(mn_stokes, axis=0)
            self.std_stokes = np.mean(std_stokes, axis=0)
            logger.info(f'Computed global statistics for Stokes for training set')
        
            
            logger.info(f'   median : {self.mn_stokes}')
            logger.info(f'   3*std    : {self.std_stokes}')
        else:
            self.mn_stokes = normalization['mn_stokes']
            self.std_stokes = normalization['std_stokes']
        

        ##########################
        # Store normalization info
        ##########################
        self.normalization = {
            'mn_models': self.mn_models,
            'std_models': self.std_models,
            'mn_stokes': self.mn_stokes,
            'std_stokes': self.std_stokes
        }

        self.mn_models = self.mn_models.flatten()
        self.std_models = self.std_models.flatten()
                
        ##########################
        # Now define the file, locations, and augmenting for each sample
        ##########################
        self.n_pixels = n_pixels
        
        self.n_samples = n_samples

        self.snapshot = np.random.choice(len(self.stokes), size=self.n_samples)
        
        self.angle = np.random.randint(0, 4, size=self.n_samples)
        self.flipx = np.random.randint(0, 2, size=self.n_samples)
        self.flipy = np.random.randint(0, 2, size=self.n_samples)

        self.top = np.random.randint(0, self.nx - self.n_pixels, size=self.n_samples)
        self.left = np.random.randint(0, self.nx - self.n_pixels, size=self.n_samples)
                                    
    def __getitem__(self, index):
        # Get snapshot
        snapshot = self.snapshot[index]

        # Get Stokes and models for that snapshot
        stokes = self.stokes[snapshot]
        models = self.models[snapshot]
        
        # Get patch
        top = self.top[index]
        left = self.left[index]

        # Remember that Stokes is transposed with respect to models    
        patch_stokes = stokes['stokes'][left:left+self.n_pixels, top:top+self.n_pixels, :, :].copy()
        patch_stokes = np.transpose(patch_stokes, (1, 0, 2, 3))

        patch = []
        for k in range(self.n_variables):
            var = models[self.variables[k]][top:top+self.n_pixels, left:left+self.n_pixels, :].copy()
            var *= self.multiplication_factor[k]
                        
            patch.append(var)
        
        # Now transform Bx and By to deal with sign ambiguity
        Bx = patch[3]
        By = patch[4]
        Bt = np.sqrt(Bx**2 + By**2)  # horizontal field strength
        phi = np.atan2(By, Bx)  # azimuthal angle
        patch[3] = Bt * np.cos(2 * phi)  # Bp1
        patch[4] = Bt * np.sin(2 * phi)  # Bp2

        # Combine all variables
        patch_model = np.concatenate(patch, axis=-1)
        
        # Augmenting
        patch_stokes = np.rot90(patch_stokes, k=self.angle[index], axes=(0, 1))
        patch_model = np.rot90(patch_model, k=self.angle[index], axes=(0, 1))
        if (self.flipx[index] == 1):
            patch_stokes = np.flip(patch_stokes, axis=0)
            patch_model = np.flip(patch_model, axis=0)
        if (self.flipy[index] == 1):
            patch_stokes = np.flip(patch_stokes, axis=1)
            patch_model = np.flip(patch_model, axis=1)

        # Reshape to have channels first
        patch_stokes = np.transpose(patch_stokes, axes=(2, 3, 0, 1))
        patch_model = np.transpose(patch_model, axes=(2, 0, 1))
        
        # Normalize only the models. Stokes will be normalized in the training loop after noise is added
        patch_model -= self.mn_models[:, None, None]
        patch_model /= self.std_models[:, None, None]
        
        return patch_model.astype('float32'), patch_stokes.astype('float32')

    def __len__(self):
        return self.n_samples
    
if __name__ == '__main__':
    import matplotlib.pyplot as pl

    dset = DatasetModel(label='Hinode', n_samples=10000, n_pixels=64, training=True)


    # out = iter(dset)

    # stokes = []
    # for i in range(6):
    #     x, c = next(out)
    #     stokes.append(c[0:1, :, :])

    # stokes = np.concatenate(stokes, axis=0)    
    
    # # fig, ax = pl.subplots(nrows=7, ncols=7, figsize=(20, 20))
    # # loop = 0
    # # for i in range(7):
    # #     for j in range(7):            
    # #         for k in range(3):
    # #             x = dset.phys[k][loop, :, :].flatten()            
    # #             ax[i, j].hist(x, bins=100, color=f'C{k}', histtype='step')
    # #             ax[i, j].set_yscale('log')
    # #         loop += 1