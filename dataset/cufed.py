import os
#from imageio import imread
#from PIL import Image
import numpy as np
import glob
import random
#import cv2
import pandas as pd
import scipy.signal as signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1))
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]) ):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'train/input', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/input') )])
        self.ref_list = sorted([os.path.join(args.dataset_dir, 'train/ref', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/ref') )])
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = pd.read_csv(self.input_list[idx])['PPG']
        HR = signal.resample(HR, 300)
        
        ### LR and LR_sr
        normalized_signal = (HR - HR.min()) / (HR.max() - HR.min())
        noise_modes = ['Gaussian', 'Poisson', 'S&P', 'Speckle', 'Uniform', 'Motion']
        #if random.random() < 1/2:
        #    r = random.randint(1, 6)
        #    noise_mode = random.sample(noise_modes, r)
        #else:
        noise_mode = ['Motion']
        LR = self.noise_synthesized(normalized_signal, noise_mode)
        '''noise_mode = noise_modes[idx%5]
        if noise_mode == 'Gaussian':
            noise = np.random.normal(0., 0.01 ** 0.5, nomalized_signal.shape)
            out = nomalized_signal + noise
            LR = np.clip(out, 0., 1.)
        elif noise_mode == 'Poisson':
            vals = len(np.unique(nomalized_signal))
            vals = 2 ** np.ceil(np.log2(vals))
            out = np.random.poisson(nomalized_signal * vals) / float(vals)
            LR = np.clip(out, 0., 1.)
        elif noise_mode == 'S&P':
            LR = nomalized_signal.copy()
            p = 0.05
            q = 0.5
            flipped = np.random.choice([True, False], size=nomalized_signal.shape, p=[p, 1 - p])
            salted = np.random.choice([True, False], size=nomalized_signal.shape, p=[q, 1 - q])
            peppered = ~salted
            LR[flipped & salted] = 1.
            LR[flipped & peppered] = 0.
        elif noise_mode == 'Speckle':
            noise = np.random.normal(0., 0.01 ** 0.5, nomalized_signal.shape)
            out = nomalized_signal + nomalized_signal * noise
            LR = np.clip(out, 0., 1.)
        elif noise_mode == 'Uniform':
            noise = np.random.uniform(low = 0., high = 0.1, size = nomalized_signal.shape)
            out = nomalized_signal + noise
            LR = np.clip(out, 0., 1.)'''
        
        LR = LR * (HR.max() - HR.min()) + HR.min()
        LR_sr = signal.resample(LR, 150)
        LR_sr = signal.resample(LR_sr, 300)
        ### Ref and Ref_sr
        Ref = pd.read_csv(self.ref_list[idx])['PPG']
        Ref = signal.resample(Ref, 300)
        Ref_sr = signal.resample(Ref, 150)
        Ref_sr = signal.resample(Ref_sr, 300)
        
        HR = (HR - HR.min()) / ((HR.max() - HR.min()) / 2.) - 1.
        LR = (LR - LR.min()) / ((LR.max() - LR.min()) / 2.) - 1.
        LR_sr = (LR_sr - LR_sr.min()) / ((LR_sr.max() - LR_sr.min()) / 2.) - 1.
        Ref = (Ref - Ref.min()) / ((Ref.max() - Ref.min()) / 2.) - 1.
        Ref_sr = (Ref_sr - Ref_sr.min()) / ((Ref_sr.max() - Ref_sr.min()) / 2.) - 1.
        
        ### change type
        h = 300
        LR = np.array(LR).reshape(1, h, 1)
        LR = np.concatenate([LR,LR,LR], axis=0).transpose(1,2,0)
        LR_sr = np.array(LR_sr).reshape(1, h, 1)
        LR_sr = np.concatenate([LR_sr,LR_sr,LR_sr], axis=0).transpose(1,2,0)
        HR = np.array(HR).reshape(1, h, 1)
        HR = np.concatenate([HR,HR,HR], axis=0).transpose(1,2,0)
        Ref = np.array(Ref).reshape(1, h, 1)
        Ref = np.concatenate([Ref,Ref,Ref], axis=0).transpose(1,2,0)
        Ref_sr = np.array(Ref_sr).reshape(1, h, 1)
        Ref_sr = np.concatenate([Ref_sr,Ref_sr,Ref_sr], axis=0).transpose(1,2,0)
        
        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def noise_synthesized(self, normalized_signal, noise_mode):
        for i in range(len(noise_mode)):
            if noise_mode[i] == 'Gaussian':
                noise = np.random.normal(0., 0.01 ** 0.5, normalized_signal.shape)
                out = normalized_signal + noise
                LR = np.clip(out, 0., 1.)
            elif noise_mode[i] == 'Poisson':
                vals = len(np.unique(normalized_signal))
                vals = 2 ** np.ceil(np.log2(vals))
                out = np.random.poisson(normalized_signal * vals) / float(vals)
                LR = np.clip(out, 0., 1.)
            elif noise_mode[i] == 'S&P':
                LR = normalized_signal.copy()
                p = 0.05
                q = 0.5
                flipped = np.random.choice([True, False], size=normalized_signal.shape, p=[p, 1 - p])
                salted = np.random.choice([True, False], size=normalized_signal.shape, p=[q, 1 - q])
                peppered = ~salted
                LR[flipped & salted] = 1.
                LR[flipped & peppered] = 0.
            elif noise_mode[i] == 'Speckle':
                noise = np.random.normal(0., 0.01 ** 0.5, normalized_signal.shape)
                out = normalized_signal + normalized_signal * noise
                LR = np.clip(out, 0., 1.)
            elif noise_mode[i] == 'Uniform':
                noise = np.random.uniform(low = 0., high = 0.1, size = normalized_signal.shape)
                out = normalized_signal + noise
                LR = np.clip(out, 0., 1.)
            elif noise_mode[i] == 'Motion':
                motion_type = random.randint(0, 3)
                df = pd.read_csv('Motion_' + str(motion_type) + '.csv', index_col=0)
                n = random.randrange(df.shape[1])
                sample = np.array(df[str(n)])
                sample = signal.resample(sample, 300)
                normalized_sample = (sample - sample.min()) / (sample.max() - sample.min())
                noise_portion = random.randrange(30, 100, 10)*3
                start = random.randrange(0, 300-noise_portion)
                LR = np.array([])
                LR = np.append(LR, normalized_signal[:start])
                LR = np.append(LR, (normalized_signal[start:start+noise_portion] + normalized_sample[start:start+noise_portion])/2)
                LR = np.append(LR, normalized_signal[start+noise_portion:])
            else:
                LR = normalized_signal
            normalized_signal = LR
            
        return LR


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.csv')))
        self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', 
            '*_' + ref_level + '.csv')))
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        sampling = 6*125
        HR = pd.read_csv(self.input_list[idx])['PPG']
        HR = signal.resample(HR, sampling)
        
        ### LR and LR_sr
        normalized_signal = (HR - HR.min()) / (HR.max() - HR.min())
        noise_modes = ['Gaussian', 'Poisson', 'S&P', 'Speckle', 'Uniform']
        noise_mode = [noise_modes[0]]
        #noise_mode = random.sample(noise_modes, 2)
        #noise_mode = ['Motion']
        LR = self.noise_synthesized(normalized_signal, noise_mode)
        
        LR = LR * (HR.max() - HR.min()) + HR.min()
        LR_sr = signal.resample(LR, sampling//2)
        LR_sr = signal.resample(LR_sr, sampling)
        ### Ref and Ref_sr
        Ref = pd.read_csv(self.ref_list[idx])['PPG']
        Ref = signal.resample(Ref, sampling)
        Ref_sr = signal.resample(Ref, sampling//2)
        Ref_sr = signal.resample(Ref_sr, sampling)
        
        HR = (HR - HR.min()) / ((HR.max() - HR.min()) / 2.) - 1.
        LR = (LR - LR.min()) / ((LR.max() - LR.min()) / 2.) - 1.
        LR_sr = (LR_sr - LR_sr.min()) / ((LR_sr.max() - LR_sr.min()) / 2.) - 1.
        Ref = (Ref - Ref.min()) / ((Ref.max() - Ref.min()) / 2.) - 1.
        Ref_sr = (Ref_sr - Ref_sr.min()) / ((Ref_sr.max() - Ref_sr.min()) / 2.) - 1.
        
        ### change type
        h = sampling
        LR = np.array(LR).reshape(1, h, 1)
        LR = np.concatenate([LR,LR,LR], axis=0).transpose(1,2,0)
        LR_sr = np.array(LR_sr).reshape(1, h, 1)
        LR_sr = np.concatenate([LR_sr,LR_sr,LR_sr], axis=0).transpose(1,2,0)
        HR = np.array(HR).reshape(1, h, 1)
        HR = np.concatenate([HR,HR,HR], axis=0).transpose(1,2,0)
        Ref = np.array(Ref).reshape(1, h, 1)
        Ref = np.concatenate([Ref,Ref,Ref], axis=0).transpose(1,2,0)
        Ref_sr = np.array(Ref_sr).reshape(1, h, 1)
        Ref_sr = np.concatenate([Ref_sr,Ref_sr,Ref_sr], axis=0).transpose(1,2,0)
        
        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def noise_synthesized(self, normalized_signal, noise_mode):
        for i in range(len(noise_mode)):
            if noise_mode[i] == 'Gaussian':
                noise = np.random.normal(0., 0.01 ** 0.5, normalized_signal.shape)
                out = normalized_signal + noise
                LR = np.clip(out, 0., 1.)
            elif noise_mode[i] == 'Poisson':
                vals = len(np.unique(normalized_signal))
                vals = 2 ** np.ceil(np.log2(vals))
                out = np.random.poisson(normalized_signal * vals) / float(vals)
                LR = np.clip(out, 0., 1.)
            elif noise_mode[i] == 'S&P':
                LR = normalized_signal.copy()
                p = 0.05
                q = 0.5
                flipped = np.random.choice([True, False], size=normalized_signal.shape, p=[p, 1 - p])
                salted = np.random.choice([True, False], size=normalized_signal.shape, p=[q, 1 - q])
                peppered = ~salted
                LR[flipped & salted] = 1.
                LR[flipped & peppered] = 0.
            elif noise_mode[i] == 'Speckle':
                noise = np.random.normal(0., 0.01 ** 0.5, normalized_signal.shape)
                out = normalized_signal + normalized_signal * noise
                LR = np.clip(out, 0., 1.)
            elif noise_mode[i] == 'Uniform':
                noise = np.random.uniform(low = 0., high = 0.1, size = normalized_signal.shape)
                out = normalized_signal + noise
                LR = np.clip(out, 0., 1.)
            elif noise_mode[i] == 'Motion':
                motion_type = random.randint(0, 3)
                df = pd.read_csv('Motion_' + str(motion_type) + '.csv', index_col=0)
                n = random.randrange(df.shape[1])
                sample = np.array(df[str(n)])
                sample = signal.resample(sample, 6*125)
                normalized_sample = (sample - sample.min()) / (sample.max() - sample.min())
                noise_portion = int(30*(6*125/100))
                start = random.randrange(6*125-noise_portion)
                LR = np.array([])
                LR = np.append(LR, normalized_signal[:start])
                LR = np.append(LR, (normalized_signal[start:start+noise_portion] + normalized_sample[start:start+noise_portion])/2)
                LR = np.append(LR, normalized_signal[start+noise_portion:])
            else:
                LR = normalized_signal
            normalized_signal = LR
            
        return LR