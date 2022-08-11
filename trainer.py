'''
@InProceedings{yang2020learning,
author = {Yang, Fuzhi and Yang, Huan and Fu, Jianlong and Lu, Hongtao and Guo, Baining},
title = {Learning Texture Transformer Network for Image Super-Resolution},
booktitle = {CVPR},
year = {2020},
month = {June}
}
'''

from utils import calc_psnr_and_ssim, calc_sqi
from model import Vgg19

import os
import numpy as np
#from imageio import imread, imsave
from PIL import Image
#import cv2
import pandas as pd
import scipy.signal as signal
import random

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils


class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if 
             args.num_gpu==1 else self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if 
             args.num_gpu==1 else self.model.module.LTE.parameters()), 
             "lr": args.lr_rate_lte
            }
        ]
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            ref_sr = sample_batched['Ref_sr']
            sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )

            if (not is_init):
                if ('per_loss' in self.loss_all):
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                    per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info( 'per_loss: %.10f' %(per_loss.item()) )
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1, 
                        S, T_lv3, T_lv2, T_lv1)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)
                    loss += adv_loss
                    if (is_print):
                        self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )

            loss.backward()
            self.optimizer.step()

        if ((not is_init)): #and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        for num in range(1):
            self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
            _list=np.zeros(425)
            lr_list=np.zeros(425)
            if (self.args.dataset == 'CUFED'):
                self.model.eval()
                with torch.no_grad():
                    psnr, ssim, cnt = 0., 0., 0
                    sqi, sqi_lr = 0., 0.
                    for i in range(1,6):
                        for i_batch, sample_batched in enumerate(self.dataloader['test'][str(i)]):
                            sample_batched = self.prepare(sample_batched)
                            lr = sample_batched['LR']
                            lr_sr = sample_batched['LR_sr']
                            hr = sample_batched['HR']
                            ref = sample_batched['Ref']
                            ref_sr = sample_batched['Ref_sr']

                            sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                            
                            if (self.args.eval_save_results):
                                sr_save = np.transpose(sr.squeeze().cpu().numpy())
                                pd.DataFrame(sr_save).to_csv(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.csv'))
                            ### calculate psnr and ssim
                            _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())
                            _sqi = calc_sqi(sr.detach())
                            _sqi_lr = calc_sqi(lr.detach())
                            lr_psnr, _ = calc_psnr_and_ssim(lr.detach(), hr.detach())
                            _list[cnt]=_psnr
                            lr_list[cnt] = lr_psnr
                            psnr += _psnr
                            ssim += _ssim
                            sqi += _sqi
                            sqi_lr += _sqi_lr
                            cnt+=1

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                sqi_ave = sqi / cnt
                sqi_lr_ave = sqi_lr / cnt
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f \t SQI: %.3f \t LR SQI: %.3f' %(psnr_ave, ssim_ave, sqi_ave, sqi_lr_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))
            df=pd.DataFrame(_list)
            #df.to_csv("Gaussian_" + str(num) + ".csv", mode='w')
            df.to_csv("Gaussian125.csv", mode='w')
            lrdf = pd.DataFrame(lr_list)
            #lrdf.to_csv("LR_Gaussian_" + str(num) + ".csv", mode = 'w')
            lrdf.to_csv("LR_Gaussian125.csv", mode = 'w')

            self.logger.info('Evaluation over.')

    def test(self):
        self.logger.info('Test process...')
        sampling = 6*50
        list_ = range(0,141)#[0, 1, 12, 16, 18]
        for i in list_:
            self.logger.info('lr path:     %s' %(self.args.lr_path + str(i) + '.csv'))
            self.logger.info('ref path:    %s' %(self.args.ref_path + str(i) + '.csv'))
            
            HR = pd.read_csv(self.args.lr_path + str(i) + '.csv')['PPG']
            HR = signal.resample(HR, sampling)
        
            ### LR and LR_sr
            normalized_signal = (HR - HR.min()) / (HR.max() - HR.min())
            noise_modes = np.array(['Gaussian', 'Poisson', 'S&P', 'Speckle', 'Uniform',
                                    'Mix2', 'Mix3', 'Mix4', 'Mix5', 
                                    'Motion30', 'Motion50', 'Motion70', 'Motion90', 'Clean'])
            #r = random.randint(1, 5)
            #noise_mode = random.sample(noise_modes, r)
            #LR = self.noise_synthesized(nomalized_signal, noise_mode)
            noise_modes = ['Clean']
            for noise in noise_modes:
                if noise[:3] == 'Mix':
                    r = int(noise[-1:])
                    random.seed(i)
                    noise_mode = random.sample(list(noise_modes[:5]), 5)[:r]
                    LR = self.noise_synthesized(normalized_signal, noise_mode, seed=i)
                elif noise[:3] == 'Mot':
                    r = int(noise[-2:])
                    noise_mode = [noise]
                    LR = self.noise_synthesized(normalized_signal, noise_mode, r, i)
                else:
                    noise_mode = [noise]
                    LR = self.noise_synthesized(normalized_signal, noise_mode, seed=i)
                #r = random.randint(1, 5)
                #noise_mode = random.sample(noise_modes, r)
                #noise_mode = ['Motion']
                #LR = normalized_signal#self.noise_synthesized(normalized_signal, noise_mode)
                
                LR = LR * (HR.max() - HR.min()) + HR.min()
                LR_sr = signal.resample(LR, sampling//2)
                LR_sr = signal.resample(LR_sr, sampling)
                
                ### Ref and Ref_sr
                Ref = pd.read_csv(self.args.ref_path + str(i) + '.csv')['PPG']
                Ref = signal.resample(Ref, sampling)
                Ref_sr = signal.resample(Ref, sampling//2)
                Ref_sr = signal.resample(Ref_sr, sampling)
                
                HR = (HR - HR.min()) / ((HR.max() - HR.min()) / 2.) - 1.
                LR = (LR - LR.min()) / ((LR.max() - LR.min()) / 2.) - 1.
                LR_sr = (LR_sr - LR_sr.min()) / ((LR_sr.max() - LR_sr.min()) / 2.) - 1.
                Ref = (Ref - Ref.min()) / ((Ref.max() - Ref.min()) / 2.) - 1.
                Ref_sr = (Ref_sr - Ref_sr.min()) / ((Ref_sr.max() - Ref_sr.min()) / 2.) - 1.
                
                h = sampling
                LR = LR.reshape(1, h, 1)
                LR = np.concatenate([LR,LR,LR], axis=0).transpose(1,2,0)
                LR_sr = LR_sr.reshape(1, h, 1)
                LR_sr = np.concatenate([LR_sr,LR_sr,LR_sr], axis=0).transpose(1,2,0)
                Ref = Ref.reshape(1, h, 1)
                Ref = np.concatenate([Ref,Ref,Ref], axis=0).transpose(1,2,0)
                Ref_sr = Ref_sr.reshape(1, h, 1)
                Ref_sr = np.concatenate([Ref_sr,Ref_sr,Ref_sr], axis=0).transpose(1,2,0)
                
                ### to tensor
                LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
                LR_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
                Ref_t = torch.from_numpy(Ref.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
                Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

                self.model.eval()
                with torch.no_grad():
                    sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t)
                    sr_save = np.transpose(sr.squeeze().cpu().numpy())
                    name = ''
                    for n in range(len(noise_mode)):
                        name = name + noise_mode[n]
                    #save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
                    save_hr_path = os.path.join(self.args.save_dir, 'save_results/HR_' + str(i) + '.csv')
                    save_lr_path = os.path.join(self.args.save_dir, 'save_results/LR_' + str(i) + '_' + name + '.csv')
                    save_sr_path = os.path.join(self.args.save_dir, 'save_results/SR_' + str(i) + '_' + name + '.csv')
                    save_ref_path = os.path.join(self.args.save_dir, 'save_results/Ref' + str(i) + '.csv')
                    pd.DataFrame(HR).to_csv(save_hr_path)
                    pd.DataFrame(LR.reshape(h,3)).to_csv(save_lr_path)
                    pd.DataFrame(sr_save).to_csv(save_sr_path)
                    pd.DataFrame(Ref.reshape(h,3)).to_csv(save_ref_path)
                    self.logger.info('output path: %s' %(save_sr_path))
        
        self.logger.info('Test over.')
        
    def noise_synthesized(self, normalized_signal, noise_mode, per=0, seed=0):
        for i in range(len(noise_mode)):
            random.seed(seed)
            np.random.seed(seed)
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
            elif noise_mode[i][:3] == 'Mot':
                motion_type = random.randint(0, 3)
                df = pd.read_csv('Motion_' + str(motion_type) + '.csv', index_col=0)
                n = random.randrange(df.shape[1])
                sample = np.array(df[str(n)])
                sample = signal.resample(sample, 50*6)
                normalized_sample = (sample - sample.min()) / (sample.max() - sample.min())
                noise_portion = int(per*((50*6)/100))
                start = 0#random.randrange(300-noise_portion)
                LR = np.array([])
                LR = np.append(LR, normalized_signal[:start])
                LR = np.append(LR, (normalized_signal[start:start+noise_portion] + normalized_sample[start:start+noise_portion])/2)
                LR = np.append(LR, normalized_signal[start+noise_portion:])
            else:
                LR = normalized_signal
            normalized_signal = LR
            
        return LR