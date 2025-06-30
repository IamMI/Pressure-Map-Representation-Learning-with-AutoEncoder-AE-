"""
Train Model
"""
import torch
import torch.nn as nn
import os
import numpy as np
import sys
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Vit_Encoder import ViT
from Vit_Decoder import Decoder
from datasets import datasets
from record import ContRecorder

def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs = 100
    """
    lr = 1e-4 * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    # Net
    encoder = ViT()
    decoder = Decoder()
    
    # Dataset
    opt = {
        'tv_fn': './essentials/promini_data_split.npy',
    }
    dataset_bsc = datasets(phase='train',opt=opt)
    
    dataloader = DataLoader(dataset_bsc, batch_size=2, shuffle=False,
                            num_workers=19, pin_memory=True)
    
    
    # GPU
    gpu_ids = [0, 1, 2, 3]
    device = torch.device('cuda:%d' % gpu_ids[0])

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder = DataParallel(encoder, gpu_ids)
    decoder = DataParallel(decoder, gpu_ids)

    # Download checkpoints
    encoder_load_path = './checkpoints/Epoch280_Encoder_params.pth'
    decoder_load_path = './checkpoints/Epoch280_Decoder_params.pth'
    if os.path.exists(encoder_load_path):
        print('Encoder : loading from %s' %encoder_load_path)
        encoder.load_state_dict(torch.load(encoder_load_path))
    else:
        print('can not find Encoder checkpoint %s'% encoder_load_path)
        
    if os.path.exists(decoder_load_path):
        print('Decoder : loading from %s' %decoder_load_path)
        decoder.load_state_dict(torch.load(decoder_load_path))
    else:
        print('can not find Decoder checkpoint %s'% decoder_load_path)  
        
    

    # Optimizer
    combined_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    
    # Image to evaluate
    press = np.load(os.path.join('./mini_data', 'pressure.npz'), allow_pickle=True)['pressure']
    gt_press = torch.from_numpy(press[400,:,:]).to(device='cuda:0').unsqueeze(0)
    gt_press = (gt_press/255).float()
    
    # Record
    record = ContRecorder()
    
    # Train
    record.init()
    for epoch in range(0, 3000):
        adjust_learning_rate(combined_optimizer, epoch)
        encoder.train(), decoder.train()
        loss_ls, loss1_ls, loss2_ls = [], [], []
        for data in tqdm(dataloader):
            data['press'] = data['press'].to(device='cuda:0').reshape(-1, 160, 120)
            
            encode = encoder(data['press'])
            pred_press = decoder(encode)
            
            # Calculate loss
            position = torch.nonzero(data['press'])
            
            loss1 = F.mse_loss(pred_press[position[:,0], position[:,1], position[:,2]], \
                              data['press'][position[:,0], position[:,1], position[:,2]])
            
            loss2 = F.mse_loss(pred_press, data['press'])
            loss = loss1*10+loss2*1
            
            loss1_ls.append(loss1)
            loss2_ls.append(loss2)
            loss_ls.append(loss)
            
            # Update
            combined_optimizer.zero_grad()
            loss.backward()
            combined_optimizer.step()
        
        loss_ls, loss1_ls, loss2_ls = torch.stack(loss_ls, dim=0), torch.stack(loss1_ls, dim=0), torch.stack(loss2_ls, dim=0)
        print('Epoch[%d]: loss:[%f], loss1:[%f], loss2:[%f]'%(epoch, loss_ls.mean(), loss1_ls.mean(), loss2_ls.mean()))
        
        result = {
            'loss': loss_ls.mean(),
            'loss1': loss1_ls.mean(),
            'loss2': loss2_ls.mean(),
        }
        record.logTensorBoard(result)
        
        # Record
        if epoch % 20 == 0:
            torch.save(encoder.state_dict(), './checkpoints/Epoch%d_Encoder_params.pth'%(epoch))
            torch.save(decoder.state_dict(), './checkpoints/Epoch%d_Decoder_params.pth'%(epoch))

            # Evaluate
            encoder.eval(), decoder.eval()
            gt_press_copy = gt_press
            encode = encoder(gt_press_copy)
            pred_press = decoder(encode)
            
            # Transform
            pred_press = pred_press.squeeze().to(device='cpu').detach().numpy()
            gt_press_copy = gt_press_copy.squeeze().to(device='cpu').detach().numpy()
            
            line = np.ones((160, 10))
            img = np.concatenate([pred_press, line, gt_press_copy], axis=1)
            
            plt.imsave('./visual/Epoch%d.jpg'%(epoch), img, cmap='gray')
            
            
            

    

    

    