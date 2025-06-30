from tensorboardX import SummaryWriter
import torch
import numpy as np

class ContRecorder():
    def __init__(self):
        self.iter = 0
        self.logdir = './logs/'
        self.logger = SummaryWriter(self.logdir)
            

    def init(self):
        self.iter = 0

    def logTensorBoard(self, l_data):
        self.logger.add_scalar('loss', l_data['loss'], self.iter)
        self.logger.add_scalar('loss1', l_data['loss1'], self.iter)
        self.logger.add_scalar('loss2', l_data['loss2'], self.iter)
        self.iter += 1

        
        

