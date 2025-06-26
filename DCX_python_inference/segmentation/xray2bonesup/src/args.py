"""
initiated by JM Kim, Ph.D., MedicalIP, Inc.
===initial: 27-August-2021
===modified: 03-September-2021
"""

import argparse

class Args():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """
        
        """
        Setting Dataset Params.: Path & Extension
        """
        self.parser.add_argument('--train_A_data_path', type=str, help='Path to the dataset')
        self.parser.add_argument('--train_B_data_path', type=str, help='Path to the dataset')
        self.parser.add_argument('--train_C_data_path', type=str, help='Path to the dataset')

        self.parser.add_argument('--val_A_data_path', type=str, help='Path to the dataset')
        self.parser.add_argument('--val_B_data_path', type=str, help='Path to the dataset')
        self.parser.add_argument('--val_C_data_path', type=str, help='Path to the dataset')

        self.parser.add_argument('--val_data_save_path', type=str, help='Path to the dataset')
        """
        Setting Random Seed
        """
        self.parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        """
        Params for Unet-Network Model
        """
        self.parser.add_argument('--InChans', type=int, default=1, help='===to be inserted===')
        self.parser.add_argument('--OutChans', type=int, default=1, help='===to be inserted===')
        self.parser.add_argument('--num-pools', type=int, default=5, help='Number of U-Net pooling layers')
        self.parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
        self.parser.add_argument('--num-chans', type=int, default=64, help='Number of U-Net channels')
        self.parser.add_argument('--switch_residualpath', type=float, default=0, help='U-Net Residual Path Option')
        """
        Hyperparameters
        """
        self.parser.add_argument('--batch_size', default=1, type=int, help='Mini batch size')
        self.parser.add_argument('--num_epochs_net', type=int, default=50, help='Number of training epochs')
        self.parser.add_argument('--optim', type=str, default="Adam", help='Optimizer')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        """
        Params for Report
        """
        self.parser.add_argument('--report-interval', type=int, default=400, help='Period of loss reporting')
        self.parser.add_argument('--report_interval_epoch', type=int, default=5000, help='Period of loss reporting')
        """
        Hardware Set-Up
        """
        self.parser.add_argument('--device', type=str, default='cuda',  help='Which device to train on. Set to "cuda" to use the GPU')
        self.parser.add_argument('--op_sys', choices=['windows', 'mac', 'linux'], default='windows', help='additional')
        self.parser.add_argument('--accel_method', type=str, default='gpu', help='Which device to train on. Set to "cuda" to use the GPU')
        self.parser.add_argument('--data_parallel', action='store_true', help='If set, use multiple GPUs using data parallelism')
        """
        Set-Up (Path and so on)
        """
        self.parser.add_argument('--exp_dir', type=str, default='checkpoints', help='Path where model and results should be saved')
        self.parser.add_argument('--exp_name', type=str, default='exp', help='Path where model and results should be saved')
        self.parser.add_argument('--resume', action='store_true', help='If set, resume the training from a previous model checkpoint. ''"--checkpoint" should be set with this')
        self.parser.add_argument('--checkpoint', type=str, help='Path to an existing checkpoint. Used along with "--resume"')

        self.initialized = True
    
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        return self.opt
