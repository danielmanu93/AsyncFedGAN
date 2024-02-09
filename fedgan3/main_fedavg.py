import os
import argparse
from trainer import Trainer
from Dataloader import get_loader
from torch.backends import cudnn
from tqdm import tqdm
from molecular_dataset import MolecularDataset
import numpy as np
import copy
# from utils import average_weights
import utils
import matplotlib.pyplot as plt

def str2bool(v):
    return v.lower() in ('true')

def main(args):
    
    # Since graph input sizes remains constant
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    data = MolecularDataset()
    data.load(args.mol_data_dir)
    idxs  = len(str(data))

    # trainer for training and testing StarGAN.
    train_data, test_data, user_groups = get_loader(args)

    trainer = Trainer(args, data, idxs)
    
    g_global_model, d_global_model = trainer.build_model()

    # global model weights
    g_global_weights = g_global_model.state_dict()
    d_global_weights = d_global_model.state_dict()

    g_train_loss, d_train_loss = [], []
    g_last_local_loss, d_last_local_loss = [], []

    if args.mode == 'train':
        for i in tqdm(range(args.epochs_global)):
            g_local_weights, g_local_losses, d_local_weights, d_local_losses = [], [], [], []
            
            # g_last_local_loss.clear()
            # d_last_local_loss.clear()  

            print(f'\n | Global Training Round : {i+1} |\n')

            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                local_model = Trainer(args=args, data=train_data, idxs=user_groups[idx])  
                g_weights, d_weights, g_loss, d_loss = local_model.tnr(model=copy.deepcopy(d_global_model), global_round=i)

                g_local_weights.append(copy.deepcopy(g_weights))
                g_local_losses.append(copy.deepcopy(g_loss))

                d_local_weights.append(copy.deepcopy(d_weights))
                d_local_losses.append(copy.deepcopy(d_loss))

                g_last_local_loss.append(g_local_losses[-1])
                d_last_local_loss.append(d_local_losses[-1])

            g_local_losses = np.array(g_last_local_loss).ravel()
            d_local_losses = np.array(d_last_local_loss).ravel()

            # average local weights
            g_global_weights = utils.average_weights(g_local_weights)
            d_global_weights = utils.average_weights(d_local_weights)

            # remove_prefix = "module."
            # g_global_weights = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in g_global_weights.items()}
            # d_global_weights = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in d_global_weights.items()}

            # update global weights
            g_global_model.load_state_dict(g_global_weights)
            d_global_model.load_state_dict(d_global_weights)

            g_loss_avg = sum(g_local_losses) / len(g_local_losses)
            d_loss_avg = sum(d_local_losses) / len(d_local_losses)

            g_train_loss.append(g_loss_avg)
            d_train_loss.append(d_loss_avg)

            g_train_loss_array = np.array(g_train_loss)
            d_train_loss_array = np.array(d_train_loss)

            np.savetxt("C:\\Users\\danie\\OneDrive\\Desktop\\fedgan6\\Generator loss for FedAvg.txt", g_train_loss_array)
            np.savetxt("C:\\Users\\danie\\OneDrive\\Desktop\\fedgan6\\Discriminator loss for FedAvg.txt", d_train_loss_array)

        plt.figure(figsize=(10, 5))
        plt.plot(range(args.epochs_global), g_train_loss, label="Generator")
        plt.plot(range(args.epochs_global), d_train_loss, label="Discriminator")
        plt.xlabel("Global rounds", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.xticks(np.arange(0, args.epochs_global, 20), fontsize=12)
        plt.title("Generator and Discriminator Loss for FedAvg")
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.show()
            
    elif args.mode == 'test':
        trainer.test()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=16, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[32, 64, 128], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[64, 128], 64, [128, 1]], help='number of conv filters in the first layer of D') #[128, 64], 128, [128, 64]
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size') #16
    parser.add_argument('--num_iters_local', type=int, default=1000, help='number of total iterations for training D') #200000
    parser.add_argument('--num_iters_decay', type=int, default=10, help='number of iterations for decaying lr') #100000
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--epochs_global', type=int, default=50, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=3, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=1000, help='test model from this step') #200000

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--data_iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    # parser.add_argument('--data_noniid', type=int, default=0, help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')

    # Directories.
    parser.add_argument('--mol_data_dir', type=str, default='C:\\Users\\danie\\OneDrive\\Desktop\\fedgan\\data_smiles\\qm8.dataset')
    parser.add_argument('--log_dir', type=str, default='C:\\Users\\danie\\OneDrive\\Desktop\\fedgan6\\logs')
    parser.add_argument('--model_save_dir', type=str, default='C:\\Users\\danie\\OneDrive\\Desktop\\fedgan6\\models')
    parser.add_argument('--sample_dir', type=str, default='C:\\Users\\danie\\OneDrive\\Desktop\\fedgan6\\samples')
    parser.add_argument('--result_dir', type=str, default='C:\\Users\\danie\\OneDrive\\Desktop\\fedgan6\\results')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10) #10
    parser.add_argument('--sample_step', type=int, default=100)  #1000
    parser.add_argument('--model_save_step', type=int, default=100) #10000
    parser.add_argument('--lr_update_step', type=int, default=100)  #1000

    args = parser.parse_args()
    print(args)
    main(args)
