# import modules
from script.Train_data import augs, transfms
from script.train import fit
from script.visualize import plot_acc_loss, plot_loss
from script.utils import load_checkpoint, seed_everything, ComboLoss, calculate_metrics, set_dataloader, set_model, set_model
from script.test import test
from script.predict import dataloaderPre, predict, postprocess

# import lib
import argparse
import torch


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default= 50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--checkpoint_path', default='./model/checkpoint.pt', type=str)
    parser.add_argument('--model_path', default='./model/UNet.pt', type=str)
    parser.add_argument('--first_train', default=1, type=int)
    opt = parser.parse_args()
    return opt


def train( device, model, optimizer, augs, transfms, start_epochs, end_epochs, batch_size, checkpoint_path, model_path, first = True):
    if first == 1:    
        print('Here')     
        train_dataloader, val_dataloader, _ = set_dataloader(batch_size, augs, transfms)
    else:
        print('No')
        model, optimizer, start_epochs, train_list, val_list = load_checkpoint(checkpoint_path, model, optimizer)
    fit(model, train_dataloader, val_dataloader, optimizer, scheduler, start_epochs, end_epochs, loss_fn, calculate_metrics, checkpoint_path, model_path, device, train_list, val_list)


if __name__ == '__main__':
    seed = 262022
    seed_everything(seed)

    opt = get_opt()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    model, optimizer, scheduler, loss_fn = set_model(device, opt.lr)

    # # TRAIN
    # train(device, model, optimizer, augs, transfms, opt.start_epoch, opt.end_epoch, opt.batch_size, opt.checkpoint_path, opt.model_path, opt.first_train)

    # # LOAD CHECKPOINT
  

    # # VE LOSS_ACC
    # plot_acc_loss(train_loss, val_loss, train_acc, val_acc, './visualize/testloss_acc.png')  

    # TEST 
    # model.load_state_dict(torch.load(opt.model_path))
    # _,_, test_dataloader = set_dataloader(opt.batch_size, augs, transfms)
    # image, y_true, y_pred, test_dict = test(test_dataloader, device, model, calculate_metrics) 
    # print(test_dict)
    
    # with open('./visualize/test_result.txt', 'w') as wf:
    #     wf.writelines(str(test_dict.items()))

    # PREDICT 
    model.load_state_dict(torch.load(opt.model_path))
    images, y_predict = predict(dataloaderPre(opt.batch_size, transfms)['Positive'], model, device)

    


    

