from __future__ import print_function, division
from saliency_map_net import SaliencyMapNet, SaliencyUNet, L_cut, Saliency_simple, Saliency_noskip, Saliency_encoder

from data_loader import ChexRays, RSNA_loader


import torch
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from apex import amp
from sklearn.metrics import roc_auc_score

from knoedl import setup_experiment
from knoedl.utils import dynamic_import


def main():
    # amp_handle = amp.init(enabled=True)
    params = setup_experiment(exp_type='training')
    # get base_dir from params file
    base_dir = params['base_dir']
    # note params file has to be given in training/ edit_configuration top right drop down menu

    # init logging
    from knoedl.log.tb_log import TbLogger as knoedl_TbLogger

    # read param file containing training parameters
    running_type = params['running_type']
    num_classes = params['model_params']['num_classes']
    batch_size = params['train']['batch_size']
    epochs = params['train']['epochs']
    patience = params['train']['patience']

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    #### Data Location ####
    if running_type == 'dev':
        data_dir = 'data/hymenoptera_data'
        root_dir = '/media/data1/max_m/thesis/chexray_dev/'
        dev_folder = 'images/'
        dev_csv_path = os.path.join(root_dir, 'Dev_data.csv')
        log_dir = root_dir

        #### Datasets and Dataloaders Generation ####
        image_datasets = {x: ChexRays(csv_dir=root_dir + '{}.csv'.format(x),
                                      root_dir=root_dir,
                                      folder=dev_folder,
                                      transform=data_transforms[x])  # changed to none / data_transforms[x]
                          for x in ['train', 'val']}
    elif running_type == 'normal':
        server = params['server']
        # if server == '95':
        #     root_dir = '/media/data2/data/ChestXray14/small/data'
        #     folder = 'images'
        # elif server =='99':
        #     root_dir = '/media/data2/data/ChestXray-NIHCC'
        #     folder = 'images/'

        csv_dir = '/media/data1/max_m/thesis/RSNA/csv_files/K_fold/unique_pids'
        root_dir = '/media/data2/data/rsna-pneumonia-detection-challenge'
        folder = 'train_images_med_png'
        log_dir = '/media/data1/max_m/logs'
        split_names = {'train': params['split_names'][0], 'val': params['split_names'][1]}

        image_datasets = {x: RSNA_loader(csv_dir=os.path.join(csv_dir, split_names[x]),
                                         root_dir=root_dir,
                                         folder=folder,
                                         num_classes=num_classes,
                                         transform=data_transforms[x])  # changed to none / data_transforms[x]
                          for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4) for x in ['train', 'val']}

    # check if datasets contain the same classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = {x: image_datasets[x].classes for x in ['train', 'val']}
    assert class_names['train'] == class_names[
        'val'], 'validation set does not contain the same classes as training set.' \
                'validation classes = {}, training classes = {}' \
        .format(class_names['train'], class_names['val'])


    #### device assignment ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #### model instantiation, based on architecture mode ####
    model_mode = params['model_params']['model_mode']
    if model_mode == 'saliency':
        # from saliency_map_net import SaliencyMapNet
        model = SaliencyMapNet(num_classes=num_classes,
                               gr=32,
                               resnet_backbone='ResNet50',
                               dense_config='normal')
    elif model_mode == 'unet':
        model = SaliencyUNet(num_classes=num_classes,
                             resnet_backbone='ResNet50')
    elif model_mode == 'cut':
        model = L_cut(num_classes=num_classes,
                             resnet_backbone='ResNet50')
    elif 'simple' in model_mode:
        if '1' in model_mode:
            model = Saliency_simple(num_classes=num_classes,
                            resnet_backbone='ResNet50', mode='mode1')
        elif '2' in model_mode:
            model = Saliency_simple(num_classes=num_classes,
                                resnet_backbone='ResNet50', mode='mode2')
        else:
            model = Saliency_simple(num_classes=num_classes,
                        resnet_backbone='ResNet50')
    elif model_mode == 'noskip':
        model = Saliency_noskip(num_classes=num_classes,
                             resnet_backbone='ResNet50')
    elif model_mode == 'encoder':
        model = Saliency_encoder(num_classes=num_classes,
                             resnet_backbone='ResNet50')
    model = model.to(device)

    #### training ####
    # when training on RSNA use softmax and NLLLoss
    loss_type = dynamic_import(['torch.nn'], params['train']['loss_type'], 'loss')
    loss = loss_type(**params['train']['loss_params'])

    optim_type = dynamic_import(['torch.optim'], params['train']['optim_type'], 'optimizer')
    optim = optim_type(model.parameters(), **params['train']['optim_params'])


    # APEX init
    model, optim = amp.initialize(model, optim, opt_level='O1')
    step_size = params['train']['lr_decay']
    gamma = params['train']['gamma']
    exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)


    #### Initialize model savers and tensorboard logging.####
    # Note that knoedl automatically creates logs for all .py files, and the console log and knoedl version
    knoedl_tb_logger = knoedl_TbLogger(base_dir, count_steps=True)

    #create a models directory in the base_dir that is created by knoedl
    # models_dir is used to save the best models, and a model from each epoch
    models_dir = os.path.join(base_dir, 'models/')
    os.makedirs(models_dir, exist_ok=True)

    #### load model params from previous training ####
    load_pretrained = params['load_pretrained']
    best_dir = params['best_dir']
    best_from = os.path.join(log_dir, best_dir)
    if load_pretrained:
        checkpoint = torch.load(best_from)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # optim.load_state_dict(checkpoint['optimizer_state_dict'])

        ## freeze weights:
        # for child in model.pretrained_resnet.children():
        #     for param in child.parameters():
        #         param.requires_grad = False

        # model.eval()
        # - or -
        model.train()

    # todo : debugging training
    result = train_model(model,
                         loss,
                         optim,
                         scheduler=exp_lr_scheduler,
                         patience=patience,
                         device=device,
                         dataloaders=dataloaders,
                         class_names=class_names,
                         dataset_sizes=dataset_sizes,
                         root_dir=root_dir,
                         epochs=epochs,
                         knoedl_tb_logger=knoedl_tb_logger,
                         log_dir=models_dir,
                         params=params)

    # visualize_model(model, dataloaders, device, class_names, num_images=4)

    return result


def show(tensor, title=None, save_location=None):
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    if save_location is not None:
        save_loc = os.path.join('/media/data1/max_m/thesis/saved_figures/', save_location)  #uncomment if image is to be saved
        plt.savefig(save_loc)
    else:
        plt.pause(0.001)  # comment out if image is to be saved


#todo: transform list of labels into one hot encoding, add this to ChexRay as a class method
def convert_labels_to_tensor(class_names, sample):
    labels_dict = {label: idx for idx, label in enumerate(class_names['train'])}
    predictions = sample['annotations']['Finding Labels']
    # convert sample labels to list of labels

    idx = []
    batch_size = sample['image'].shape[0]
    num_classes = len(class_names['train'])
    one_hot = torch.zeros((batch_size, num_classes))
    for batch in range(batch_size):
        idx = []
        if isinstance(predictions[batch], str):
            predictions[batch] = [predictions[batch]]
        for i, label in enumerate(predictions[batch]):
            idx.append(labels_dict[label])

        one_hot[batch, idx] = 1
    return one_hot


## TOdo: start a training with basic parameters, little data augmentation etc. % done
# define training function
def train_model(model,
                criterion,
                optimizer,
                scheduler,
                patience,
                device,
                dataloaders,
                class_names,
                dataset_sizes,
                root_dir,
                knoedl_tb_logger,
                epochs,
                log_dir,
                params,
                ):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_roc_auc = 0.0


    with knoedl_tb_logger:
        for epoch in range(0, epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    # update learning_rate scheduler:
                    scheduler.step()

                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                counter = 0

                if phase == 'val':
                    labels_epoch = []
                    outs_epoch = []

                # running_roc_auc_score = 0.0
                # running_add = 0.0
                ## use a array to save the outputs

                # Iterate over data.
                if phase == 'val':
                    print('Wait !')
                for i, sample in enumerate(dataloaders[phase]):
                    inputs = sample['image'].to(device)

                    # formatting labels, where multiple labels are present, create a list
                    # for roc auc score required in one hot labels
                    labels = sample['annotations'].squeeze()
                    #convert labels to int encoding if loss function is CrossEntropyLoss
                    if len(labels.shape) == 2:
                        labels_int = labels.max(1)[1]
                    else:
                        labels_int = labels.max()


                    labels = labels.to(device)
                    labels_int = labels_int.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        _, outputs = model(inputs)
                        #change labels passed to loss if BCELoss or NLLLoss
                        if params['train']['loss_type'] == 'NLLLoss':
                            labels = labels.type(dtype=torch.long)
                            if labels_int.dim() != 0:
                                loss = criterion(outputs, labels_int)  # assumes that labels are given as one hot encoded
                            else:
                                labels_int = labels_int.unsqueeze(0).type(dtype=torch.long)
                                loss = criterion(outputs, labels_int)
                        else:
                            loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()

                            optimizer.step()

                    if phase == 'val':
                        # append outputs and labels of batch to overall cached outputs and labels of epoch
                        with torch.no_grad():
                            outs_epoch.append(outputs.cpu().numpy())
                            # if BCELoss, labels are one hot encoded
                            labels_epoch.append(labels.cpu().numpy())

                    # logging...
                    if i % 100 == 0:
                        if phase =='val':
                            print('current i is: {}/{} \n'
                                  ' current loss is {} \n'.format(i,
                                                                       len(dataloaders[phase].dataset)/
                                                                       dataloaders[phase].batch_size,
                                                                       loss.item(),))
                        else:
                            print('current i is: {}/{} \n'
                                  ' current loss is {}'.format(i,
                                                                       len(dataloaders[phase].dataset)/
                                                                       dataloaders[phase].batch_size,
                                                                       loss.item()))
                    running_loss += loss.item() * inputs.size(0)

                #calculate loss
                epoch_loss = running_loss / dataset_sizes[phase]

                # write TensorBoard output...
                if phase == 'val':
                    # save predicted classes for statistics
                    outs_epoch = np.concatenate(outs_epoch, axis=0)
                    labels_epoch = np.concatenate(labels_epoch, axis=0)

                    roc_auc = roc_auc_score(y_true=labels_epoch, y_score=outs_epoch, average="macro")

                    if model.pooling.beta.shape[0] == 3:
                        current_lr = optimizer.param_groups[0]['lr']
                        res_list = [model.pooling.beta[0], model.pooling.beta[1], model.pooling.beta[2], current_lr, epoch, epoch_loss, roc_auc]
                        tb_tags = ['beta0', 'beta1', 'beta2', 'lr', 'epoch', 'val_epoch_loss', 'roc_auc']
                        knoedl_tb_logger.add_scalars(tb_tags, res_list, step=epoch)
                        print('{} Loss: {:.4f} roc_auc: {:.4f}'.format(
                            phase, epoch_loss, roc_auc))
                    else:
                        current_lr = optimizer.param_groups[0]['lr']
                        res_list = [model.pooling.beta, current_lr, epoch, epoch_loss, roc_auc]
                        tb_tags = ['beta0', 'lr', 'epoch', 'val_epoch_loss', 'roc_auc']
                        knoedl_tb_logger.add_scalars(tb_tags, res_list, step=epoch)
                        print('{} Loss: {:.4f} roc_auc: {:.4f}'.format(
                            phase, epoch_loss, roc_auc))

                elif phase == 'train':
                    res_list = [epoch, epoch_loss]
                    tb_tags = ['epoch', 'train_epoch_loss']
                    knoedl_tb_logger.add_scalars(tb_tags, res_list, step=epoch)
                    print('{} Loss: {:.4f}'.format(
                        phase, epoch_loss))


            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_wts = copy.deepcopy(model.state_dict())

                # save model weights to file
                save_at = os.path.join(log_dir, 'checkpoint{}_{}.pth.tar'.format(epoch, phase))
                # torch.save(model.state_dict(), save_at)

                # save more checkpoints
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_at)

                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best roc_auc: {:4f}'.format(best_roc_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_best = os.path.join(log_dir, 'best.pth.tar')

    # save the best model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_best)

    return model

if __name__ == '__main__':
    main()

print('done')


