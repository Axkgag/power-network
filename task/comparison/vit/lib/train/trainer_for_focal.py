import torch
import numpy as np
import os

# def fit(train_loader, 
#         val_loader, 
#         model, 
#         loss_fn, 
#         ohem_fn, 
#         optimizer, 
#         scheduler, 
#         n_epochs, 
#         cuda, 
#         log_interval, 
#         metrics=[],
#         start_epoch=0, 
#         model_bn_path='model_',
#         cache_dir="cache",
#         train_status_sender=None,
#         train_controller=None):
#     """
#     Loaders, model, loss function and metrics should work together for a given task,
#     i.e. The model should be able to process data output of loaders,
#     loss function should process target output of loaders and outputs from the model

#     Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
#     Siamese network: Siamese loader, siamese model, contrastive loss
#     Online triplet learning: batch loader, embedding model, online triplet loss
#     """
#     for epoch in range(0, start_epoch):
#         scheduler.step()
#     last_epoch=-1
#     for epoch in range(start_epoch, n_epochs):
        
#         # Train stage
#         train_loss, metrics = train_epoch(train_loader, model, loss_fn, ohem_fn, optimizer, cuda, log_interval, metrics)

#         scheduler.step()

#         message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
#         # if train_status_sender :
#         #     train_status_sender.send(data={"epoch":epoch,"train_loss":train_loss,"train_accuracy":0,"val_loss":0,"val_accuracy":0},method="train_stat")
#         # print(message)
#         if (epoch+1) % 30 == 0 or (epoch == (n_epochs-1)):
#             if last_epoch>0 and os.path.exists(cache_dir+ "/"+str(last_epoch) +'.pkl') :
#                 os.remove(cache_dir+ "/"+str(epoch) +'.pkl')
#             torch.save(model.state_dict(), cache_dir+"/"+str(epoch) +'.pkl',_use_new_zipfile_serialization=False)
#             last_epoch=epoch
#         # if train_controller and train_controller.stop_signal:
#         #     print("stopped at", epoch)
#         #     break
    
#     torch.save(model.state_dict(), model_bn_path,_use_new_zipfile_serialization=False)

def train_epoch(train_loader, model, loss_fn, ohem_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = (tuple(d.cuda() for d in data),)
            if target is not None:
                target = target.cuda()
        else:
            data = (tuple(d for d in data),)

        optimizer.zero_grad()
        outputs = model(*data)

        loss_outputs = 0
        if target is not None:
            loss_outputs = loss_fn(outputs, target)
        
        loss_output = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        loss = loss_output
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0][0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics
