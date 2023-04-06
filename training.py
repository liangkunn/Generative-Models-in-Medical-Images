import numpy as np
import torch
import time
import torchvision
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms as tt
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def Linear_GAN_training(data, model, 
                     num_epochs, 
                     latent_dim, 
                     optim_discr,
                     optim_gen,
                     device):

    loss_function = F.binary_cross_entropy_with_logits

    log_dict={'train_generator_loss_per_batch': [],
            'train_discriminator_loss_per_batch': [],
            'train_discriminator_real_acc_per_batch': [],
            'train_discriminator_fake_acc_per_batch': [],
            'images_from_noise_per_epoch': []
            }
    # fixed noise is used to generate images when generator is trained
    fixed_noise = torch.randn(128, latent_dim, 1, 1, device=device)
    start_time = time.time()

    logging_interval=100
    
    for step in range(num_epochs):
        # for step in range(num_epoch):
        model.train()
        for batch_indx, images in enumerate(data): # with batch_index 0-16, and iamges shape NCHW (16,3,28,28)
            batch_size = images.size(0) # how many images in one batch
            
            # real images
            real_images = images.to(device)
            real_labels = torch.ones(batch_size, device=device)
        
            # fake images
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=device)
            flipped_fake_labels = real_labels # flip the labels for the generator training
            
            #######################################################
            ## train the discriminator first, from the GAN paper
            #######################################################
            
            optim_discr.zero_grad()
            
            # calculate loss on real images
            real_img_pred = model.discriminator_forward(real_images).view(-1)
            real_img_loss = loss_function(real_img_pred, real_labels)
            
            # caculate loss on fake images
            fake_img_pred = model.discriminator_forward(fake_images).view(-1) # [batch_n*1] --> batch_n
            fake_img_loss = loss_function(fake_img_pred, fake_labels) 
            
            # backward propagation
            discriminator_loss = 0.5*(real_img_loss + fake_img_loss)
            discriminator_loss.backward(retain_graph=True)
            
            # performs a parameter update based on the current gradient 
            optim_discr.step()
            
            #######################################################
            ## train the generator
            #######################################################
            
            optim_gen.zero_grad()
            
            # calculate loss on generate images
            fake_img_pred = model.discriminator_forward(fake_images).view(-1)
            # tricks applied here, flip the label for fake/real iamges to use gradient descent instead of modifying the loss function
            generator_loss = loss_function(fake_img_pred, flipped_fake_labels)
            generator_loss.backward(retain_graph=True)
            
            # performs a parameter update based on the current gradient 
            
            optim_gen.step()
            
            #######################################################
            ## logging
            #######################################################   
            log_dict['train_generator_loss_per_batch'].append(generator_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discriminator_loss.item())
            
            predicted_labels_real = torch.where(real_img_pred.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(fake_img_pred.detach() > 0., 1., 0.) # because we have flipped lables
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            if not batch_indx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                     % (step+1, num_epochs, batch_indx, 
                        len(data), generator_loss.item(), discriminator_loss.item()))

        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(
            torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))    
    return log_dict

def VAE_training(data, model, num_epochs, optimizer, device,
                 logging_interval=100, 
                 reconstruction_term_weight=1,
                 save_model=None):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}
    
    # use MSE as loss function
    loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, images in enumerate(data):

            images = images.to(device)

            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(images)
            
            # total loss = reconstruction loss + KL divergence
            #kl_divergence = (0.5 * (z_mean**2 + 
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
    
            pixelwise = loss_fn(decoded, images, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
            
            loss = reconstruction_term_weight * pixelwise + kl_div
            
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                      % (epoch+1, num_epochs, batch_idx,
                          len(data), loss))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict



# a train function which will be included in another .py file
def CWGAN_training(data, model, 
                   num_epochs, 
                   latent_dim, 
                   optim_discr,
                   optim_gen,
                   device,
                   logging_interval=100,
                   gradient_penalty=True,
                   discr_iter_per_generator_iter=5,
                   gradient_penalty_weight=10,
                   save_model=None):

    # Cross entropy for multi-class
    # loss_function = F.cross_entropy 
    # Binary Cross Entropy for binary class

    # implement wasserstain loss
    def loss_function(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    log_dict={'train_generator_loss_per_batch': [],
            'train_discriminator_loss_per_batch': [],
            'images_from_noise_per_epoch': []
            }
    if gradient_penalty:
        log_dict['train_gradient_penalty_loss_per_batch'] = []

    # fixed noise is used to generate images when generator is trained
    fixed_noise = torch.randn(128, latent_dim, 1, 1, device=device)
    # fixed_noise = torch.randn(128, latent_dim, device=device)
    start_time = time.time()

    # in Wgan, discriminator is going to be trained multiple times, default here is 5
    # skip_generator = 1

    for epoch in range(num_epochs):
        # need to print the image out
        # for step in range(num_epoch):
        model.train()
        for batch_indx, (images, labels) in enumerate(data): # with batch_index 0-16, and iamges shape NCHW (16,3,28,28)

            batch_size = images.size(0) # how many images in one batch
            
            # real images
            real_images = images.to(device)
            cell_labels = labels.view(-1)
            cell_labels = cell_labels.to(torch.int64).to(device)
            real_labels = torch.ones(batch_size, device=device)
            # random_labels = torch.randint(0, 7, (batch_size,), device=device) # command out option 1

            # fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = model.generator_forward(noise, cell_labels) # option 2
            # fake_images = model.generator_forward(noise, random_labels) # command out option 1

            # fake labels here are -1 instead of 0
            fake_labels = -real_labels # -1
            flipped_fake_labels = real_labels # flip the labels for the generator training, which is 1
            
            #######################################################
            ## same training process as DCGAN, but add gradient penalty
            ## with discrimator trained 5 times / generator 1 time
            ## we train the discriminator first
            #######################################################
            for i in range(discr_iter_per_generator_iter):
                optim_discr.zero_grad()
                
                # calculate loss on real images
                real_img_pred = model.discriminator_forward(real_images, cell_labels).view(-1)
                real_img_loss = loss_function(real_img_pred, real_labels) 
                
                # caculate loss on fake images
                fake_img_pred = model.discriminator_forward(fake_images.detach(), cell_labels).view(-1) # option 2
                # fake_img_pred = model.discriminator_forward(fake_images.detach(), random_labels).view(-1) # # command out option 1
                fake_img_loss = loss_function(fake_img_pred, fake_labels) 
                
                # calculate loss
                discriminator_loss = 0.5*(real_img_loss + fake_img_loss)
                
                # add gradient penalty
                if gradient_penalty:
                    
                    # alpha value U(0,1)
                    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

                    # x_hat = alpha * x + (1 - alpha) x_generated 
                    interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                    interpolated.requires_grad = True
                    # dis(x_hat) for calculating the loss
                    discr_out = model.discriminator_forward(interpolated, cell_labels)
                    # compute gradient of discr_out 
                    grad_values = torch.ones(discr_out.size()).to(device)
                    gradients = torch.autograd.grad(
                        outputs=discr_out,
                        inputs=interpolated,
                        grad_outputs=grad_values,
                        create_graph=True,
                        retain_graph=True)[0]

                    gradients = gradients.view(batch_size, -1)

                    # calc. norm of gradients, adding epsilon to prevent 0 values
                    epsilon = 1e-13
                    gradients_norm = torch.sqrt(
                        torch.sum(gradients ** 2, dim=1) + epsilon)

                    gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * gradient_penalty_weight
                    
                    discriminator_loss += gp_penalty_term    

                    log_dict['train_gradient_penalty_loss_per_batch'].append(gp_penalty_term.item())

                # backward propagation
                discriminator_loss.backward(retain_graph=True)
                
                # performs a parameter update based on the current gradient 
                optim_discr.step()
                
                # if it's not gradient penalty, then do regular wasserstain distance
                if not gradient_penalty:
                    for p in model.discriminator.parameters():
                        # clamp the weights 
                        p.data.clamp_(-0.01, 0.01)           
                
                # skip_generator += 1
                # print('discriminator done')
            #######################################################
            ## train the generator
            #######################################################

            optim_gen.zero_grad()
            
            # calculate loss on generate images
            fake_images = model.generator_forward(noise, cell_labels) # option 2
            fake_img_pred = model.discriminator_forward(fake_images, cell_labels).view(-1) # option 2

            # calculate loss on generate images
            # fake_images = model.generator_forward(noise, random_labels) # option 1
            # fake_img_pred = model.discriminator_forward(fake_images, random_labels).view(-1) # option 1

            # flip the label for fake iamges to use gradient descent instead of modifying the loss function
            generator_loss = loss_function(fake_img_pred, flipped_fake_labels) # tricks applied here, fool discriminator
            generator_loss.backward(retain_graph=True)
                
            # performs a parameter update based on the current gradient 
            optim_gen.step()
            
            # skip_generator = 1
            # gener_loss = torch.tensor(0.)    
            # print('generator done')
            # break
            #######################################################
            ## logging
            #######################################################   
            log_dict['train_generator_loss_per_batch'].append(generator_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discriminator_loss.item())    
            
            if not batch_indx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                    # % (epoch+1, 100, batch_indx, 
                    % (epoch+1, num_epochs, batch_indx, 
                        len(data), generator_loss.item(), discriminator_loss.item()))

        with torch.no_grad():
            fake_labels_ = torch.randint(0, 7, (128, ), device=device)
            fake_images = model.generator_forward(fixed_noise, fake_labels_).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(
            torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
        
        if epoch % 20 ==0:
        # plot the images for each epoch, no good then cut it off
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            plt.figure(figsize=(12, 12))
            plt.axis('off')
            plt.title(f'Generated images at epoch {epoch}')
            plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][epoch], (1, 2, 0)))
            plt.show()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))  

    if save_model is not None:
        torch.save(model.state_dict(), save_model) 
    return log_dict

# a train function which will be included in another .py file
def WGAN_training(data, model, 
                   num_epochs, 
                   latent_dim, 
                   optim_discr,
                   optim_gen,
                   device,
                   logging_interval=100,
                   gradient_penalty=False,
                   discr_iter_per_generator_iter=5,
                   gradient_penalty_weight=10,
                   save_model=None):

    # Cross entropy for multi-class
    # loss_function = F.cross_entropy 
    # Binary Cross Entropy for binary class

    # implement wasserstain loss
    def loss_function(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    log_dict={'train_generator_loss_per_batch': [],
            'train_discriminator_loss_per_batch': [],
            'train_discriminator_real_acc_per_batch': [],
            'train_discriminator_fake_acc_per_batch': [],
            'images_from_noise_per_epoch': []
            }
    if gradient_penalty:
        log_dict['train_gradient_penalty_loss_per_batch'] = []

    # fixed noise is used to generate images when generator is trained
    # fixed_noise = torch.randn(128, latent_dim, 1, 1, device=device)
    fixed_noise = torch.randn(128, latent_dim, device=device)
    start_time = time.time()

    # in Wgan, discriminator is going to be trained multiple times, default here is 5
    skip_generator = 1

    for epoch in range(num_epochs):
        # need to print the image out
        # for step in range(num_epoch):
        model.train()
        for batch_indx, images in enumerate(data): # with batch_index 0-16, and iamges shape NCHW (16,3,28,28)
            batch_size = images.size(0) # how many images in one batch
            
            # real images
            real_images = images.to(device)
            real_labels = torch.ones(batch_size, device=device)

            # fake images
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = model.generator_forward(noise)

            # fake labels here are -1 instead of 0
            fake_labels = -real_labels # -1
            flipped_fake_labels = real_labels # flip the labels for the generator training, 1
            
            #######################################################
            ## same training process as DCGAN, but add gradient penalty
            ## with discrimator trained 5 times / generator 1 time
            ## we train the discriminator first
            #######################################################
            
            optim_discr.zero_grad()
            
            # calculate loss on real images
            real_img_pred = model.discriminator_forward(real_images).view(-1)

            # label smoothing here to avoid discriminator being too strong
            real_img_loss = loss_function(real_img_pred, real_labels) 
            
            # caculate loss on fake images
            fake_img_pred = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_img_loss = loss_function(fake_img_pred, fake_labels) 
            
            # calculate loss
            discriminator_loss = 0.5*(real_img_loss + fake_img_loss)
            
            # add gradient penalty
            if gradient_penalty:
                
                # alpha value U(0,1)
                alpha = torch.rand(batch_size, 1, 1, 1).to(device)

                # x_hat = alpha * x + (1 - alpha) x_generated 
                interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                interpolated.requires_grad = True
                # dis(x_hat) for calculating the loss
                discr_out = model.discriminator_forward(interpolated)
                # compute gradient of discr_out 
                grad_values = torch.ones(discr_out.size()).to(device)
                gradients = torch.autograd.grad(
                    outputs=discr_out,
                    inputs=interpolated,
                    grad_outputs=grad_values,
                    create_graph=True,
                    retain_graph=True)[0]

                gradients = gradients.view(batch_size, -1)

                # calc. norm of gradients, adding epsilon to prevent 0 values
                epsilon = 1e-13
                gradients_norm = torch.sqrt(
                    torch.sum(gradients ** 2, dim=1) + epsilon)

                gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * gradient_penalty_weight
                
                discriminator_loss += gp_penalty_term    

                log_dict['train_gradient_penalty_loss_per_batch'].append(gp_penalty_term.item())

            # backward propagation
            discriminator_loss.backward(retain_graph=True)
            
            # performs a parameter update based on the current gradient 
            optim_discr.step()
            
            # if it's not gradient penalty, then do regular wasserstain distance
            if not gradient_penalty:
                for p in model.discriminator.parameters():
                    # clamp the weights 
                    p.data.clamp_(-0.01, 0.01)           
            
            
            #######################################################
            ## train the generator
            #######################################################
            if skip_generator <= discr_iter_per_generator_iter:

                optim_gen.zero_grad()
            
                # calculate loss on generate images
                fake_img_pred = model.discriminator_forward(fake_images).view(-1)
                # flip the label for fake iamges to use gradient descent instead of modifying the loss function
                generator_loss = loss_function(fake_img_pred, flipped_fake_labels) # tricks applied here, fool discriminator
                generator_loss.backward(retain_graph=True)
                
                skip_generator += 1
                
                # performs a parameter update based on the current gradient 
                optim_gen.step()
    
            else:
                skip_generator = 1
                generator_loss = torch.tensor(0.)    

            #######################################################
            ## logging
            #######################################################   
            log_dict['train_generator_loss_per_batch'].append(generator_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discriminator_loss.item())
            
            predicted_labels_real = torch.where(real_img_pred.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(fake_img_pred.detach() > 0., 1., 0.) # because we have flipped lables
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            if not batch_indx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                    # % (epoch+1, 100, batch_indx, 
                    % (epoch+1, num_epochs, batch_indx, 
                        len(data), generator_loss.item(), discriminator_loss.item()))

        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(
            torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
        
        if epoch % 20 == 0:
            # plot the images for each epoch, no good then cut it off
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.title(f'Generated images at epoch {epoch}')
            plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][epoch], (1, 2, 0)))
            plt.show()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))   
    if save_model is not None:
        torch.save(model.state_dict(), save_model) 
    return log_dict

# a train function which will be included in another .py file
def W_GAN_training(data, model, 
                   num_epochs, 
                   latent_dim, 
                   optim_discr,
                   optim_gen,
                   device,
                   logging_interval=100,
                   gradient_penalty=True,
                   discr_iter_per_generator_iter=5,
                   gradient_penalty_weight=10,
                   save_model=None):

    # Cross entropy for multi-class
    # loss_function = F.cross_entropy 
    # Binary Cross Entropy for binary class

    # implement wasserstain loss
    def loss_function(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    log_dict={'train_generator_loss_per_batch': [],
            'train_discriminator_loss_per_batch': [],
            'train_discriminator_real_acc_per_batch': [],
            'train_discriminator_fake_acc_per_batch': [],
            'images_from_noise_per_epoch': []
            }
    if gradient_penalty:
        log_dict['train_gradient_penalty_loss_per_batch'] = []

    # fixed noise is used to generate images when generator is trained
    # fixed_noise = torch.randn(128, latent_dim, 1, 1, device=device)
    fixed_noise = torch.randn(128, latent_dim, device=device)
    start_time = time.time()

    # in Wgan, discriminator is going to be trained multiple times, default here is 5
    skip_generator = 1

    for epoch in range(num_epochs):
        # need to print the image out
        # for step in range(num_epoch):
        model.train()
        for batch_indx, images in enumerate(data): # with batch_index 0-16, and iamges shape NCHW (16,3,28,28)
            batch_size = images.size(0) # how many images in one batch
            
            # real images
            real_images = images.to(device)
            real_labels = torch.ones(batch_size, device=device)

            # fake images
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = model.generator_forward(noise)

            # fake labels here are -1 instead of 0
            fake_labels = -real_labels # -1
            flipped_fake_labels = real_labels # flip the labels for the generator training, 1
            
            #######################################################
            ## same training process as DCGAN, but add gradient penalty
            ## with discrimator trained 5 times / generator 1 time
            ## we train the discriminator first
            #######################################################
            
            for i in range(discr_iter_per_generator_iter):
                optim_discr.zero_grad()
                
                # calculate loss on real images
                real_img_pred = model.discriminator_forward(real_images).view(-1)

                # label smoothing here to avoid discriminator being too strong
                real_img_loss = loss_function(real_img_pred, real_labels) 
                
                # caculate loss on fake images
                fake_img_pred = model.discriminator_forward(fake_images.detach()).view(-1)
                fake_img_loss = loss_function(fake_img_pred, fake_labels) 
                
                # calculate loss
                discriminator_loss = 0.5*(real_img_loss + fake_img_loss)
                
                # add gradient penalty
                if gradient_penalty:
                    
                    # alpha value U(0,1)
                    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

                    # x_hat = alpha * x + (1 - alpha) x_generated 
                    interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                    interpolated.requires_grad = True
                    # dis(x_hat) for calculating the loss
                    discr_out = model.discriminator_forward(interpolated)
                    # compute gradient of discr_out 
                    grad_values = torch.ones(discr_out.size()).to(device)
                    gradients = torch.autograd.grad(
                        outputs=discr_out,
                        inputs=interpolated,
                        grad_outputs=grad_values,
                        create_graph=True,
                        retain_graph=True)[0]

                    gradients = gradients.view(batch_size, -1)

                    # calc. norm of gradients, adding epsilon to prevent 0 values
                    epsilon = 1e-13
                    gradients_norm = torch.sqrt(
                        torch.sum(gradients ** 2, dim=1) + epsilon)

                    gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * gradient_penalty_weight
                    
                    discriminator_loss += gp_penalty_term    

                    log_dict['train_gradient_penalty_loss_per_batch'].append(gp_penalty_term.item())

            # backward propagation
            discriminator_loss.backward(retain_graph=True)
            
            # performs a parameter update based on the current gradient 
            optim_discr.step()
            
            # if it's not gradient penalty, then do regular wasserstain distance
            if not gradient_penalty:
                for p in model.discriminator.parameters():
                    # clamp the weights 
                    p.data.clamp_(-0.01, 0.01)           
            
            
            #######################################################
            ## train the generator
            #######################################################

            optim_gen.zero_grad()
            
            # calculate loss on generate images
            fake_img_pred = model.discriminator_forward(fake_images).view(-1)
            # flip the label for fake iamges to use gradient descent instead of modifying the loss function
            generator_loss = loss_function(fake_img_pred, flipped_fake_labels) # tricks applied here, fool discriminator
            generator_loss.backward(retain_graph=True)
 
            # performs a parameter update based on the current gradient 
            optim_gen.step()

            #######################################################
            ## logging
            #######################################################   
            log_dict['train_generator_loss_per_batch'].append(generator_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discriminator_loss.item())
            
            predicted_labels_real = torch.where(real_img_pred.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(fake_img_pred.detach() > 0., 1., 0.) # because we have flipped lables
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            if not batch_indx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                    % (epoch+1, num_epochs, batch_indx, 
                        len(train), generator_loss.item(), discriminator_loss.item()))

        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(
            torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
        
        if epoch % 20 == 0:
            # plot the images for each epoch, no good then cut it off
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.title(f'Generated images at epoch {epoch}')
            plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][epoch], (1, 2, 0)))
            plt.show()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))   
    if save_model is not None:
        torch.save(model.state_dict(), save_model) 
    return log_dict