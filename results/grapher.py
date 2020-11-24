from pathlib import Path
import re
import matplotlib.pyplot as plt

def get_all_lines(folder, name):
    path = Path(folder) / name
    opened = open(path, encoding='utf8')
    return opened.readlines()

def get_useful_lines(all_lines):
    useful_lines = []

    for line in all_lines:
        if line.startswith('Epoch') and '100%' in line and 'top' in line:
            useful_lines.append(line)

    return useful_lines

def get_tokens(lines):
    epochs = []
    train_losses = []
    val_losses = []

    for line in lines:
        epoch = re.search('Epoch (.+?): 100%', line).group(1)
        train_loss = re.search('it/s.*loss=(.+?), v_num', line).group(1)
        val_loss = re.search('val_loss=(.+?)\]', line)

        epochs.append(epoch)
        train_losses.append(train_loss)
        
        if val_loss:
            val_losses.append(val_loss.group(1))
        else:
            val_losses.append(None)

    return epochs, train_losses, val_losses

def plot(epochs, losses, ylabel):
    _, ax = plt.subplots()
    plt.plot(epochs, losses)
    
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 3 != 0:
            label.set_visible(False)
    
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.gca().invert_yaxis()
    plt.show()

def clean_val(epochs, val_losses):
    new_epochs = []
    new_val_losses = []

    for i in range(len(val_losses)):
        if val_losses[i] is not None:
            new_epochs.append(epochs[i])
            new_val_losses.append(val_losses[i])
    
    return new_epochs, new_val_losses

def run():
    unet_all_lines = get_all_lines('unet_original/', 'output.txt')
    unet_useful_lines = get_useful_lines(unet_all_lines)
    unet_epochs, unet_train_losses, unet_val_losses = get_tokens(unet_useful_lines)
    unet_clean_epochs, unet_clean_val_losses = clean_val(unet_epochs, unet_val_losses)
    
    plot(unet_epochs, unet_train_losses, 'Training Loss')
    plot(unet_clean_epochs, unet_clean_val_losses, 'Validation Loss')

run()
