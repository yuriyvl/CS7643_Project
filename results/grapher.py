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

def clean_train(epochs, train_losses):
    new_epochs = []
    new_train_losses = []

    for i in range(len(epochs)):
        if '%' not in epochs[i] and len(train_losses[i]) < 7:
            new_epochs.append(epochs[i])
            new_train_losses.append(train_losses[i])
    
    return new_epochs, new_train_losses

def clean_val(epochs, val_losses):
    new_epochs = []
    new_val_losses = []

    for i in range(len(val_losses)):
        if val_losses[i] is not None and len(val_losses[i]) < 7 and '%' not in epochs[i]:
            new_epochs.append(epochs[i])
            new_val_losses.append(val_losses[i])
    
    return new_epochs, new_val_losses

def plot(epochs, losses, y_label, name, debug=True):
    epochs = [float(i) for i in epochs]
    losses = [float(i) for i in losses]

    _, ax = plt.subplots()
    plt.plot(epochs, losses)
    
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    
    if debug:
        plt.show()
    else:
        plt.savefig(name + '_' + y_label + '.png', bbox_inches='tight')

def run_model(model_name, folder, file_name, debug=True):
    all_lines = get_all_lines(folder, file_name)
    useful_lines = get_useful_lines(all_lines)
    epochs, train_losses, val_losses = get_tokens(useful_lines)
    
    clean_train_epochs, clean_train_losses = clean_train(epochs, train_losses)
    clean_val_epochs, clean_val_losses = clean_val(epochs, val_losses)
    
    plot(clean_train_epochs, clean_train_losses, 'Training Loss', model_name, debug)
    plot(clean_val_epochs, clean_val_losses, 'Validation Loss', model_name, debug)

def run(debug=True):
    run_model('U-Net', 'unet_original/', 'output.txt', debug)
    run_model('Dense U-Net', 'denseunet/', 'results.txt', debug)
    run_model('U-Net+', 'unetplus/', 'results.txt', debug)
    run_model('U-Net+ Deep Supervision', 'unetplus_deepsupervision/', 'results.txt', debug)
    run_model('NNRET', 'nnret/', 'nnrnet_output.txt', debug)

run(False)
