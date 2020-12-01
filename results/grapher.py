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

def get_plottable(model_name, folder, file_name):
    all_lines = get_all_lines(folder, file_name)
    useful_lines = get_useful_lines(all_lines)
    epochs, train_losses, val_losses = get_tokens(useful_lines)
    
    clean_train_epochs, clean_train_losses = clean_train(epochs, train_losses)
    clean_val_epochs, clean_val_losses = clean_val(epochs, val_losses)

    return {model_name: [clean_train_epochs, clean_train_losses, clean_val_epochs, clean_val_losses]}

def plot(plottables, x_i, y_i, graph_name, debug=True):

    for model_name, values in plottables.items():
        epochs = [float(i) for i in values[x_i]]
        losses = [float(i) for i in values[y_i]]
        plt.plot(epochs, losses, label=model_name)
    
    plt.title(graph_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=6)
    
    if debug:
        plt.show()
    else:
        plt.savefig(graph_name + '.png')

    plt.clf()

def run_models(model_names, folders, file_names, debug=True):
    plottables = {}

    for i in range(0, len(model_names)):
        plottables.update(get_plottable(model_names[i], folders[i], file_names[i]))
    
    plot(plottables, 0, 1, 'Training Results', debug)
    plot(plottables, 2, 3, 'Validation Results', debug)

def run(debug=True):
    model_names = [
        'U-Net', 
        'Dense U-Net', 
        'U-Net++', 
        'U-Net++ Deep Supervision', 
        'NNRET U-Net',
        'R2U-Net',
        'Attention U-Net',
        'Residual Dense U-Net',
        'ResNet'
    ]
    folders = [
        'unet_original/',
        'denseunet/',
        'unetplus/',
        'unetplus_deepsupervision/',
        'nnret/',
        'r2unet/',
        'attunet/',
        'residualdenseunet/',
        'resnet/'
    ]
    file_names = [
        'output.txt',
        'results.txt',
        'results.txt',
        'results.txt',
        'nnrnet_output.txt',
        'results.txt',
        'results.txt',
        'output.txt',
        'output.txt'
    ]

    run_models(model_names, folders, file_names, debug)

run(False)
