from pathlib import Path
import re

def get_all_lines():
    path = Path("nnret/") / 'nnrnet_output.txt'
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

all_lines = get_all_lines()
useful_lines = get_useful_lines(all_lines)
epochs, train_losses, val_losses = get_tokens(useful_lines)
