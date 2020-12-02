import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt 
import torch.fft

# tutorial: https://medium.com/@khodges42/how-to-get-started-diy-neurohacking-with-the-fastmri-dataset-and-why-you-should-6a02a6bb896e
# ctrl+f "exploring the dataset"

# this is fine
def get_data():
    # replace the following with the path to the reconstruction
    path = 'C:\\Users\\dkapoor\\Documents\\School\\proj\\CS7643_Project\\experimental\\unet\\unet\\unet_demo\\reconstructions\\file1000022_v2.h5'

    # crack open the file
    hf = h5py.File(path, 'r')
    # some debugging stuff
    print('keys:', list(hf.keys())) # prints ['reconstruction']
    print('atrrs:', list(hf.attrs)) # prints []
    # grab the data
    data = hf['reconstruction'][()]
    # more debugging stuff
    print(data.dtype) # prints float32
    print(data.shape) # prints (37, 320, 320)

    return data

# NEEDS HELP
def run():
    data = get_data()

    # convert the data to a tensor -> straight from tutorial (i just extracted what the utility method does)
    data_tensor = torch.from_numpy(data)
    
    # *** this is 100% where an issue is ***
    # ifft2 utility function from data.transforms only applies to complex inputs so we can't use it (assertion will fail)
    # so what i tried to do here is call the one from torch.ifft
    # reference: https://pytorch.org/docs/stable/generated/torch.ifft.html
    # calling it with a second param of 0, 1, 2 looks even worse
    # i know the method is meant for complex-to-complex but i couldnt find a real-to-complex one to use
    # i even tried casting to a complex64 type before passing it into this function but i got the same results
    data_transformed = torch.fft.ifft(data_tensor)
    
    # * maybe this is an issue too? *
    # we can't call complex_abs() from data.transforms because again, it only applies to complex input types
    # so what i'm doing here is just normal abs()
    data_abs = torch.abs(data_transformed)

    # root mean sum of squares
    # i know this is right, i just extracted the utility method's inner processing
    data_rss = torch.sqrt((data_abs ** 2).sum(0))

    # show the output, i know this is right too
    plt.imshow(np.abs(data_rss.numpy()), cmap='gray')
    plt.show()

    # what you will see is a black square

# shows something, but doesn't look right
def run_partial():
    data = get_data()
    data_tensor = torch.from_numpy(data)

    # skip ifft() and apply abs() directly to the data tensor
    data_abs = torch.abs(data_tensor)

    # the rest is the same
    data_rss = torch.sqrt((data_abs ** 2).sum(0))
    plt.imshow(np.abs(data_rss.numpy()), cmap='gray')
    plt.show()

    # what you will see looks kind of like a skull

# the fix needs to be made in run(), see comments in method
run()

# the run_partial() is what i got to work, but i skipped over ifft
# uncomment and run this instead of run() to see how it looks
#run_partial()