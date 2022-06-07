import matplotlib.pyplot as plt
import numpy as np


# https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/notebook
def imshow_img_mask_label(img, mask, label):
    """ Imshow for Tensor"""
    sample = []

    for idx in range(8):
        left = img[idx]
        right = mask[idx]
        combined = np.hstack((left, right))
        sample.append(combined)

    for idx in range(0,8,4):  # 0 4 
        plt.figure(figsize = (25,10))

        plt.subplot(2,4, 1+idx)
        plt.imshow(sample[idx], cmap='gray')
        plt.title(label[idx])

        plt.subplot(2,4,2+idx)
        plt.imshow(sample[idx +1], cmap='gray')
        plt.title(label[idx+1])

        plt.subplot(2,4, 3+idx)
        plt.imshow(sample[idx+2], cmap='gray')
        plt.title(label[idx+2])

        plt.subplot(2,4, 4 +idx)
        plt.imshow(sample[idx +3], cmap='gray')
        plt.title(label[idx+3])

        plt.show()

def imshow_img_mask(img, mask, label):
    """ Imshow for Tensor"""
    plt.figure(figsize = (30,10))
    for idx in range(2):        

        plt.subplot(2,8, 1+idx*8)
        plt.imshow(img[idx], cmap='gray')
        plt.title(label[idx])

        plt.subplot(2,8, 2+idx*8)
        plt.imshow(mask[idx], cmap='binary')
        

        plt.subplot(2,8,3+idx*8)
        plt.imshow(img[idx +1], cmap='gray')
        plt.title(label[idx+1])

        plt.subplot(2,8,4+idx*8)
        plt.imshow(mask[idx +1], cmap='binary')

        plt.subplot(2,8, 5+idx*8)
        plt.imshow(img[idx+2], cmap='gray')
        plt.title(label[idx+2])

        plt.subplot(2,8, 6+idx*8)
        plt.imshow(mask[idx+2], cmap='binary')

        plt.subplot(2,8, 7 +idx*8)
        plt.imshow(img[idx +3], cmap='gray')
        plt.title(label[idx+3])

        plt.subplot(2,8, 8 +idx*8)
        plt.imshow(mask[idx +3], cmap='binary')

    plt.savefig('../visualize/train_dataloader.png')

    plt.show()

# VE BIEU DO PHAN PHOI CAC CLASS TRONG DATASET

# https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/barchart.html
# https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
# https://datagy.io/python-transpose-list-of-lists/
# https://www.geeksforgeeks.org/bar-plot-in-matplotlib/#:~:text=The%20matplotlib%20API%20in%20Python%20provides%20the%20bar,with%20a%20rectangle%20depending%20on%20the%20given%20parameters.

def visualize_distribute_dataset(lst_train, lst_val, lst_test):
    y = [lst_train, lst_val, lst_test]
    # set width of bar
    barWidth = 0.25


    data = np.array(y).T.tolist()

    X = np.arange(3)
    fig = plt.figure()
    ax = fig.add_axes([0,0, 1,2])

    covid = ax.bar(X + 0.00, data[0], color = 'r', width = 0.25, label = 'COVID')
    non_covid = ax.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = 'Non-COVID')
    normal = ax.bar(X + 0.50, data[2], color = 'b', width = 0.25, label = 'Normal')

    # Adding Xticks
    plt.xlabel('Dataset', fontweight ='bold', fontsize = 15)
    plt.ylabel('Total images', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(X))],
            ['Train', 'Val', 'Test'])

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(covid)
    autolabel(non_covid)
    autolabel(normal)
    plt.title('Distribution of data sources')
    plt.legend()

    # saving the file.Make sure you 
    # use savefig() before show()
    plt.savefig('../visualize/DistributeDataset.png', bbox_inches='tight')
    plt.show()
##VE BIỂU ĐỒ ACCURACY VÀ LOSS CỦA TRAIN VÀ VALIDATION
def plot_acc_loss (loss, val_loss, acc, val_acc, path):
    """ plot training and validation loss and accuracy """
    plt.figure (figsize = (12, 4))
    plt.subplot (1, 2, 1)
    plt.plot (range (len (loss)), loss, 'r-', label = 'Training')
    plt.plot (range (len (loss)), val_loss, 'go-', label = 'Validation')
    plt.xlabel ('Epochs')
    plt.ylabel ('Loss')
    plt.title ('Loss')
    plt.legend ()

    plt.subplot (1, 2, 2)
    plt.plot (range (len (acc)), acc, 'b-', label = 'Training')
    plt.plot (range (len (acc)), val_acc, 'ro-', label = 'Validation')
    plt.xlabel ('Epochs')
    plt.ylabel ('accuracy')
    plt.title ('Accuracy')
    plt.legend ()
    plt.savefig(path, bbox_inches='tight')
    plt.show ()

def plot_loss(train_loss, val_loss):
    fig, ax = plt.subplots(figsize = (18, 14.5))
    ax.plot(train_loss, '-gx', label='Training loss')
    ax.plot(val_loss , '-ro', label='Validation loss')
    ax.set(title="Loss over epochs of Model ",
    xlabel='Epoch',
    ylabel='Loss')
    ax.legend()
    fig.show()
    plt.show ()