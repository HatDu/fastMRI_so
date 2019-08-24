import matplotlib.pyplot as plt
import torchvision
def show_images(images):
    '''
    images: (B, C, H, W)
    '''
    # insert channel=1
    images = images.unsqueeze(1)
    grid = torchvision.utils.make_grid(images, nrow=4, pad_value=1)
    print(grid.size())
    grid = grid.permute(1, 2, 0)
    plt.imshow(grid, cmap='gray')
    plt.show()