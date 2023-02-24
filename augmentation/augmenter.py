import numpy as np




def augment(img,mask):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """

    def random_fliprot(img, mask):
        # assert img.ndim >= mask.ndim
        axes = tuple(range(mask.ndim))
        perm = tuple(np.random.permutation(axes))
        img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
        mask = mask.transpose(perm)
        for ax in axes:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return (img, mask)

    def random_intensity_change(img):
        img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
        return (img)
    x, y = random_fliprot(img, mask)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return (x, y)