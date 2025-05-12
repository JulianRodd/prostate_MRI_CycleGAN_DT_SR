import random

import torch


class ImagePool:
    """
    A buffer that stores previously generated images.

    This class implements an image buffer that takes images from previous iterations
    and returns them randomly according to the buffer size. This helps to stabilize
    GAN training by providing the discriminator with a mix of current and past
    generated images.

    Args:
        pool_size (int): The maximum size of the image buffer.
    """

    def __init__(self, pool_size: int):
        self.pool_size = pool_size  # Store pool size as attribute
        self.num_imgs = 0
        self.images = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        Query the image buffer for a batch of images.

        For each image in the input batch, either add it to the buffer and return it,
        or replace a random image in the buffer with this one and return the old image.
        This mechanism helps to reduce model oscillation during GAN training.

        Args:
            images (torch.Tensor): New generated images to query the buffer

        Returns:
            torch.Tensor: A batch of images from the buffer
        """
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)
