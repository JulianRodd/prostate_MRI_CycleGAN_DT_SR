import SimpleITK as sitk


class Padding(object):
    """
    Add padding to the image if size is smaller than patch size with improved handling
    for prostate MRI data.
    """

    def __init__(self, output_size):
        self.name = "Padding"

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert all(i > 0 for i in list(self.output_size))

    def calculate_padding_extent(self, current_size, desired_size):
        """Calculate padding needed on each side to center the image."""
        padding_needed = [max(0, d - c) for d, c in zip(desired_size, current_size)]
        padding_before = [p // 2 for p in padding_needed]
        padding_after = [p - b for p, b in zip(padding_needed, padding_before)]
        return padding_before, padding_after

    def pad_image(self, image, output_size):
        """Pad image with improved boundary handling."""
        size_old = image.GetSize()

        if all(s_old >= s_new for s_old, s_new in zip(size_old, output_size)):
            return image

        padding_before, padding_after = self.calculate_padding_extent(
            size_old, output_size
        )

        padFilter = sitk.MirrorPadImageFilter()
        padFilter.SetPadLowerBound(padding_before)
        padFilter.SetPadUpperBound(padding_after)

        return padFilter.Execute(image)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        size_old = image.GetSize()

        if all(s_old >= s_new for s_old, s_new in zip(size_old, self.output_size)):
            return sample

        output_size = list(self.output_size)
        for i in range(3):
            if size_old[i] > self.output_size[i]:
                output_size[i] = size_old[i]
        output_size = tuple(output_size)

        try:
            padded_image = self.pad_image(image, output_size)

            padded_label = self.pad_image(label, output_size)

            assert (
                padded_image.GetSize() == padded_label.GetSize()
            ), "Padding resulted in mismatched sizes"

            return {"image": padded_image, "label": padded_label}

        except Exception as e:
            print(f"Padding error: {str(e)}")
            return sample
