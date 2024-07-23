# EndoscopyCorruptions

The `endoscopycorruptions` Python package provides utilities to simulate common image corruptions that might occur during endoscopic procedures. This tool is designed to assist in the development and testing of image processing algorithms intended for endoscopic imagery by introducing realistic corruptions into clean images. By evaluating algorithms against corrupted images, developers can better understand the robustness and limitations of their solutions.

![Alt text](https://raw.githubusercontent.com/Ivanrs297/endoscopycorruptions/main/assets/results.png "a title")



## Features

- **Corrupt Function**: Applies a specified corruption to an input image.
- **Get Corruption Names**: Lists all available corruptions that can be applied.

## Getting Started

To use the `endoscopycorruptions` package, start by importing the necessary functions:

```python
from endoscopycorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
```

### Loading an Image

Load your test image using PIL and convert it to a numpy array:

```python
image = np.asarray(Image.open('test_image.png'))
plt.imshow(image)
```

### Preprocessing

If your image includes an alpha channel, you can remove it to ensure compatibility:

```python
if len(image.shape) > 2 and image.shape[2] == 4:
    image = image[:, :, :3]
```

### Listing Available Corruptions

To see what types of corruptions you can apply, use:

```python
get_corruption_names()
```

### Applying Corruptions

You can apply a corruption to your image as follows:

```python
# Example for applying lens distortion with a severity of 5
corrupted_image = corrupt(image, corruption_name='lens_distortion', severity=5)
plt.imshow(corrupted_image)
plt.show()
```

To apply all available corruptions with varying severities and save the results:

```python
for corruption in get_corruption_names():
    for severity in range(5):
        corrupted = corrupt(image, corruption_name=corruption, severity=severity+1)
        plt.imshow(corrupted)
        plt.axis('off')
        
        folder_path = f"data/{corruption}/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plt.savefig(f'{folder_path}/c_{corruption}_sev{severity + 1}.png', bbox_inches='tight')
    print(corruption)
```

## Requirements

This package requires:

- Python 3.x
- PIL (Pillow)
- NumPy
- Matplotlib
- An environment that can run Jupyter Notebooks if you wish to use the provided notebook for demonstrations.


The `endoscopycorruptions` package is an essential tool for researchers and developers working on image processing applications for endoscopy. By facilitating the simulation of realistic image corruptions, it allows for thorough testing and improvement of image analysis algorithms.

Credits to [imagecorruptions](https://github.com/bethgelab/imagecorruptions) for the basis of this project.

Cite this project:

```
@misc{Ivanrs297_endoscopycorruptions,
  author = {Ivan Reyes-Amezcua and Ricardo Espinosa and Andres Mendez-Vazquez and Gilberto Ochoa-Ruiz and Christian Daul},
  title = {EndoscopyCorruptions: A Python package to simulate common image corruptions in endoscopic procedures},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Ivanrs297/endoscopycorruptions}},
}
```

 
