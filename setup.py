import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Can be used to load install_require
# with open('requirements.txt', 'r') as freq:
#     required = freq.read().splitlines()


setuptools.setup(
    name="endoscopycorruptions",
    version="1.0.0",
    author="Ivan Reyes",
    author_email="ivan.reyes@cinvestav.mx",
    description="This package provides a set of endoscopy image corruptions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ivanrs297/endoscopycorruptions",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy >= 1.16',
          'Pillow >= 5.4.1',
          'scikit-image >= 0.15',
          'opencv-python >= 3.4.5',
          'scipy >= 1.2.1',
          'numba >= 0.53.0'
      ],
      include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
