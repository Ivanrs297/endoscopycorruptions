# -*- coding: utf-8 -*-

import numpy as np
import math
from PIL import Image
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import os
from numba import njit, prange
import skimage.transform as sk_transform
import skimage.util as sk_util
from skimage.transform import warp


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)




# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble,
                                                      array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
        0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    # clipping along the width dimension:
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2

    # clipping along the height dimension:
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                  (zoom_factor, zoom_factor, 1), order=1)

    return img

def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1

def gauss_function(x, mean, sigma):
    return (np.exp(- (x - mean)**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)

def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k/Z

def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted

def _motion_blur(x, radius, sigma, angle):
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
            # simulated motion exceeded image borders
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred

# Numba nopython compilation to shuffle_pixles
@njit()
def _shuffle_pixels_njit_glass_blur(d0,d1,x,c):

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(d0 - c[1], c[1], -1):
            for w in range(d1 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x

# /////////////// End Corruption Helpers ///////////////

# mod of https://github.com/bethgelab/imagecorruptions
# /////////////// Corruptions ///////////////

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][
        severity - 1]

    x = np.uint8(
        gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    x = _shuffle_pixels_njit_glass_blur(np.array(x).shape[0],np.array(x).shape[1],x,c)

    return np.clip(gaussian(x / 255., sigma=c[0]), 0,
                   1) * 255

def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    if len(x.shape) < 3 or x.shape[2] < 3:
        channels = np.array(cv2.filter2D(x, -1, kernel))
    else:
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    shape = np.array(x).shape
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.array(x)

    angle = np.random.uniform(-45, 45)
    x = _motion_blur(x, radius=c[0], sigma=c[1], angle=angle)

    if len(x.shape) < 3 or x.shape[2] < 3:
        gray = np.clip(np.array(x).transpose((0, 1)), 0, 255)
        if len(shape) >= 3 or shape[2] >=3:
            return np.stack([gray, gray, gray], axis=2)
        else:
            return gray
    else:
        return np.clip(x, 0, 255)


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)

    set_exception = False
    for zoom_factor in c:
        if len(x.shape) < 3 or x.shape[2] < 3:
            x_channels = np.array([x, x, x]).transpose((1, 2, 0))
            zoom_layer = clipped_zoom(x_channels, zoom_factor)
            zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], 0]
        else:
            zoom_layer = clipped_zoom(x, zoom_factor)
            zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]

        try:
            out += zoom_layer
        except ValueError:
            set_exception = True
            out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer

    if set_exception:
        print('ValueError for zoom blur, Exception handling')
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def fog(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    shape = np.array(x).shape
    max_side = np.max(shape)
    map_size = next_power_of_2(int(max_side))

    x = np.array(x) / 255.
    max_val = x.max()

    x_shape = np.array(x).shape
    if len(x_shape) < 3 or x_shape[2] < 3:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
                    :shape[0], :shape[1]]
    else:
        x += c[0] * \
             plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0],
             :shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x_PIL = x
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)

        if len(x.shape) < 3 or x.shape[2] < 3:
            add_spatter_color = cv2.cvtColor(np.clip(m * color, 0, 1),
                                             cv2.COLOR_BGRA2BGR)
            add_spatter_gray = rgb2gray(add_spatter_color)

            return np.clip(x + add_spatter_gray, 0, 1) * 255

        else:

            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return cv2.cvtColor(np.clip(x + m * color, 0, 1),
                                cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        x_rgb = np.array(x_PIL.convert('RGB'))

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x_rgb[..., :1]),
                                42 / 255. * np.ones_like(x_rgb[..., :1]),
                                20 / 255. * np.ones_like(x_rgb[..., :1])),
                               axis=2)
        color *= m[..., np.newaxis]
        if len(x.shape) < 3 or x.shape[2] < 3:
            x *= (1 - m)
            return np.clip(x + rgb2gray(color), 0, 1) * 255

        else:
            x *= (1 - m[..., np.newaxis])
            return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.clip(x + c, 0, 1)
    else:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.

    gray_scale = False
    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.array([x, x, x]).transpose((1, 2, 0))
        gray_scale = True
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    if gray_scale:
        x = x[:, :, 0]

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    gray_scale = False
    if x.mode != 'RGB':
        gray_scale = True
        x = x.convert('RGB')
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)
    if gray_scale:
        x = x.convert('L')

    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x_shape = np.array(x).shape

    x = x.resize((int(x_shape[1] * c), int(x_shape[0] * c)), Image.BOX)

    x = x.resize((x_shape[1], x_shape[0]), Image.NEAREST)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    sigma = np.array(shape_size) * 0.01
    alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][
        severity - 1]
    max_dx = shape[0] * 0.005
    max_dy = shape[0] * 0.005

    dx = (gaussian(np.random.uniform(-max_dx, max_dx, size=shape[:2]),
                   sigma, mode='reflect', truncate=3) * alpha).astype(
        np.float32)
    dy = (gaussian(np.random.uniform(-max_dy, max_dy, size=shape[:2]),
                   sigma, mode='reflect', truncate=3) * alpha).astype(
        np.float32)

    if len(image.shape) < 3 or image.shape[2] < 3:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    else:
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                              np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx,
                                                          (-1, 1)), np.reshape(
            z, (-1, 1))
    return np.clip(
        map_coordinates(image, indices, order=1, mode='reflect').reshape(
            shape), 0, 1) * 255

# /////////////// End Corruptions ///////////////


# ////////////// Start Endoscopy Corruptions ///////////////
def darkness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]  # Severity levels

    x = np.array(x) / 255.  # Normalize the image to 0-1

    if len(x.shape) < 3 or x.shape[2] < 3:  # If the image is grayscale
        x = np.clip(x - c, 0, 1)  # Subtract c, ensuring values stay in the 0-1 range
    else:  # If the image is in color
        x = sk.color.rgb2hsv(x)  # Convert RGB to HSV
        x[:, :, 2] = np.clip(x[:, :, 2] - c, 0, 1)  # Subtract c from the V channel, ensuring values stay in the 0-1 range
        x = sk.color.hsv2rgb(x)  # Convert back to RGB

    return np.clip(x, 0, 1) * 255  # Convert back to 0-255 range and ensure values stay within this range

def lens_distortion(x, severity=1):
    # Check if x is a PIL Image and convert to numpy array if true
    if isinstance(x, Image.Image):
        x = np.array(x)
    
    severity_scale = [0.01, 0.02, 0.03, 0.04, 0.05][severity - 1]

    # Normalize the image if not already in float format
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float32) / 255.

    # Apply distortion
    distorted = warp(x, inverse_map=lambda coords: _distortion_map(coords, severity_scale, x.shape))

    # Assuming the input image is in 0-255 range, scale back
    return np.clip(distorted, 0, 1) * 255

def _distortion_map(coords, severity, shape):
    # Extract the original shape
    height, width = shape[:2]
    # Calculate the center of the image
    center_y, center_x = (height - 1) / 2., (width - 1) / 2.
    
    # Normalize coordinates
    y, x = np.transpose(coords, (1, 0))
    y = (y - center_y) / center_y
    x = (x - center_x) / center_x
    
    # Calculate radial distance from the center
    r = np.sqrt(x**2 + y**2)
    # Apply distortion effect based on radial distance
    distortion = 1 + r**2 * severity
    
    # Convert back to original coordinates system
    y_distorted = y * distortion * center_y + center_y
    x_distorted = x * distortion * center_x + center_x
    
    return np.stack((y_distorted, x_distorted), axis=-1)


def resolution_change(x, severity=1):
    factor = [0.5, 0.4, 0.3, 0.2, 0.1][severity - 1]  # Downscale factor
    if isinstance(x, Image.Image):
        x = np.array(x)
    original_shape = x.shape
    # Downscale
    x_downscaled = sk_transform.resize(x, (int(original_shape[0] * factor), int(original_shape[1] * factor)), anti_aliasing=True)
    # Upscale back to original size
    x_upscaled = sk_transform.resize(x_downscaled, (original_shape[0], original_shape[1]), anti_aliasing=True)
    return x_upscaled

def specular_reflection(x, severity=1):
    intensity = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    
    if isinstance(x, Image.Image):
        x = np.array(x)
    
    # Ensure x is in float format to handle the addition correctly
    original_dtype = x.dtype  # Store original dtype to convert back later
    x = x.astype(np.float32) / 255.0  # Normalize to 0-1 range for float operations
    
    # Calculate reflection parameters
    reflection_shape = (int(x.shape[0] * 0.1), int(x.shape[1] * 0.1))  # Reflection size
    reflection_position = (np.random.randint(0, x.shape[0] - reflection_shape[0]), np.random.randint(0, x.shape[1] - reflection_shape[1]))
    
    x[reflection_position[0]:reflection_position[0]+reflection_shape[0], reflection_position[1]:reflection_position[1]+reflection_shape[1]] += intensity
    
    x = np.clip(x, 0, 1) * 255.0
    x = x.astype(original_dtype)
    
    return x

def color_changes(x, severity=1):
    shift = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.  # Normalize
    for i in range(3):  # Assuming x is in the format (height, width, channels)
        x[:, :, i] = np.clip(x[:, :, i] + shift * (np.random.rand() - 0.5), 0, 1)
    return x * 255

def iso_noise(x, severity=1):
    x = np.array(x, dtype=np.float32) / 255.
    
    mean = 0
    var = [0.01, 0.02, 0.03, 0.04, 0.05][severity - 1]
    sigma = var**0.5
    
    amount = [0.01, 0.02, 0.03, 0.04, 0.05][severity - 1]
    salt_vs_pepper = 0.5  
    gauss = np.random.normal(mean, sigma, x.shape)
    noisy = x + gauss
    
    noisy = sk_util.random_noise(noisy, mode='s&p', amount=amount, salt_vs_pepper=salt_vs_pepper)
    noisy = np.clip(noisy, 0, 1)
    return noisy * 255


# ////////////// End Endoscopy Corruptions ///////////////
