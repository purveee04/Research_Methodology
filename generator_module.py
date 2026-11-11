"""
generator_module.py  --  Member 1 (Generator & Gaussian noise)

Provides, for the whole team:
    add_gaussian_noise(clean, sigma)          clean images -> noisy images (clipped to [-1, 1])
    build_image_generator(img_size)           noisy image -> generated image (same shape)
    reconstruction_l1(clean, generated)       mean absolute error (used ONLY in Experiment B)
    identity_score / input_dependence         diagnostics (is the generator copying / ignoring input?)

Interface contract
    images are float32 tensors of shape (B, img_size, img_size, 3), range [-1, 1]
    img_size is 192 (paper), 96 or 48 (cheaper). The team must agree ONE value.
    generator(noisy, training=True) returns the same shape, range [-1, 1] (tanh).

The WGAN-GP critic is NOT touched here. Experiment A uses the adversarial loss only;
Experiment B adds lambda_rec * reconstruction_l1(clean, generated) -- that switch lives in
Member 2's train_step, not in this file.

NOTE: if you use TensorFlow >= 2.16, set  os.environ["TF_USE_LEGACY_KERAS"] = "1"
(and `pip install tf_keras`) BEFORE importing tensorflow, as in gc_swgan_fixed.py.
"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.initializers import RandomNormal

# (channels of the stride-2 encoder convs, hidden decoder channels) per resolution.
# The encoder compresses the image to a 6x6x256 map (12x fewer numbers than the 192x192x3
# image at 192; 3x fewer at 96; NO compression at 48, which is for smoke tests only), the SAME
# starting map as the original generator's Dense->Reshape(6,6,256).
# There are deliberately NO skip connections, so the network cannot simply copy its input.
_CFG = {
    192: dict(enc=[64, 128, 128, 256, 256], dec=[256, 128, 128, 64]),
    96:  dict(enc=[64, 128, 256, 256],      dec=[256, 128, 64]),
    48:  dict(enc=[64, 128, 256],           dec=[256, 128]),
}


def add_gaussian_noise(clean, sigma, seed=None):
    """noisy = clip(clean + N(0, sigma^2), -1, 1).  sigma is in pixel units of the [-1,1] range.
    sigma = 0 returns the clean image unchanged. No sigma value is 'instructor approved':
    choose it on VALIDATION results and record it in config.py."""
    clean = tf.convert_to_tensor(clean, tf.float32)
    if sigma == 0:
        return clean
    noise = tf.random.normal(tf.shape(clean), stddev=sigma, seed=seed)
    return tf.clip_by_value(clean + noise, -1.0, 1.0)


def build_image_generator(img_size=192, name="G_i2i"):
    if img_size not in _CFG:
        raise ValueError("img_size must be one of %s" % sorted(_CFG))
    cfg = _CFG[img_size]
    seed_counter = [0]

    def init():       # new seeded initializer per layer (shared instance gives identical weights)
        seed_counter[0] += 1
        return RandomNormal(0.0, 0.02, seed=seed_counter[0])

    def block_down(m, f):
        m.add(layers.Conv2D(f, 4, strides=2, padding="same", use_bias=False,
                            kernel_initializer=init()))
        m.add(layers.BatchNormalization())
        m.add(layers.LeakyReLU(0.2))

    def block_up(m, f, stride):
        m.add(layers.Conv2DTranspose(f, 4, strides=stride, padding="same", use_bias=False,
                                     kernel_initializer=init()))
        m.add(layers.BatchNormalization())
        m.add(layers.LeakyReLU(0.2))

    m = Sequential(name=name)
    m.add(layers.Input((img_size, img_size, 3)))
    for f in cfg["enc"]:                       # encoder: img_size -> 6x6
        block_down(m, f)
    m.add(layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_initializer=init()))
    m.add(layers.BatchNormalization())
    m.add(layers.LeakyReLU(0.2))               # bottleneck (6, 6, 256)
    block_up(m, 256, 1)                        # decoder = same stack as the original generator
    for f in cfg["dec"]:
        block_up(m, f, 2)
    m.add(layers.Conv2DTranspose(3, 4, strides=2, padding="same", use_bias=False,
                                 activation="tanh", kernel_initializer=init()))
    assert m.output_shape[1:] == (img_size, img_size, 3), m.output_shape
    return m


def reconstruction_l1(clean, generated):
    """Mean absolute error between the CLEAN image and G(noisy). Returns a scalar tensor.
    Only called when the experiment uses lambda_rec > 0 (Experiment B)."""
    return tf.reduce_mean(tf.abs(tf.cast(clean, tf.float32) - tf.cast(generated, tf.float32)))


# ------------------------------- diagnostics -------------------------------------
def identity_score(clean, noisy, generated):
    """Is the generator doing anything beyond copying its input?
    Returns (|G(noisy)-clean|, |noisy-clean|) as mean absolute errors.
    If the first is about equal to (or larger than) the second, it is not denoising; if it is
    nearly 0 with a tiny bottleneck-free model, it may be copying."""
    return (float(reconstruction_l1(clean, generated)),
            float(reconstruction_l1(clean, noisy)))


def input_dependence(generator, noisy_a, noisy_b):
    """Mean |G(a) - G(b)| for two DIFFERENT noisy batches (inference mode).
    A value near 0 means the generator ignores its input (expected danger in Experiment A,
    where nothing ties the output to the input)."""
    ga = generator(noisy_a, training=False)
    gb = generator(noisy_b, training=False)
    return float(tf.reduce_mean(tf.abs(ga - gb)))
