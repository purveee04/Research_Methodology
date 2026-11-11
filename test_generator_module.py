"""
Unit tests for generator_module.py  (Member 1, task M1.5)

Run on Kaggle / laptop (uses 48x48 images so it is fast):
    python -m unittest test_generator_module -v
Set  RUN_FULL=1  to also test the real 192x192 size (slower, needs more memory):
    RUN_FULL=1 python -m unittest test_generator_module -v
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # must be before importing tensorflow
import unittest
import numpy as np
import tensorflow as tf
import generator_module as gm

SIZE = 48
B = 4


def fake_batch(size=SIZE, b=B, seed=0):
    rng = np.random.default_rng(seed)
    return tf.constant(rng.uniform(-1, 1, (b, size, size, 3)).astype(np.float32))


class TestNoise(unittest.TestCase):
    def test_shape_and_range(self):
        x = fake_batch()
        y = gm.add_gaussian_noise(x, 0.3)
        self.assertEqual(y.shape, x.shape)
        self.assertGreaterEqual(float(tf.reduce_min(y)), -1.0)
        self.assertLessEqual(float(tf.reduce_max(y)), 1.0)

    def test_sigma_zero_is_identity(self):
        x = fake_batch()
        np.testing.assert_allclose(gm.add_gaussian_noise(x, 0.0).numpy(), x.numpy())

    def test_noise_changes_image_and_grows_with_sigma(self):
        x = fake_batch()
        d_small = float(tf.reduce_mean(tf.abs(gm.add_gaussian_noise(x, 0.05, seed=1) - x)))
        d_large = float(tf.reduce_mean(tf.abs(gm.add_gaussian_noise(x, 0.5, seed=1) - x)))
        self.assertGreater(d_small, 0.0)
        self.assertGreater(d_large, d_small)


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = gm.build_image_generator(SIZE)

    def test_output_shape_range_finite(self):
        x = fake_batch()
        y = self.G(gm.add_gaussian_noise(x, 0.3), training=True)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(y))))
        self.assertLessEqual(float(tf.reduce_max(tf.abs(y))), 1.0)      # tanh output

    def test_gradients_flow_to_every_trainable_variable(self):
        x = fake_batch()
        noisy = gm.add_gaussian_noise(x, 0.3)
        with tf.GradientTape() as t:
            y = self.G(noisy, training=True)
            loss = gm.reconstruction_l1(x, y)
        grads = t.gradient(loss, self.G.trainable_variables)
        self.assertTrue(all(g is not None for g in grads), "some variable has no gradient")
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads))
        self.assertGreater(sum(float(tf.reduce_sum(tf.abs(g))) for g in grads), 0.0)

    def test_output_depends_on_input(self):
        a = gm.add_gaussian_noise(fake_batch(seed=1), 0.3)
        b = gm.add_gaussian_noise(fake_batch(seed=2), 0.3)
        self.assertGreater(gm.input_dependence(self.G, a, b), 0.0)

    def test_unsupported_size_raises(self):
        with self.assertRaises(ValueError):
            gm.build_image_generator(100)

    def test_compresses_input(self):
        """no skip connections: at 96x96 the narrowest layer holds fewer numbers than the image
        (at 192 it is 12x fewer; at 48 there is NO compression, so 48 is for smoke tests only)"""
        G96 = gm.build_image_generator(96)
        widths = [int(np.prod(l.output_shape[1:])) for l in G96.layers
                  if isinstance(l, tf.keras.layers.Conv2D)]
        self.assertLess(min(widths), 96 * 96 * 3)


class TestReconLoss(unittest.TestCase):
    def test_zero_for_identical(self):
        x = fake_batch()
        self.assertAlmostEqual(float(gm.reconstruction_l1(x, x)), 0.0, places=7)

    def test_known_value(self):
        a = tf.zeros((2, 4, 4, 3))
        b = tf.fill((2, 4, 4, 3), 0.5)
        self.assertAlmostEqual(float(gm.reconstruction_l1(a, b)), 0.5, places=6)

    def test_scalar(self):
        x = fake_batch()
        self.assertEqual(gm.reconstruction_l1(x, x * 0.9).shape, ())


@unittest.skipUnless(os.environ.get("RUN_FULL") == "1", "set RUN_FULL=1 for the 192x192 test")
class TestFullSize(unittest.TestCase):
    def test_192(self):
        G = gm.build_image_generator(192)
        x = fake_batch(192, 2)
        y = G(gm.add_gaussian_noise(x, 0.3), training=False)
        self.assertEqual(y.shape, (2, 192, 192, 3))
        print("192x192 generator params:", G.count_params())


if __name__ == "__main__":
    unittest.main()
