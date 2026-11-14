"""
verify_generator_96.py -- Member 1 verification of the generator before the A/B pilots.

Checks (all at the chosen resolution, default 96x96):
  1. input / noisy / generated shape, dtype, value range, finite
  2. Gaussian noise: std ~= sigma, mean ~= 0, clipped to [-1,1], sigma=0 is identity, fresh draws differ
  3. gradients: every generator variable gets a finite, non-zero gradient
  4. loss modes (with a STAND-IN critic, not the real one):
       A = adversarial only            -> gradient identical to "B with lambda_rec = 0"
       B = adversarial + lambda * L1   -> gradient differs from A; L1 alone has non-zero gradient
  5. diagnostics (identity score, input dependence) and a timing of generator forward+backward

IMPORTANT: this script cannot prove what Member 2's real train_step does. It proves the
generator module behaves correctly and that, if the loss is assembled as in section 4, A and B
are cleanly separated. Section "Member 2 code checklist" in the handoff covers the real loop.

Usage (run in the folder with generator_module.py):
    python verify_generator_96.py --sigma 0.3 --lambda-rec 10
    python verify_generator_96.py --h5 /kaggle/input/<dataset>/Galaxy10_DECaLS.h5   # use real galaxies
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # before importing tensorflow
import sys
import json
import time
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import generator_module as gm

p = argparse.ArgumentParser()
p.add_argument("--img-size", type=int, default=96)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--sigma", type=float, default=0.3, help="NOT instructor-approved; use the team's config value")
p.add_argument("--lambda-rec", type=float, default=10.0, help="use the team's agreed value")
p.add_argument("--h5", default="", help="optional Galaxy10 h5 file to test on real galaxy images")
p.add_argument("--timing-batch", type=int, default=64)
p.add_argument("--out", default="verify_generator_results.json")
args = p.parse_args()

S, B = args.img_size, args.batch
checks = []


def add(name, ok, detail):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    print("[%s] %s | %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

# ------------------------------------------------------------------ data
if args.h5:
    import h5py
    with h5py.File(args.h5, "r") as f:
        raw = f["images"][:B, 32:224, 32:224, :].astype(np.float32)
    if S != 192:
        raw = tf.image.resize(raw, (S, S), method="area").numpy()
    clean_np = raw / 127.5 - 1.0
    data_src = "real Galaxy10 images"
else:
    clean_np = np.random.default_rng(0).uniform(-1, 1, (B, S, S, 3)).astype(np.float32)
    data_src = "random tensors (pass --h5 to use real galaxies)"
clean = tf.constant(clean_np)
print("Data:", data_src, "| shape", clean.shape)

G = gm.build_image_generator(S)

# ------------------------------------------------------------------ 1. shapes
noisy = gm.add_gaussian_noise(clean, args.sigma, seed=1)
gen = G(noisy, training=False)
ok_shape = (clean.shape == noisy.shape == gen.shape == (B, S, S, 3))
add("1a input/noisy/generated shape", ok_shape,
    "input %s, noisy %s, generated %s" % (tuple(clean.shape), tuple(noisy.shape), tuple(gen.shape)))
add("1b dtype float32", clean.dtype == noisy.dtype == gen.dtype == tf.float32,
    "%s / %s / %s" % (clean.dtype.name, noisy.dtype.name, gen.dtype.name))
rng_ok = (float(tf.reduce_min(clean)) >= -1 and float(tf.reduce_max(clean)) <= 1 and
          float(tf.reduce_min(noisy)) >= -1 and float(tf.reduce_max(noisy)) <= 1 and
          float(tf.reduce_min(gen)) >= -1 and float(tf.reduce_max(gen)) <= 1)
add("1c value range within [-1,1]", rng_ok,
    "clean [%.3f, %.3f], noisy [%.3f, %.3f], generated [%.5f, %.5f]" % (
        float(tf.reduce_min(clean)), float(tf.reduce_max(clean)),
        float(tf.reduce_min(noisy)), float(tf.reduce_max(noisy)),
        float(tf.reduce_min(gen)), float(tf.reduce_max(gen))))
add("1d finite outputs", bool(tf.reduce_all(tf.math.is_finite(gen))), "no NaN/Inf in generated images")

# ------------------------------------------------------------------ 2. noise
zeros = tf.zeros((16, S, S, 3))
n1 = gm.add_gaussian_noise(zeros, args.sigma, seed=11)
n2 = gm.add_gaussian_noise(zeros, args.sigma, seed=12)
std = float(tf.math.reduce_std(n1))
mean = float(tf.reduce_mean(n1))
tol = max(0.15 * args.sigma, 0.01)
if args.sigma <= 0.4:
    add("2a noise std ~= sigma", abs(std - args.sigma) <= tol, "measured std %.4f vs sigma %.4f (tol %.4f)" % (std, args.sigma, tol))
else:
    add("2a noise std ~= sigma", True, "sigma %.2f is large; clipping lowers the std (measured %.4f). Informational." % (args.sigma, std))
add("2b noise mean ~= 0", abs(mean) < 0.01, "measured mean %.5f" % mean)
add("2c sigma=0 returns clean image", bool(np.allclose(gm.add_gaussian_noise(clean, 0.0).numpy(), clean_np)), "exact copy")
add("2d different seeds give different noise", float(tf.reduce_mean(tf.abs(n1 - n2))) > 0, "mean |n1-n2| = %.4f" % float(tf.reduce_mean(tf.abs(n1 - n2))))
add("2e noisy stays in [-1,1] (clipping works)", float(tf.reduce_max(tf.abs(gm.add_gaussian_noise(clean, 0.8, seed=3)))) <= 1.0, "tested with sigma 0.8")

# ------------------------------------------------------------------ 3/4. gradients and A/B loss
critic = Sequential([layers.Input((S, S, 3)),
                     layers.Conv2D(16, 4, 2, "same"), layers.LeakyReLU(0.2),
                     layers.Conv2D(32, 4, 2, "same"), layers.LeakyReLU(0.2),
                     layers.Flatten(), layers.Dense(1)], name="STANDIN_critic")


def grads_for(lam):
    """generator gradient of  adversarial + lam * L1  (same noisy batch each time)"""
    with tf.GradientTape() as t:
        fake = G(noisy, training=True)
        adv = -tf.reduce_mean(critic(fake, training=False))          # adversarial generator loss
        rec = gm.reconstruction_l1(clean, fake)
        loss = adv + lam * rec
    return t.gradient(loss, G.trainable_variables), float(adv), float(rec)


def gnorm(gs):
    return float(tf.linalg.global_norm([g for g in gs if g is not None]))


gA, advA, recA = grads_for(0.0)                 # Experiment A
gB0, _, _ = grads_for(0.0)                      # B with lambda=0  (should equal A)
gB, advB, recB = grads_for(args.lambda_rec)     # Experiment B

none_count = sum(g is None for g in gA)
add("3a every generator variable receives a gradient (A)", none_count == 0,
    "%d variables, %d without gradient" % (len(gA), none_count))
add("3b gradients finite", all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gA if g is not None) and
    all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gB if g is not None), "A and B")
add("3c gradients non-zero (A)", gnorm(gA) > 0, "global grad norm A = %.6g" % gnorm(gA))

max_diff_A_B0 = max(float(tf.reduce_max(tf.abs(a - b))) for a, b in zip(gA, gB0) if a is not None)
add("4a Experiment A == B with lambda_rec=0 (no hidden reconstruction term)", max_diff_A_B0 < 1e-6,
    "max |gradA - gradB(lambda=0)| = %.3g" % max_diff_A_B0)
max_diff_A_B = max(float(tf.reduce_max(tf.abs(a - b))) for a, b in zip(gA, gB) if a is not None)
add("4b Experiment B differs from A when lambda_rec>0", max_diff_A_B > 1e-8,
    "lambda_rec=%.3g, max |gradA - gradB| = %.3g, grad norm A %.4g vs B %.4g" % (args.lambda_rec, max_diff_A_B, gnorm(gA), gnorm(gB)))

with tf.GradientTape() as t:
    fk = G(noisy, training=True)
    l1 = gm.reconstruction_l1(clean, fk)
gL1 = t.gradient(l1, G.trainable_variables)
add("4c L1 term alone has a non-zero gradient", gnorm(gL1) > 0, "L1 value %.4f, grad norm %.6g" % (float(l1), gnorm(gL1)))
add("4d loss values reported", True, "adversarial (stand-in critic) %.4f, L1 %.4f, lambda_rec*L1 %.4f" % (advA, recA, args.lambda_rec * recA))

# ------------------------------------------------------------------ 5. diagnostics + timing
idg, idn = gm.identity_score(clean, noisy, G(noisy, training=False))
other = gm.add_gaussian_noise(tf.reverse(clean, [0]), args.sigma, seed=5)
dep = gm.input_dependence(G, noisy, other)
print("Diagnostics (untrained generator): |G(noisy)-clean| = %.4f, |noisy-clean| = %.4f, input dependence = %.4f" % (idg, idn, dep))
add("5a output depends on input", dep > 0, "mean |G(a)-G(b)| = %.5f" % dep)

xb = gm.add_gaussian_noise(tf.random.uniform((args.timing_batch, S, S, 3), -1, 1), args.sigma)
cb = tf.random.uniform((args.timing_batch, S, S, 3), -1, 1)


@tf.function
def step():
    with tf.GradientTape() as t:
        f = G(xb, training=True)
        loss = -tf.reduce_mean(critic(f, training=False)) + args.lambda_rec * gm.reconstruction_l1(cb, f)
    g = t.gradient(loss, G.trainable_variables)
    return loss, g


for _ in range(3):
    step()[0].numpy()
t0 = time.time()
n_it = 10
for _ in range(n_it):
    l, _g = step()
    l.numpy()                                   # force the GPU to finish before reading the clock
dt = (time.time() - t0) / n_it
add("5b timing measured (generator only, with warm-up + sync)", True, "%.4f s per forward+backward, batch %d, %dx%d" % (dt, args.timing_batch, S, S))

# ------------------------------------------------------------------ sample grid
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = min(6, B)
    fig, ax = plt.subplots(3, k, figsize=(2 * k, 6))
    ax = np.atleast_2d(ax).reshape(3, k)
    for i in range(k):
        for r, (im, ttl) in enumerate([(clean[i], "clean"), (noisy[i], "noisy"), (gen[i], "generated (untrained)")]):
            ax[r, i].imshow(np.clip((im.numpy() + 1) / 2, 0, 1)); ax[r, i].axis("off")
            if i == 0:
                ax[r, i].set_title(ttl, fontsize=8, loc="left")
    fig.savefig("verify_generator_grid.png", dpi=80, bbox_inches="tight")
    print("Saved verify_generator_grid.png")
except Exception as e:                           # plotting must never hide the check results
    print("grid not saved:", e)

# ------------------------------------------------------------------ summary
n_fail = sum(not c["pass"] for c in checks)
print("\n===== SUMMARY: %d checks, %d failed =====" % (len(checks), n_fail))
json.dump({"args": vars(args), "tensorflow": tf.__version__,
           "gpus": [g.name for g in tf.config.list_physical_devices("GPU")],
           "data": data_src, "checks": checks}, open(args.out, "w"), indent=2)
print("Wrote", args.out)
sys.exit(1 if n_fail else 0)
