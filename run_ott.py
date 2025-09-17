import jax
import jax.numpy as jnp
import json , pickle

import optax

import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from ott import datasets
from ott.geometry import pointcloud
from ott.neural.methods import neuraldual
from ott.neural.networks import potentials
from ott.tools import sinkhorn_divergence

import optax, flax
from flax import linen as nn

from ott import datasets
from ott.neural.methods import neuraldual
from ott.neural.networks import icnn

from utils import *
import h5py
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

import jax 
#jax.config.update("jax_debug_nans", True)
#jax.config.update("jax_disable_jit", False)   # keep JIT, but still crash on NaN
#jax.config.update("jax_enable_x64", True)     # try fp64 for stability
gpus = [d for d in jax.devices() if d.platform == 'gpu']
print("GPUs:", gpus)

import sys, os
from datetime import datetime


dim = int(sys.argv[1])
key = f"latn_{dim}"
outdir = f'''results/{key}_{datetime.now().strftime("%Y:%m:%d:%H:%M:%S")}/'''
os.mkdir(outdir)

def estimate_stats(src_iter, tgt_iter, n_batches=100):
    xs, xt = [], []
    for _ in range(n_batches):
        xs.append(next(src_iter))
        xt.append(next(tgt_iter))
    X = jnp.concatenate(xs + xt, axis=0)
    mean = jnp.mean(X, axis=0)
    std  = jnp.std(X, axis=0) + 1e-6
    return mean, std


train_dataloader = JAXDualH5DataLoader(
    'data/predictions/QGPythiaTrain/sample_0.h5',
    'data/predictions/QGHerwigTrain/sample_0.h5',
    source_key=key, 
    target_key=key,
    batch_size=512, 
    shuffle=True, 
    seed=42,
    #prefetch_size=2,
)

valid_dataloader = JAXDualH5DataLoader(
    'data/predictions/QGPythiaVal/sample_0.h5',
    'data/predictions/QGHerwigVal/sample_0.h5',
    source_key=key, 
    target_key=key,
    batch_size=512, 
    shuffle=False, 
    seed=42,
    #prefetch_size=2,
)


gauss_init_dataloader = JAXDualH5DataLoader(
    'data/predictions/QGPythiaTrain/sample_0.h5',
    'data/predictions/QGHerwigTrain/sample_0.h5',
    source_key=key, 
    target_key=key,
    batch_size=100000, 
    shuffle=True, 
    seed=42,
    #prefetch_size=2,
)
print("Running on latent space with dim =",dim)

mean, std = estimate_stats(
    gauss_init_dataloader.source_iter, 
    gauss_init_dataloader.target_iter,
    n_batches=500
)

gm_t = as_2d(normalize(next(gauss_init_dataloader.target_iter),mean,std))
gm_s = as_2d(normalize(next(gauss_init_dataloader.source_iter),mean,std))
print(gm_s.shape, gm_t.shape)
for name, arr in [("gm_t", gm_t), ("gm_s", gm_s)]:
    assert arr.ndim == 2 and arr.shape[1] == dim, f"{name} must be (N,{dim}), got {arr.shape}"
    
#optimizer_f = optax.adam(learning_rate=1e-3, b1=0.5, b2=0.9, eps=1e-8)
#optimizer_g = optax.adam(learning_rate=1e-3, b1=0.5, b2=0.9, eps=1e-8)

# --------- 1) Optimizers with clipping + AdamW ----------
'''
tx = optax.chain(
    optax.clip_by_global_norm(1.0),                    # crucial
    optax.adamw(learning_rate=1e-5, weight_decay=1e-4)
)
optimizer_f = tx
optimizer_g = tx
'''
tx = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(learning_rate=1e-5)  # no weight_decay
)
optimizer_f = tx
optimizer_g = tx

# --------- 2) Safer ICNNs ----------
# Use SiLU (smooth), softplus on PSD diag with input clipping, and a sane init.
'''
def clipped_softplus(x):
    return nn.softplus(jnp.clip(x, -10.0, 10.0))
'''

for _ in range(10):
    s = next(train_dataloader.source_iter)
    t = next(train_dataloader.target_iter)
    for name, x in [("source", s), ("target", t)]:
        assert np.isfinite(x).all(), f"{name} contains NaN/Inf"
        assert x.ndim == 2 and x.shape[1] == dim
        
def clipped_softplus(x, beta=1.0, limit=20.0):
    # avoids exp overflow; behaves like softplus but bounded input
    x = jnp.clip(x, -limit, limit)
    return jnp.log1p(jnp.exp(beta * x)) / beta
    
neural_f = icnn.ICNN(
    #dim_hidden=[128, 128, 128],        # smaller stack is often more stable
    dim_hidden=[64,64],
    dim_data=dim, 
    init_fn=nn.initializers.lecun_normal(),
    #pos_weights=True, 
    rectifier_fn=flax.linen.activation.softplus,
    #rectifier_fn=clipped_softplus,
    pos_weights=True,
    act_fn=flax.linen.activation.celu,
    gaussian_map_samples=[gm_t, gm_s] if dim!= 1 else None,

)
neural_g = icnn.ICNN(
    #dim_hidden=[128, 128, 128],        # smaller stack is often more stable
    dim_hidden=[64,64],
    dim_data=dim, 
    init_fn=nn.initializers.lecun_normal(),
    #pos_weights=True, 
    rectifier_fn=flax.linen.activation.softplus,
    #rectifier_fn=clipped_softplus,
    pos_weights=True,
    act_fn=flax.linen.activation.celu,
    gaussian_map_samples=[gm_t, gm_s] if dim!= 1 else None,

)

#set up main training objective 
neural_dual_solver = neuraldual.W2NeuralDual(
    dim, 
    neural_f, 
    neural_g, 
    optimizer_f, 
    optimizer_g, 
    num_train_iters=0, 
    #amortization_loss="objective",
    #conjugate_solver=None,
    num_inner_iters=4,
    logging=True,
    log_freq=10


)

def estimate_stats(src_iter, tgt_iter, n_batches=100):
    xs, xt = [], []
    for _ in range(n_batches):
        xs.append(next(src_iter))
        xt.append(next(tgt_iter))
    X = jnp.concatenate(xs + xt, axis=0)
    mean = jnp.mean(X, axis=0)
    std  = jnp.std(X, axis=0) + 1e-6
    return mean, std

#mean, std = estimate_stats(
#    train_dataloader.source_iter, 
#    train_dataloader.target_iter,
#    n_batches=50
#)


#normalize or not?
train_src_iter = normalized_iter(train_dataloader.source_iter, mean, std)
train_tgt_iter = normalized_iter(train_dataloader.target_iter, mean, std)
valid_src_iter = normalized_iter(valid_dataloader.source_iter, mean, std)
valid_tgt_iter = normalized_iter(valid_dataloader.target_iter, mean, std)

### TESTS BELOW 


# Grab one normalized batch (same normalization as training!)
s0 = next(normalized_iter(train_dataloader.source_iter, mean, std))
t0 = next(normalized_iter(train_dataloader.target_iter, mean, std))

# Force 2D shape
s0 = jnp.asarray(s0, jnp.float32)
t0 = jnp.asarray(t0, jnp.float32)

# Forward values should be finite
f_vals = neural_f.apply({"params": neural_f.init(jax.random.PRNGKey(0), s0)["params"]}, s0)
g_vals = neural_g.apply({"params": neural_g.init(jax.random.PRNGKey(1), t0)["params"]}, t0)
print("pretrain f finite:", jnp.isfinite(f_vals).all(), "g finite:", jnp.isfinite(g_vals).all())

# Build params once
params_f = neural_f.init(jax.random.PRNGKey(0), s0)["params"]
params_g = neural_g.init(jax.random.PRNGKey(1), t0)["params"]

# POTENTIALS API varies; adapt if your ICNN exposes .transport or .grad
def f_on_x(x):
    return neural_f.apply({"params": params_f}, x).sum()

def g_on_y(y):
    return neural_g.apply({"params": params_g}, y).sum()

gf = jax.grad(f_on_x)(s0)  # ∇f
gg = jax.grad(g_on_y)(t0)  # ∇g
print("pretrain ∇f finite:", jnp.isfinite(gf).all(), "pretrain ∇g finite:", jnp.isfinite(gg).all())
print("||∇f||:", float(jnp.linalg.norm(gf)), "||∇g||:", float(jnp.linalg.norm(gg)))

### TESTS ABOVE 

#run training
learned_potentials, logs = neural_dual_solver(
    train_src_iter,
    train_tgt_iter,
    valid_src_iter,
    valid_tgt_iter,
)

#make full validation dataset
samples_s = []
samples_t = []
it_s = iter(valid_dataloader.source_iter)
it_t = iter(valid_dataloader.target_iter)

for _ in range(10000):  # take 10 batches
    batch_s = next(it_s)
    batch_t = next(it_t)
    samples_s.append(np.asarray(batch_s).reshape(-1, batch_s.shape[-1]))
    samples_t.append(np.asarray(batch_t).reshape(-1, batch_t.shape[-1]))

data_source = np.concatenate(samples_s, axis=0)
data_target = np.concatenate(samples_t, axis=0)


'''
#data_source = jnp.asarray(data_source, dtype=jnp.float32)
#data_target = jnp.asarray(data_target, dtype=jnp.float32)
x_src = jnp.asarray(data_source, dtype=jnp.float32)

# 1) Per-sample finiteness check for the transport
def check_one(xi):
    yi = learned_potentials.transport(xi[None, :], forward=True)
    return jnp.isfinite(yi).all()

good_mask = jax.vmap(check_one)(x_src)          # shape (N,)
bad_idx = np.where(~np.array(good_mask))[0]
print(f"[debug] bad transports: {bad_idx.size} / {x_src.shape[0]}")
if bad_idx.size:
    print("[debug] first 10 bad indices:", bad_idx[:10])

# 2) If there are bad rows, inspect one example
if bad_idx.size:
    i = int(bad_idx[0])
    xi = x_src[i:i+1]
    yi = learned_potentials.transport(xi, forward=True)
    print("[debug] example xi finite?", np.isfinite(np.array(xi)).all())
    print("[debug] example yi finite?", np.isfinite(np.array(yi)).all())
    
# 2) Transport one sample to localize failure
x0 = jnp.asarray(data_source[:1], dtype=jnp.float32)
t0 = learned_potentials.transport(x0, forward=True)
print("single transport NaN?", jnp.isnan(t0).any())

jax.config.update("jax_debug_nans", True)  # crash at the first op that makes a NaN

x_src = jnp.asarray(data_source, jnp.float32)

# Try to access a scalar potential u_f; OTT variants differ in API naming.
def _first_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            fn = getattr(obj, n)
            if callable(fn):
                return fn
    return None

u_f = _first_attr(learned_potentials, [
    "potential_f", "f_potential", "f", "u_f", "apply_f", "apply_potential_f"
])
grad_u_f = None
if u_f is not None:
    # grad wrt inputs
    def u_sum(z):  # sum to make grad scalar-safe
        return u_f(z).sum()
    grad_u_f = jax.grad(u_sum)

def check_piecewise(xi):
    xi = xi[None, :]
    # 1) potential value
    u_ok = True
    if u_f is not None:
        u = u_f(xi)
        u_ok = jnp.isfinite(u).all()
    # 2) gradient of potential
    g_ok = True
    gu = None
    if grad_u_f is not None:
        gu = grad_u_f(xi)
        g_ok = jnp.isfinite(gu).all()
    # 3) transport via API
    try:
        yi = learned_potentials.transport(xi, forward=True)
        t_ok = jnp.isfinite(yi).all()
    except Exception:
        t_ok = False
    return u_ok, g_ok, t_ok, (u if u_f is not None else None), gu

# Probe one sample
u_ok, g_ok, t_ok, u_val, gu_val = check_piecewise(x_src[0])
print("[probe] u finite:", bool(u_ok), "| grad_u finite:", bool(g_ok), "| transport finite:", bool(t_ok))
if u_val is not None:
    print("[probe] u mean/std:", float(jnp.nanmean(u_val)), float(jnp.nanstd(u_val)))
if gu_val is not None:
    print("[probe] grad_u norm:", float(jnp.sqrt((gu_val**2).sum())))
'''

# --- pick CPU or GPU batching ---
USE_CPU_FOR_TRANSPORT = True

if USE_CPU_FOR_TRANSPORT:
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax, jax.numpy as jnp
import numpy as np

device = jax.devices("cpu")[0] if USE_CPU_FOR_TRANSPORT else jax.devices()[0]

'''
def preprocess(x):
    x = jnp.asarray(x, jnp.float32)
    z = (x - mean) / std
    return jnp.clip(z, -10.0, 10.0)
    
def transport_batched(pots, X, bs=1024, forward=True, device_index=0):
    X = preprocess(X)
    if X.ndim > 2:                     # e.g. (N,16,32) → (N, 512)
        X = X.reshape(X.shape[0], -1)

    outs = []
    dev = jax.devices()[device_index]
    with jax.default_device(dev):      # all arrays created inside go to this GPU
        X = jnp.asarray(X)             # no explicit device_put
        for i in range(0, X.shape[0], bs):
            xi = X[i:i+bs]             # already on device, no H2D per step
            yi = pots.transport(xi, forward=forward)
            outs.append(yi)
        Y = jnp.concatenate(outs, axis=0)
        Y = jax.block_until_ready(Y)   # sync before leaving the context
    return np.array(Y)
def transport_batched(pots, X, bs=1024, forward=True, device=device):
    X = preprocess(X)
    outs = []
    for i in range(0, X.shape[0], bs):
        xi = jax.device_put(X[i:i+bs], device)
        yi = pots.transport(xi, forward=forward)
        outs.append(yi.block_until_ready())
    return np.array(jnp.concatenate(outs, axis=0))
'''
#z_src = (data_source - mean) / (std  + 1e-6)           # if (A)

def make_preprocessor(mean, std, clip=10.0):
    mean = jnp.asarray(mean); std = jnp.asarray(std)
    def preprocess(x):
        x = jnp.asarray(x, jnp.float32)
        z = (x - mean) / (std + 1e-6)
        return jnp.clip(z, -clip, clip)
    return preprocess

def make_denormalizer(mean, std):
    mean = jnp.asarray(mean); std = jnp.asarray(std)
    def denorm(z):
        return z * (std + 1e-6) + mean
    return denorm

def transport_batched(pots, X, bs=1024, forward=True, device_index=0, return_original_units=True):
    Xn = preprocess(X)                        # 1) normalize like training
    if Xn.ndim > 2:
        Xn = Xn.reshape(Xn.shape[0], -1)

    outs = []
    dev = jax.devices()[device_index]
    with jax.default_device(dev):
        Xn = jnp.asarray(Xn)
        for i in range(0, Xn.shape[0], bs):
            xi = Xn[i:i+bs]
            yi = pots.transport(xi, forward=forward)  # 2) map in normalized space
            outs.append(yi)
        Yn = jnp.concatenate(outs, axis=0)
        Yn = jax.block_until_ready(Yn)

    Yn = np.array(Yn)
    if return_original_units:
        return np.array(denorm(Yn))           # 3) back to original units
    else:
        return Yn                              # keep normalized if you prefer
        
preprocess = make_preprocessor(mean, std, clip=10.0)
denorm     = make_denormalizer(mean, std)

#normalize? 
transported_source = transport_batched(learned_potentials, data_source, bs=100000)
#transported_source = denorm(transported_source)
#transported_source = (np.array(transported_source) * std) + mean


#transported_source = learned_potentials.transport(data_source)
print("data_source",data_source[:10])
print("data_target",data_target[:10])
print("transported_source",transported_source[:10])

#transported_target = np.array(learned_potentials.transport(data_target,forward=False))

print(logs)
print(type(logs))

# --- Save as JSON (human readable, portable) ---
with open(f"{outdir}/logs.json", "w") as f:
    json.dump(logs, f, indent=2)

# --- Save as Pickle (Python only, preserves types) ---
with open(f"{outdir}/logs.pkl", "wb") as f:
    pickle.dump(logs, f)


np.savez(
    f"{outdir}/array.npz",
    source=data_source,
    target=data_target,
    transported_source=transported_source,
    #transported_target=transported_target,
)

#make plots
#fit a quantile map, plot transported source->target (on valid data), source, target
rng = np.random.default_rng(0)
src = np.asarray(data_source).ravel()
tgt = np.asarray(data_target).ravel()
perm_s = rng.permutation(len(src))
perm_t = rng.permutation(len(tgt))
S_tr, S_te = src[perm_s[: len(src)//2]], src[perm_s[len(src)//2:]]
T_tr, T_te = tgt[perm_t[: len(tgt)//2]], tgt[perm_t[len(tgt)//2:]]

# fit map on training halves
T_S2T = fit_quantile_map_S2T(S_tr, T_tr)

# apply to held-out source
Z = T_S2T(S_te)  # mapped source (should look like T_te)

plt.hist(Z,    bins=np.linspace(-5,5,20), histtype='step', color='red',   label='mapped source $T(S)$', density=True)
plt.hist(data_source,histtype='step',color='purple',linewidth=2,alpha=0.5, label='source',bins=np.linspace(-5,5,20), density=True)
plt.hist(data_target,histtype='step',color='orange',linewidth=2,alpha=0.5, label='target',bins=np.linspace(-5,5,20), density=True)
plt.hist(learned_potentials.transport(data_source),histtype='step',color='blue',linewidth=2,alpha=0.5, label='transported',bins=np.linspace(-5,5,20), density=True)

plt.xlabel('q vs g output feature')
plt.savefig(f"{outdir}/output_feature.pdf")
plt.savefig(f"{outdir}/output_feature.png")

f_grad, g_grad = get_maps(learned_potentials)

# compute both directions on your validation splits
s2t = f_grad(data_source)   # source → target (what you want to compare to target)
#t2s = g_grad(data_target)   # target → source (should match source)

# Plot: which one aligns?
bins = np.linspace(-5, 5, 20)
plt.hist(s2t,        histtype='step', linewidth=2, alpha=0.7, label='source→target', bins=bins)
plt.hist(data_target,histtype='step', linewidth=2, alpha=0.7, label='target',          bins=bins)
plt.legend(); 
plt.savefig(f"{outdir}/forward.pdf")
plt.savefig(f"{outdir}/forward.png")


plt.figure()
#plt.hist(t2s,        histtype='step', linewidth=2, alpha=0.7, label='target→source', bins=bins)
plt.hist(data_source,histtype='step', linewidth=2, alpha=0.7, label='source',        bins=bins)
plt.legend(); 
plt.savefig(f"{outdir}/backward.pdf")
plt.savefig(f"{outdir}/backward.png")

# pick the right map: f_grad should be source→target
f_grad, g_grad = get_maps(learned_potentials)  # as in previous snippet

# build a 1D grid over the support of your data
x_grid = np.linspace(-5, 5, 200).reshape(-1, 1)   # (200, 1)

# compute transported points
T_x = f_grad(x_grid)   # shape (200, 1)

Tstar = empirical_quantile_map(x_grid, data_source, data_target)
# plot T(x) vs x
plt.figure()
plt.plot(x_grid, T_x, label="Transport map T(x)")
plt.plot(x_grid, x_grid, 'k--', alpha=0.5, label="identity")
plt.plot(x_grid, Tstar, label="Empirical quantile map")
plt.xlabel("x")
plt.ylabel("T(x)")
plt.legend()
plt.savefig(f"{outdir}/quantile_map.pdf")
plt.savefig(f"{outdir}/quantile_map.png")
