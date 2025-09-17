import h5py, numpy as np
import jax, jax.numpy as jnp
from flax.jax_utils import prefetch_to_device


def as_2d(x):
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = x[:, None]                # (N,) -> (N,1)
        print(x.shape)
    return x


def fit_quantile_map_S2T(src_train, tgt_train):
    s_sorted = np.sort(src_train.ravel())
    t_sorted = np.sort(tgt_train.ravel())
    nS, nT = len(s_sorted), len(t_sorted)

    def T(x):
        x = np.asarray(x).ravel()
        # F_S(x): empirical CDF by right rank / nS
        ranks = np.searchsorted(s_sorted, x, side="right") / nS
        # map ranks to target quantiles
        idx = np.clip((ranks * (nT - 1)).astype(int), 0, nT - 1)
        return t_sorted[idx]
    return T

def empirical_quantile_map(x_grid, src, tgt):
    # src,tgt: (N,1) arrays; x_grid: (M,1)
    import numpy as np
    src = np.asarray(src).ravel()
    tgt = np.asarray(tgt).ravel()
    xg  = np.asarray(x_grid).ravel()
    s_q = np.argsort(src)
    F_s = np.searchsorted(np.sort(src), xg, side="right") / len(src)
    qs  = np.clip((F_s * (len(tgt)-1)).astype(int), 0, len(tgt)-1)
    Tstar = np.sort(tgt)[qs]
    return Tstar  # shape (M,)

def get_maps(pots):
    # Try to grab the vmapped gradients in a version-agnostic way
    if hasattr(pots, "f_grad") and hasattr(pots, "g_grad"):
        f_grad, g_grad = pots.f_grad, pots.g_grad
    elif hasattr(pots, "transport"):  # some versions let you pick a direction
        # wrap to explicit maps
        f_grad = lambda x: pots.transport(x, forward=True)   # source→target
        g_grad = None #lambda x: pots.transport(x, forward=False)  # target→source
    elif hasattr(pots, "transport_source") and hasattr(pots, "transport_target"):
        f_grad, g_grad = pots.transport_source, pots.transport_target
    else:
        raise RuntimeError("Can't find gradient/transport methods on learned_potentials.")
    return f_grad, g_grad
    
def merge_shards(it):
    it = iter(it)
    while True:
        x = jax.device_get(next(it))         # (ndev, per_dev, ...)
        x = np.asarray(x).reshape(-1, *x.shape[2:])  # (ndev*per_dev, ...)
        yield x


def normalize(x, mean, std):
    z = (x - mean) / (std + 1e-6)
    return jnp.clip(z, -10.0, 10.0).astype(jnp.float32)

def normalized_iter(raw_iter, mean, std):
    while True:
        yield normalize(next(raw_iter), mean, std)


def normalize_for_ott(it, dim=1, name=""):
    ndev = jax.local_device_count()
    step = 0
    for batch in it:
        arr = np.asarray(batch)

        # Make 3D: (n_dev, per_dev, dim)
        if arr.ndim == 1:                   # (B,) → (1,B,1)
            arr = arr.reshape(1, -1, 1)
        elif arr.ndim == 2:                 # (B,d) → (1,B,d)
            arr = arr[None, ...]
        elif arr.ndim > 3:                  # collapse extra tails
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)

        # Move device axis to front if needed
        if arr.shape[0] != ndev and ndev in arr.shape:
            dev_ax = int(list(arr.shape).index(ndev))
            arr = np.moveaxis(arr, dev_ax, 0)
        if arr.shape[0] != ndev:            # assume single device
            arr = arr[None, ...]

        # Ensure feature last; if some axis equals dim, move it to -1
        if dim in arr.shape[1:]:
            feat_ax = (1 + list(arr.shape[1:]).index(dim))
            arr = np.moveaxis(arr, feat_ax, -1)

        # Final collapse to (n_dev, per_dev, dim)
        if arr.ndim != 3:
            arr = arr.reshape(arr.shape[0], -1, arr.shape[-1])

        # Asserts
        assert arr.ndim == 3, (name, arr.shape)
        assert arr.shape[0] == ndev, (name, "ndev", arr.shape, ndev)
        assert arr.shape[-1] == dim, (name, "dim", arr.shape, dim)

        # Per-device & per-sample sanity
        a0 = arr[0]
        assert a0.ndim == 2, (name, "dev slice ndim", a0.shape)
        assert a0.shape[-1] == dim, (name, "dev slice dim", a0.shape)
        s0 = a0[0]
        assert s0.shape == (dim,), (name, "sample shape", s0.shape, dim)

        # Log a few steps so you see the first mismatch when it happens
        if step < 3 or step % 100 == 0:
            print(f"{name} step {step}: batch {arr.shape} | dev0 {a0.shape} | sample {s0.shape}")
        step += 1

        yield jnp.asarray(arr)
        
class JAXDualH5DataLoader:
    """

    Args:
        source_path (str): Path to the source .h5 file.
        target_path (str): Path to the target .h5 file.
        source_key (str):  Dataset key inside source HDF5 file.
        target_key (str):  Dataset key inside target HDF5 file.
        batch_size (int):  Number of samples per batch (per stream).
        shuffle (bool):    Shuffle each epoch for both streams.
        seed (int):        Base RNG seed; target stream uses seed+1.
        prefetch_size(int):Number of batches to prefetch onto device.
        align (bool):      If True, iter(self) yields (source, target) in lockstep.
        drop_last (bool):  If True, drop tail that doesn't fill a full batch.
        cache_in_mem (bool): If True, read the full dataset into RAM once.
                             (Fast, but can be memory heavy.)
    """
    def __init__(
        self,
        source_path: str,
        target_path: str,
        source_key: str = "latn_1",
        target_key: str = "latn_1",
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        prefetch_size: int = 0,
        align: bool = True,
        drop_last: bool = True,
        cache_in_mem: bool = True,
        device_index: int = 0,
    ):
        self.source_path = source_path
        self.target_path = target_path
        self.source_key = source_key
        self.target_key = target_key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch_size = prefetch_size
        self.align = align
        self.drop_last = drop_last
        self.cache_in_mem = cache_in_mem
        self.device = jax.devices()[device_index]
        
        self.ndev = jax.local_device_count()
        print(f"Using {self.ndev} devices for prefetching.")

        # Sanity: ensure batch_size divisible by ndev if we shard
        if self.ndev > 1 and (self.batch_size % self.ndev != 0):
            per_dev = self.batch_size // self.ndev
            new_bs = per_dev * self.ndev
            print(f"[JAXDualH5DataLoader] batch_size {self.batch_size} not divisible by "
                  f"{self.ndev} devices; truncating to {new_bs}.")
            self.batch_size = new_bs

        # Determine dataset sizes
        with h5py.File(self.source_path, "r") as fs:
            self.n_source = fs[self.source_key].shape[0]
        with h5py.File(self.target_path, "r") as ft:
            self.n_target = ft[self.target_key].shape[0]

        with h5py.File(self.source_path, "r") as fs:
            self.n_source = fs[self.source_key].shape[0]
        with h5py.File(self.target_path, "r") as ft:
            self.n_target = ft[self.target_key].shape[0]

        # Build the two iterators
        self._build_iterators()


    def _shard(self, arr):
        """Shard batch across devices if needed, and add a leading axis for pmap."""
        if self.ndev > 1:
            per_dev = arr.shape[0] // self.ndev
            arr = arr[: per_dev * self.ndev]
            arr = arr.reshape((self.ndev, per_dev) + arr.shape[1:])
        # Add a leading axis so users can opt to vmap/pmap over it if they want
        return arr[None]

    def _data_generator(self, path, key, n_samples, seed, cached, device_arr=None):
        """Yield batches; if device_arr is set, batches are sliced on GPU."""
        # If we have a full device-resident array, we never open the file.
        f = None if (cached is not None or device_arr is not None) else h5py.File(path, "r", swmr=True)
        try:
            if device_arr is not None:
                dataset = device_arr  # jax.Array on GPU
                dataset_on_device = True
            else:
                dataset = cached if cached is not None else f[key]  # np.ndarray or h5py dataset
                dataset_on_device = False
    
            rng = np.random.RandomState(seed)
            base_idx = np.arange(n_samples)
    
            # enforce per-device sharding bookkeeping (you keep this logic)
            ndev = self.ndev
            per_dev = max(1, self.batch_size // ndev)
            eff_bs = per_dev * ndev
            if eff_bs != self.batch_size:
                print(f"[JAXDualH5DataLoader] Adjusting batch_size {self.batch_size} -> {eff_bs} for {ndev} devices.")
                self.batch_size = eff_bs
    
            while True:
                if self.shuffle:
                    rng.shuffle(base_idx)
    
                # drop tail to keep equal shards
                for start in range(0, n_samples - eff_bs + 1, eff_bs):
                    batch_idx = base_idx[start : start + eff_bs]
                    # sorted slicing is faster for HDF5, harmless for device gather
                    batch_idx_sorted = np.sort(batch_idx)
    
                    if dataset_on_device:
                        # Indices on device; gather stays on device, no H2D copy
                        idx_dev = jnp.asarray(batch_idx_sorted, dtype=jnp.int32)
                        data = dataset[idx_dev]     # jax device gather
                    else:
                        # Host path: RAM cache or direct HDF5 read
                        data = dataset[batch_idx_sorted]  # np.ndarray
                    #print(data.shape)
                    yield data
        finally:
            if f is not None:
                f.close()
    '''
    def _data_generator(self, path, key, n_samples, seed, cached):
        """Yield batches as (ndev, per_dev, ...) for prefetch_to_device."""
        # Open the file only if we're not caching to RAM
        f = None if cached is not None else h5py.File(path, "r", swmr=True)
        try:
            dataset = cached if cached is not None else f[key]

            rng = np.random.RandomState(seed)
            base_idx = np.arange(n_samples)

            # enforce per-device sharding
            ndev = self.ndev
            per_dev = max(1, self.batch_size // ndev)
            eff_bs = per_dev * ndev
            if eff_bs != self.batch_size:
                print(f"[JAXDualH5DataLoader] Adjusting batch_size {self.batch_size} -> {eff_bs} for {ndev} devices.")
                self.batch_size = eff_bs

            while True:
                if self.shuffle:
                    rng.shuffle(base_idx)

                # drop tail to keep equal shards
                for start in range(0, n_samples - eff_bs + 1, eff_bs):
                    batch_idx = base_idx[start : start + eff_bs]
                    # sorted slicing is faster in HDF5
                    batch_idx_sorted = np.sort(batch_idx)

                    data = dataset[batch_idx_sorted]  # (eff_bs, ...)
                    # reshape to (ndev, per_dev, ...)
                    #data = data.reshape(ndev, per_dev, *data.shape[1:]) #adding this line even for 1 GPU makes the code not run 
                    yield data
                    
        finally:
            if f is not None:
                f.close()
    '''
    def _load_dataset(self, path, key):
        """Optional RAM cache."""
        if not self.cache_in_mem:
            return None
        with h5py.File(path, "r") as f:
            return f[key][()]  # load entire dataset once



    def _build_iterators(self):
        # prepare caches (or None)
        src_cached = self._load_dataset(self.source_path, self.source_key)
        tgt_cached = self._load_dataset(self.target_path, self.target_key)
    
        # --- NEW: make persistent device arrays if cached is available ---
        self.src_dev = None
        self.tgt_dev = None
        if src_cached is not None:
            # move once to GPU
            print("moving source to GPU")
            self.src_dev = jax.device_put(jnp.asarray(src_cached), device=self.device)
        if tgt_cached is not None:
            print("moving target to GPU")
            self.tgt_dev = jax.device_put(jnp.asarray(tgt_cached), device=self.device)
    
        # raw infinite generators
        source_gen = self._data_generator(
            self.source_path, self.source_key, self.n_source, seed=self.seed,
            cached=src_cached, device_arr=self.src_dev      # <- pass device array
        )
        target_gen = self._data_generator(
            self.target_path, self.target_key, self.n_target, seed=self.seed + 1,
            cached=tgt_cached, device_arr=self.tgt_dev      # <- pass device array
        )
    
        self.source_iter = source_gen
        self.target_iter = target_gen
        
    def __iter__(self):
        if not self.align:
            # If not aligning, iterate source by default
            return self.source_iter
        # Align: yield paired (source_batch, target_batch) in lockstep
        # Note: stops when the shorter iterator would run out this epoch.
        return zip(self.source_iter, self.target_iter)

    def reset(self):
        """Reset both iterators (start a new epoch)."""
        self._build_iterators()