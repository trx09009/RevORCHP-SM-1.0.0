"""Small, stat-aware cache for repeated local prediction calls."""
from functools import lru_cache
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits


@lru_cache(maxsize=4)
def _load(path: str, modified_ns: int, size: int):
    return joblib.load(path)


def load_bundle(path):
    resolved = Path(path).resolve()
    info = resolved.stat()
    return _load(str(resolved), info.st_mtime_ns, info.st_size)


def clear_bundle_cache():
    _load.cache_clear()


def predict_model(model, inputs):
    # Tiny per-module calls benefit from one native worker; avoid OpenMP startup
    # overhead for each target of a multioutput boosted model.
    with threadpool_limits(limits=1):
        return model.predict(inputs)
