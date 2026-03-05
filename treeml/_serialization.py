import warnings

import joblib
from sklearn.exceptions import NotFittedError

from treeml.version import __version__


def save_model(model, path: str) -> str:
    """Save a fitted treeml estimator to disk.

    Args:
        model: A fitted treeml estimator (must have inner_model_ attribute).
        path: File path. If it doesn't end with '.treeml', the extension is appended.

    Returns:
        The actual path the model was saved to.

    Raises:
        NotFittedError: If the model has not been fitted yet.
    """
    path = str(path)
    if not hasattr(model, "inner_model_"):
        raise NotFittedError(
            "This model is not fitted yet. Call 'fit' before saving."
        )

    if not path.endswith(".treeml"):
        path = path + ".treeml"

    bundle = {
        "model": model,
        "metadata": {
            "treeml_version": __version__,
            "estimator_class": type(model).__name__,
        },
    }
    joblib.dump(bundle, path)
    return path


def load_model(path: str):
    """Load a treeml estimator from disk.

    Only load model files you trust. Loading a malicious file
    can execute arbitrary code via pickle deserialization.

    Args:
        path: Path to a .treeml file.

    Returns:
        The fitted treeml estimator.

    Raises:
        ValueError: If the file is not a valid treeml model bundle.
    """
    path = str(path)
    bundle = joblib.load(path)

    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError(
            f"'{path}' is not a valid treeml model file."
        )

    metadata = bundle["metadata"]
    saved_version = metadata.get("treeml_version", "unknown")
    if saved_version != __version__:
        warnings.warn(
            f"Model was saved with treeml version {saved_version} but you are "
            f"loading with treeml version {__version__}. This may cause issues.",
            UserWarning,
            stacklevel=2,
        )

    return bundle["model"]
