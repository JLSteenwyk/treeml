# Serialization Design

## Decision Log

| Question | Choice | Rationale |
|----------|--------|-----------|
| Format | joblib | sklearn standard, handles numpy efficiently, Bio.Phylo trees serialize fine |
| API shape | Module-level functions | Simpler than methods on 11 classes, matches joblib/pickle convention |
| Bundle contents | Model + metadata | version + class name for validation on load |
| File extension | `.treeml` | Distinctive, auto-appended |
| Version mismatch | Warn but load | UserWarning, don't block users |

## API

```python
from treeml import save_model, load_model

# Save
save_model(model, "my_model.treeml")

# Load
loaded = load_model("my_model.treeml")
```

## Implementation

- `save_model(model, path)`: validates fitted (has `inner_model_`), bundles `{"model": model, "metadata": {"treeml_version": ..., "estimator_class": ...}}`, joblib.dump, auto-appends `.treeml`
- `load_model(path)`: joblib.load, validates bundle structure, version warning, returns model
- Location: `treeml/_serialization.py`
