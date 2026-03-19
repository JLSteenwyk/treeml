## Building the documentation

```bash
cd docs
pipenv install --dev
pipenv run serve          # Development server with auto-rebuild
pipenv run make html      # Manual build
```

Alternatively, using pip directly:

```bash
cd docs
pip install -r requirements.txt
make html                 # Build HTML
make watch                # Development server with auto-rebuild
```
