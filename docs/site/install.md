# Install

## Development Install

```bash
python -m pip install -e ".[dev]"
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
```

## Runtime Import

```python
import evocore

print(evocore.__version__)
```
