# londongrad

*Let for them it be London, but for us it is Londongrad.*

Deep learning framework based on NumPy. Unlike London which founded on Thames it founded on Tensor.

### Pylint

Run to keep it clean.

```
find . -name "*.py" -not -path "./venv/*" | xargs -i pylint --rcfile setup.cfg {} 
```

### Pytest

Run to keep it correct.

```
python -m pytest -s tests
```