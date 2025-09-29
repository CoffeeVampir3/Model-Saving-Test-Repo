(Fork and clone)
```
uv run python test-save.py
git tag xyz
uv run python test-load.py
git tag xyz2
uv run python test-load.py
uv run python test-save.py
```
