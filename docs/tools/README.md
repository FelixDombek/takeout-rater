# Tools

## Infer Takeout sidecar schema

`infer_takeout_sidecar_schema.py` walks a Google Photos Takeout folder, reads
every `*.supplemental-metadata.json` file, and infers an aggregate schema
(which fields are present / required / optional).

**Run the test script as a standalone summary tool:**

```bash
# Linux / macOS
python tests/test_infer_takeout_sidecar_schema.py /path/to/Takeout

# Windows (PowerShell or cmd) – use python explicitly so the .py association
# to an editor (e.g. VS Code) is bypassed
python tests\test_infer_takeout_sidecar_schema.py "H:\Takeout"
```

**Run the underlying inference script directly (outputs raw JSON):**

```bash
python docs/tools/infer_takeout_sidecar_schema.py "H:\Takeout"
python docs/tools/infer_takeout_sidecar_schema.py "H:\Takeout" --out schema.json
python docs/tools/infer_takeout_sidecar_schema.py "H:\Takeout" --show-progress
```

> **Windows tip:** On Windows, double-clicking or running `.\script.py` in
> PowerShell uses the registered file association for `.py` files, which is
> often an editor (VS Code, IDLE, …).  Always prefix with `python` to run the
> script with the interpreter instead.
