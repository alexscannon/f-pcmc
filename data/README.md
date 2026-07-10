# Precomputed Embeddings — Access Contract

F-PCMC never copies or symlinks embedding data into this repo. `fpcmc/data.py`
resolves file paths directly from `roots.env` (gitignored, one per machine —
copy `roots.env.example` to `roots.env` and set `DATA_ROOT`; see that file for
the full variable contract). This mirrors the convention already used
throughout the source project (`msproject_misc/roots.env`), rather than
inventing a second mechanism — see the "Data access mechanism" decision in
`docs/ASSETS.md` §1 for the rationale (portable across machines/CI, no ~260MB
duplication of data the PRD already treats as frozen and never regenerated).

**There is no `data/embeddings/` directory in this repo.** `EMBEDDINGS_DIR`
(resolved from `roots.env`) points at an external location; the four `.pt`
files below are read from there, wherever that is on the running machine.

## Deviation from the literal PRD §9 layout

PRD §9 lists `data/embeddings/` as a gitignored directory and TASKS.md T1
frames `test_real_pool_schemas` as "skipped with a clear message if
`data/embeddings/` absent." Under this mechanism there is no local directory
to check for absence — **the equivalent skip condition is: `roots.env`
missing, `EMBEDDINGS_DIR` unset, or the resolved path's `.pt` files not
found.** Implement the skip check against the resolved path, not a literal
`Path("data/embeddings").exists()` call.

## The five pools, four files

| Pool (PRD §3 name) | File | Count | D | Selector |
|---|---|---|---|---|
| IND Reference | `real_cifar100.pt` | 50,000 | 1024 | `sources == "cifar100_train"` |
| IND Test | `real_cifar100.pt` | 10,000 | 1024 | `sources == "cifar100_test"` |
| Synthetic IND | `ind.pt` | 250 | 1024 | (whole file) |
| Near-OOD | `novel_subclasses.pt` | 500 | 1024 | (whole file) |
| Far-OOD | `novel_superclasses.pt` | 2,576 | 1024 | (whole file) |

All four files live directly under `EMBEDDINGS_DIR` (no subdirectories). For
the A6 ResNet-50 ablation, point `EMBEDDINGS_DIR` at the `ResNet50_32px`
sibling directory instead (same four filenames, D=2048).

## `.pt` file schema (identical structure across all four files)

```python
{
  "embeddings":       torch.FloatTensor (N, D),   # NOT L2-normalized on disk — normalize on load
  "subclass_names":   list[str] (len N),
  "superclass_names": list[str] (len N),
  "sources":          list[str] (len N),
  "image_paths":      list[str] (len N),
  "label_mappings": {
    "subclass_to_id": dict[str,int], "id_to_subclass": dict[int,str],
    "superclass_to_id": dict[str,int], "id_to_superclass": dict[int,str],
  },
}
```

Each file's `label_mappings` is scoped to only the classes present in that
file — join across files on class-name strings, not on `label_mappings` ids.
`real_cifar100.pt`'s map is the canonical full 100-subclass/20-superclass
CIFAR-100 taxonomy.

Full provenance (source commit/blob hashes, verification results, known data
gaps) lives in `../docs/ASSETS.md` §1, §5, §6 — this file is the terse
loader-facing contract; that one is the audit trail.
