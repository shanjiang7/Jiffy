# Post-processing

## Melt-history VTK export (`global_view.py`)

Converts a run's temperature snapshots into VTK time series for ParaView,
written to `<run dir>/VTK_global/` as two series:

- `melt_history/` — cells that have melted up to each snapshot, accumulated
  onto one global grid;
- `snapshot_domain/` — the moving-domain footprint at each snapshot.

For the quick-start run from the repository README (Section 3):

```bash
python src/hermes/post/global_view.py \
  --output-path outputs/example_straight_line/par2 \
  --sim-config configs/examples/sim_calibration.ini \
  --path-config configs/accuracy/straight_line_tol1e4.ini \
  --dt-us 10 --write-every 10
```

The sim/path configs must match the ones that produced the run's
snapshots; `--write-every` thins the output frames. In ParaView, open the
`global_melt_..vtk` file group inside each series directory to load it as
a time sequence.
