# Benchmarks

The active benchmarks are organized by controller:

- [MPC](MPC/README.md): one test per plant, manually assigned discrete complexity.
- [Adaptive](Adaptive/README.md): the existing Adaptive setup, unchanged.

The complete previous continuous-complexity workflow, its test experiments,
artifacts, reports, and plots are preserved in
`../benchmarks_backup_2026-09-16/`. Nothing was deleted from that backup.

## MPC workflow

1. Edit the single test in each `MPC/artifacts/*.json`.
2. Fill `total_complexity` with your own integer from **1 (easy) to 10 (hard)**.
3. Run `MPC/mpc_benchmark.py` to tune and verify one test per selected plant.
4. Run `MPC/finalize_results.py` to copy the manual ratings and calculate success.
5. Run `MPC/plot_dashboard.py` to compare LLM success at each complexity level, mean
   tokens by LLM, and mean cost by LLM.

There is no MPC test-combination maker, complexity scorer, or separate
plant/condition complexity in this version. Changing a manual rating only
requires finalizing and plotting again, not rerunning the tuner.
