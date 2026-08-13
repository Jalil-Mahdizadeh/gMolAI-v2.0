# Frozen speed-benchmark protocol

## Status and scope

The machine-readable protocol in `protocol.json` was frozen on 13 August 2026
before definitive Slurm job 1230738. This benchmark is inference only. It
cannot train or resume a model, alter a checkpoint or calibrator, regenerate a
scientific benchmark embedding artifact, or affect the promoted gMolAI
seed-42/10k representation.

The input is an exact local copy of
`extra-benchmark/test-partition/inputs/common_test.tsv`: 49,844 ordered,
collision-checked molecular identities accepted by every encoder. Its TSV
SHA-256 is
`fac4a8abffd0431b36245c6b6eaa447ce1f1628373cad590e59e7a7e2a0fc18e`;
the ordered-identity SHA-256 is
`a9e3e63eade2c542fef670184ed1a34e054167d085ad299ee080217d7301e237`.
Preflight verifies both copies and all frozen source, model and SIF hashes
before loading any encoder.

## Common conditions

- All seven models run sequentially within one Slurm job on Arrhenius node n54.
- One NVIDIA GH200 120GB GPU is visible to each neural encoder.
- Morgan runs on CPU in the same allocation and is never labelled as
  GPU-forward throughput.
- Every container receives the same 48-CPU ceiling. `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are all fixed to one.
- The fixed condition order is batch size 64, 128, 256 and 512.
- Every condition has one untimed warm-up batch followed by one complete timed
  pass over the ordered panel.
- Each model remains loaded across its four conditions; no tokenized or
  pre-featurized panel is cached between conditions.
- Native scientific inference paths return complete ordered FP32 host
  representations.

“Same settings” means the same panel, job, host, GPU visibility, CPU ceiling,
thread limits, condition order, warm-up count, measured-pass count and timing
boundary. Model-specific parsing and inference remain part of each frozen
native implementation. The optimized gMolAI path uses its selected 48-worker
RDKit pool; competitors have the same CPU allowance but retain their native
preprocessors rather than being modified to imitate gMolAI.

## Optimized gMolAI path

gMolAI uses the repository-canonical `OptimizedSmilesEncoder`
(`optimized_gine_v1`) backed by `fast_graph.py` and `fast_inference.py`.
It performs multiprocess RDKit graph construction, packed-array transfer and
the qualified equivalent GINE/readout implementation with 48 workers. The
worker pool is created and warmed outside the primary timer, then reused.

The requested batch size is the maximum graphs per internal batch. gMolAI also
retains its canonical 16,384-node safety budget, so a graph batch can end
earlier when that node limit is reached. No panel graph is precomputed or
cached. The checkpoint, calibrator, configuration and training plan are
hash-pinned in `protocol.json`; none is changed by this benchmark.

## Timing boundary

The sustained end-to-end timer starts with canonical SMILES already resident
in RAM and before allocation of the final host output. It ends only after every
ordered FP32 vector is resident in host RAM. It includes output allocation,
model-specific parsing/normalization, tokenization or 2D graph construction,
host-to-device transfer, frozen forward inference, device-to-host transfer and
host-matrix materialization.

For gMolAI, the harness times one complete-panel encoder call per condition so
its persistent preprocessing pool can feed successive GPU batches as designed.
Wrapping each mini-batch in a separate call would disable the optimized
parallel pipeline. The other adapters are called at each requested native
batch. In all cases, all per-molecule preprocessing occurs inside the timer;
only worker creation/warm-up is excluded.

CUDA is synchronized immediately before and after each complete panel pass.
The timer excludes SIF startup, model/checkpoint/calibrator loading, input-file
reading, worker-process startup, warm-up, validation, checksums and disk
serialization. Model-load, worker-startup and warm-up times are recorded
separately. Scientific embedding matrices are never written to disk.

## Integrity gates

Every condition must return exactly 49,844 finite, nonzero vectors of its
frozen native dimension and preserve row order. Outputs at batch sizes 128,
256 and 512 are checked against the batch-64 output using frozen scale-aware
gates:

- minimum per-row cosine similarity: 0.9999;
- maximum per-row relative-L2 delta: 0.005.

Exact equality, absolute and RMS deltas, relative-L2 quantiles, cosine
quantiles and matrix SHA-256 values are also recorded outside the timed region.
All encoders except the declared native KERMT limitation fail closed when a gate
is violated.

Before timing, two calls on the same ordered eight-molecule fixture must meet
the stricter fixed-batch repeatability gates: minimum cosine similarity
0.999999 and maximum relative-L2 delta 0.001. Bitwise identity is recorded but
not required because valid GPU kernels can introduce much smaller
floating-point differences.

## KERMT v2 limitation

KERMT v2 fails the unchanged cross-batch gates because its native graph builder
uses batch-local adjacency-padding width and the frozen encoder does not keep
the nominal padding state neutral. No KERMT code or checkpoint is patched,
because doing so would change the representation used in the earlier
scientific benchmarks. Its four timings are retained as native computational
throughput for batch-dependent outputs and are excluded from claims of
representation-equivalent scaling. `KERMT_BATCH_DEPENDENCE.md` contains the
controlled root-cause audit and definitive job values.

## Superseded execution

This execution replaces Slurm job 1226415 and its batch-size 8/32/64 results,
which used the pre-optimization gMolAI path. Those files are not part of the
current result set. The earlier diagnostic amendments that established the
scale-aware integrity gates and KERMT treatment remain preserved in
`protocol.json` as historical audit metadata; they did not modify a model,
panel or scientific representation.

## Reporting limits

Throughput is reported as molecules per second and milliseconds per molecule.
Native batch-latency quantiles are reported where instrumentation does not
perturb the implementation; gMolAI batch latencies are intentionally null
because per-batch instrumentation would interfere with its parallel pipeline.

There is one measured pass per condition, so there are no confidence intervals.
The allowed claim is descriptive end-to-end throughput on this fixed panel,
host, GPU and resource envelope. Small differences must not be presented as
established speed superiority, Morgan must not be described as GPU throughput,
and KERMT must not be described as representation-equivalent batch scaling.
