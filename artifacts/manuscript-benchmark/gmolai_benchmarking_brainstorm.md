# gMolAI comparator and encoding-speed benchmark: feasibility memo

**Date:** 2026-08-12
**Status:** historical planning document; benchmark program completed
**Scope:** 1D/2D molecular inputs only; Arrhenius HPC; current gMolAI evidence structure

## Execution status (updated 2026-08-20)

This memo preserves the original design reasoning and therefore uses future
tense below. The proposed comparator program was subsequently completed and
extended. The authoritative executed results are the
[locked-test representation study](../../extra-benchmark/test-partition/RESULTS.md),
[MoleculeNet plus HIV study](../../extra-benchmark/moleculenet/RESULTS.md),
[controlled encoding-speed study](../../extra-benchmark/speed/RESULTS.md),
[TDC ADMET transfer study](../../extra-benchmark/tdc-admet/RESULTS.md), and
[external molecular-clustering study](../../extra-benchmark/clustering/RESULTS.md).
Those frozen result records supersede this planning memo for realized coverage,
metrics, and interpretation.

## Executive verdict

The proposed benchmarking is feasible and would materially strengthen the manuscript, but it should be presented as three linked analyses rather than one undifferentiated leaderboard:

1. a **frozen-representation comparison on the locked internal pretraining test partition**, using only label-free geometry/topology diagnostics that are meaningful for every encoder;
2. a **frozen-feature comparison on the current MoleculeNet development/promotion panel**, using exactly the existing canonical molecules, split identities, fold-local preprocessing, linear models, hyperparameter grids and metrics; and
3. a **separate encoding-speed and resource benchmark** on Arrhenius, with both end-to-end feature-production throughput and model-only timing.

The most defensible first-wave panel is:

- gMolAI seed-42/10k, with its released 384-D representation and frozen train-only calibrator;
- RDKit Morgan radius 2, 2,048 bits;
- the previously published MolAI epoch-6 encoder (512-D);
- IBM MoLFormer-XL-both-10pct;
- IBM SMI-TED 289M; and
- the official pretrained MolCLR-GIN 2D graph encoder.

NVIDIA KERMT v2 is a particularly attractive contemporary 2D graph-transformer addition, but it should enter the headline panel only after an ARM64/`cuik-molmaker` smoke test. ChemBERTa-77M-MLM is a low-effort optional continuity control because MolAI was previously compared with ChemBERTa. Chemprop v2 is useful only as a **separate task-specific supervised track**: it is not a pretrained frozen molecular encoder, so mixing its fine-tuned scores into the main representation table would be misleading.

This design gives reviewers a strong non-neural baseline, the direct MolAI lineage comparator, two established SMILES foundation-model families, and at least one independent 2D graph-pretraining family. It also preserves the manuscript's current evidence chronology: the five MoleculeNet datasets remain selection-conditioned development evidence, while the locked internal partition remains post-selection evidence.

## 1. What can validly be compared in each partition?

### 1.1 Locked internal pretraining test partition

The population is the **1,088,766-graph locked internal pretraining test partition**. It has no experimental endpoint labels. It therefore cannot support BACE/BBBP/ESOL/FreeSolv/Lipophilicity performance comparisons.

The common, scientifically valid comparison is representation quality on the exact existing identity panels:

- deterministic encoding coverage and rejection reasons;
- non-finite output checks, zero-norm checks and deterministic repeatability;
- effective rank, effective-rank fraction, participation ratio and coordinate dispersion;
- the existing 13-target held-out topology Ridge probe, using the same 10,000 training-partition identities and 50,000 locked-test identities;
- cosine-neighbour agreement with Morgan/Tanimoto, chemical-neighbour enrichment and scaffold-neighbour enrichment;
- scaffold clustering on the exact recurring-scaffold subset, using row-L2-normalized inputs followed by standard Euclidean K-means, exactly as implemented now; and
- encoding throughput, peak accelerator memory, coverage and output size.

The exact selected-representation geometry artifact already binds a 50,000-molecule stratified locked-test sample to seed-42/10k and the released calibrator, with 5,000 similarity queries and 16,382 molecules in the recurring-scaffold clustering subset. Those identities should be reused. The existing up-to-250,000-molecule locked-test evaluation contains reconstruction and masked-graph health metrics specific to gMolAI; those architecture-specific quantities must not be placed in a common encoder leaderboard.

Important interpretation limits:

- Morgan-neighbour agreement measures how closely a latent geometry follows Morgan chemistry; it is not an absolute definition of embedding quality.
- Raw effective rank is dimension-dependent, so both rank and rank/dimension should be shown.
- Decoder reconstruction may be reported separately for encoder-decoder models, but it is not a fair common metric for encoder-only and fingerprint methods.
- The locked internal partition must not be used to select a candidate model, layer, pooling rule, KERMT output head, precision or batch-size policy. All such choices must be fixed on the pretraining validation partition before the one-shot comparative audit.

Repository anchors: the common diagnostics are implemented in [`probes.py`](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/src/gmolai_retrain/probes.py), including Morgan-neighbour analysis and the precise non-spherical K-means wording. The protected-test policy is stated in the [repository README](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/README.md). The authoritative selected 50k artifact is `runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050/representation-probes-test-standardized-raw-hybrid-w3-step10000-calibration100k-seed20260810-50k.json`.

### 1.2 Current MoleculeNet development/promotion panel

The current default panel is BACE, BBBP, ESOL, FreeSolv and Lipophilicity. The post-filter molecule counts that must remain unchanged are:

| Dataset | Task | Accepted molecules |
|---|---:|---:|
| BACE | Classification | 1,513 |
| BBBP | Classification | 1,860 |
| ESOL | Regression | 1,116 |
| FreeSolv | Regression | 639 |
| Lipophilicity | Regression | 4,198 |

For every frozen encoder, export one vector for every accepted canonical identity and run the **same** downstream evaluator:

- the exact current canonicalization, filtering, deduplication and deterministic molecule order;
- the same 10 accepted outer splits, where `GroupShuffleSplit(test_size=0.20)` assigns approximately 80/20% of **scaffold groups**, not necessarily molecules;
- the same three inner folds (`StratifiedGroupKFold` for classification and `GroupKFold` for regression);
- the existing fold-local feature scaling;
- the same Ridge and logistic-regression models and hyperparameter grids;
- the same ROC-AUC, average precision and balanced accuracy for classification, and RMSE, MAE and R² for regression; and
- paired per-split reporting, because all encoders see identical outer and inner identities.

The split identity manifests already exist in [`descriptor_only_control.json`](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/artifacts/manuscript-rev4/descriptor_only_control.json), including SHA-256 digests for every outer train/test identity set and every inner fit/validation identity set. New adapters should fail closed if any digest differs. The evaluator and current Morgan definition are in [`downstream.py`](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/src/gmolai_retrain/downstream.py).

Do not copy scores from model papers into the gMolAI table. Published scores generally use different canonicalization, dataset versions, splits, fine-tuning regimes and metrics. Every headline number must be recomputed locally under the frozen gMolAI protocol.

The existing 13-descriptor-only control must remain beside gMolAI and Morgan, especially for ESOL, FreeSolv and Lipophilicity. It answers a different and important question: how much regression performance is already explained by inexpensive physicochemical descriptors.

HIV may be added as a separately labelled **external post-selection confirmatory endpoint**, using a frozen protocol. It must not be merged into the five-dataset development/promotion panel and must never be called part of the locked internal test.

## 2. Recommended headline encoders

| Comparator | Exact frozen representation to test | Why it earns a place | Locked internal | MoleculeNet frozen probe | Arrhenius assessment |
|---|---|---|---:|---:|---|
| **RDKit Morgan** | Radius 2, 2,048-bit fingerprint already used by gMolAI | Mandatory strong, interpretable, fast non-neural baseline | Yes | Yes | **Ready now**; already implemented and validated |
| **MolAI** | Published epoch-6 encoder output, 512-D | Direct scientific lineage and architecture comparison; user-authored prior encoder | Yes | Yes | **Feasible with adapter**; legacy TensorFlow 2.10 on ARM64 is the main risk |
| **MoLFormer** | `ibm-research/MoLFormer-XL-both-10pct`, official `pooler_output` | Established self-supervised SMILES transformer; official frozen-feature use | Yes | Yes | **High feasibility** through pinned Hugging Face/PyTorch code |
| **SMI-TED** | Base SMI-TED 289M, official `model.encode` output | Recent peer-reviewed SMILES encoder-decoder with a 91M-molecule corpus and explicit feature extraction | Yes | Yes | **High-to-medium feasibility**; memory is ample, but pin/validate fast-transformer dependencies |
| **MolCLR-GIN** | Official pretrained GIN backbone representation before the task head | Established peer-reviewed, genuinely independent 2D graph contrastive encoder | Yes | Yes | **Medium feasibility**; small inference port from PyG 1.6/PyTorch 1.7 to current PyG is likely needed |
| **KERMT v2** | `nvidia/NV-KERMT-70M-v2`; predeclare either the 512-D cMIM latent or 800-D graph readout | Contemporary 70.6M-parameter 2D graph transformer, official feature-extractor use and Hopper support | Yes | Yes | **Conditional medium feasibility**; PyTorch 2.x is suitable, but ARM64 and `cuik-molmaker` must pass a pilot |

Suggested publication choice: run the first five comparators as the minimum panel. Add KERMT if its pilot passes without changing its weights or chemistry. If implementation time becomes tight, do not drop Morgan, MolAI, MoLFormer or the 2D graph comparator; SMI-TED is the most valuable fifth neural comparison because it is recent and architecturally close enough to make the encoder-decoder discussion informative.

## 3. Other possible tools and how to treat them

| Tool | 1D/2D status | Feasibility | Recommendation |
|---|---|---|---|
| **ChemBERTa-77M-MLM** | SMILES only | High; standard Hugging Face/PyTorch | Good optional continuity/sensitivity control. Use MLM, not the MTR variant, as the main ChemBERTa choice: MTR was pretrained against many RDKit properties and would complicate the descriptor-channel interpretation. Pooling/layer choice must be frozen on validation. |
| **GROVER-base** | 2D atom/bond graph | Medium; official fingerprint generation exists, but the repository targets Python 3.6 | Established graph-transformer fallback if MolCLR or KERMT cannot be made reproducible. Prefer base over large for a tractable speed comparison. |
| **MolMIM** | SMILES only; fixed latent | Medium; NVIDIA provides an embedding endpoint, but the model stack/container and 126-token input limit add operational and coverage issues | Optional encoder-decoder sensitivity analysis, not needed if SMI-TED and MolAI are both included. Never truncate overlength molecules silently. |
| **Chemprop v2 D-MPNN** | 2D graph | High for current PyTorch; fingerprints can be exported | Useful only in a separate task-specific, end-to-end MoleculeNet track. It trains on endpoint labels and therefore is not commensurable with frozen encoders plus Ridge/logistic probes. It has no meaningful locked-test comparator before task training. |
| **Graphormer** | 2D graph | Low-to-medium; Fairseq/submodule stack is heavy, and advanced pretrained versions are not generally exposed through the open repository | Reserve, not first wave. The engineering cost is high relative to the extra scientific information. |
| **KPGT/LiGhT** | 2D graph plus knowledge features | Low-to-medium; legacy DGL stack and archived artifacts | Reserve. Its knowledge-guided inputs include descriptor/fingerprint information, which confounds the manuscript's auxiliary-descriptor interpretation. |
| **SELFIES-TED** | SELFIES string | Medium; open weights but larger and adds SMILES-to-SELFIES preprocessing | Defer. It is computationally heavier and largely redundant after SMI-TED unless a reviewer specifically requests string-representation diversity. |
| **CDDD** | SMILES encoder-decoder | Technically possible but legacy TensorFlow and redundant with MolAI | Do not prioritize. MolAI is the more relevant lineage comparison. |

### Explicit 3D exclusions

Under a strict no-3D policy, exclude Uni-Mol, SchNet, DimeNet/DimeNet++, PaiNN, SphereNet and other coordinate/conformer encoders. Also exclude GraphMVP from the main panel: its deployed branch can emit a 2D graph embedding, but its representation-learning claim explicitly relies on paired 2D/3D pretraining. This avoids an avoidable reviewer dispute over what “2D-only” means. No conformer generation or 3D coordinate calculation should occur anywhere in the benchmark pipeline.

## 4. Fair frozen-feature protocol

### 4.1 Freeze the degrees of freedom before locked-test access

For every model, record and hash:

- source repository commit and license;
- checkpoint/model revision and file SHA-256;
- tokenizer/vocabulary or graph-featurizer revision;
- input canonical SMILES identity hash;
- maximum accepted length and rejection policy;
- exact hidden layer and pooling/readout;
- output dimensionality and dtype;
- numerical precision;
- deterministic inference settings; and
- validation-chosen batch and length-bucketing policy.

Use the pretraining validation partition for adapter development, pooling/readout decisions, batch-size tuning and determinism tests. Once frozen, run each adapter once on the locked identities. A model that cannot encode a molecule must emit a reason; do not silently truncate a SMILES, remove stereochemistry, substitute a zero vector or change the common identity set.

### 4.2 Keep raw native representation dimensions

Do not force every encoder to 384 dimensions with PCA or a learned projection. That would add another fitted transformation and another selection surface. Use each published native vector, report its dimension and storage cost, and rely on the same fold-local StandardScaler and regularized linear predictor. For dimension-sensitive geometry diagnostics, report normalized quantities as well as raw values.

### 4.3 Report coverage two ways

The main table should use the common set encoded successfully by every comparator so paired statistics are honest. A second table must report each model's full-dataset coverage and every rejection category. This prevents a restrictive tokenizer from appearing artificially strong by omitting difficult molecules.

### 4.4 Do not overstate independence

The exact gMolAI pretraining membership and actual checkpoint exposure are known for each downstream molecule. Comparable row-level corpora are generally unavailable for third-party checkpoints trained on PubChem, ZINC or ChEMBL. Therefore:

- retain gMolAI's exact corpus/training/seen-at-10k exposure disclosure, including that seed-42/10k had consumed exactly 57,504,265 unique training graphs;
- document each comparator's published pretraining sources;
- compute exact overlap only where the authors release a row-level pretraining manifest; and
- otherwise state that molecule-level pretraining overlap is unknown.

Do not call the MoleculeNet results structurally unseen or independent final-test results. New comparator results would be a retrospective benchmark on the existing selection-conditioned development panel. The locked internal comparison is post-selection for gMolAI, but it is also being opened retrospectively to new comparator choices; the manuscript should say this plainly.

## 5. Encoding-speed benchmark for scalability

### 5.1 Primary hardware and scaling modes

Use one Arrhenius GH200 GPU as the primary neural-encoder result. The currently observed node `n137` is ARM64 and has 4 × NVIDIA GH200 120GB GPUs, 485,854 MB node memory, driver 580.159.04 and compute capability 9.0. The project image is already pinned to an NVIDIA ARM64 PyG container with CUDA 13.0.1, PyTorch 2.8 and PyG 2.7 in [`arrhenius.def`](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/arrhenius.def).

Report:

1. single-GPU neural throughput and latency;
2. Morgan CPU throughput with one fixed core and one declared multi-core allocation;
3. optional four-GPU **sharded inference** throughput, speedup and parallel efficiency; and
4. peak GPU memory, host memory where measurable, output dimension and bytes per molecule.

Four-GPU encoding should shard molecule identities across independent workers. There is no scientific reason to use gradient/DDP synchronization for frozen inference.

### 5.2 Two timing boundaries

Measure both:

**A. Representation-ready pipeline (headline scalability number).** Start from the same accepted canonical SMILES already resident in memory and end with a predictor-ready in-memory feature matrix. Include model-specific tokenization or graph construction, collation, host-to-device transfer, forward pass, device-to-host transfer and conversion to the declared output dtype. Exclude shared one-time corpus canonicalization and model download. Report model load/start-up separately.

**B. Model-only forward pass (diagnostic).** Time already-collated device inputs through the frozen network. This identifies whether tokenization/graph featurization or the neural network is the bottleneck.

For storage/export sensitivity, separately time serialization to a fixed format. Do not let a filesystem bottleneck masquerade as encoder compute.

### 5.3 Timing controls

- Use the exact same ordered molecules for all models.
- Tune batch size and deterministic length bucketing only on validation; freeze both before locked-test timing.
- Include batch-1 latency and maximum sustained-throughput mode.
- Use warm-up batches, at least five timed repetitions, CUDA synchronization at timing boundaries, and report median plus interquartile range.
- Pin precision. FP32 is the clean common primary result; an officially supported BF16 sensitivity result may be added, but must not replace FP32 silently.
- Record software/container digests, GPU model, driver, CPU thread count, batch size, sequence-length distribution and all failure counts.
- Prevent implicit model recompilation or first-use cache creation from contaminating steady-state numbers; report cold start separately.
- For Morgan, report both the current predictor-ready float32 2,048-vector path and the native packed-bit path if storage scalability is discussed.

On the full 1,088,766-molecule partition, a float32 embedding requires `1,088,766 × dimension × 4` bytes before container overhead. For example, the released 384-D gMolAI matrix alone is about 1.67 GB. This makes dimension and serialization format part of the scalability story, not just molecules/second.

The MolAI paper reports more than 4,000 molecules/s on an A100 and approximately 550 and 860 molecules/s for CDDD and ChemBERTa, respectively. Those values are useful prior context but cannot serve as the new comparison: they were produced on different code paths and possibly different batch/input boundaries. All models must be rerun under the same GH200 protocol.

## 6. MolAI-specific integration plan

MolAI is scientifically essential here. The published model is a SMILES-to-SMILES sequence autoencoder with three stacked 1,024-unit LSTMs and a 512-D `tanh` latent vector; the public repository targets Python 3.8 and TensorFlow 2.10. Its paper reports a 21M-parameter encoder and provides the published epoch-6 artifact through split HDF5 parts.

When the checkpoint and encoding script are supplied:

1. preserve their original bytes and record SHA-256 hashes;
2. freeze the exact published tokenizer, canonicalization, vocabulary, maximum length and encoder layer;
3. run a small reference fixture in an isolated TensorFlow environment;
4. first attempt direct inference in an Apptainer-compatible environment;
5. if TensorFlow 2.10 GPU support on Grace/ARM64 is impractical, make an inference-only PyTorch weight port without changing any tensor, activation or token rule;
6. compare the port with the reference TensorFlow outputs on a chemically diverse fixture, with declared absolute/relative tolerances and cosine agreement; and
7. accept the port only if it is numerically equivalent and deterministic.

No MolAI retraining or downstream fine-tuning is needed for the main benchmark. The direct public checkpoint is preferable to recreating the model. The user-supplied script is valuable because it resolves preprocessing and latent-extraction ambiguity that the HDF5 file alone cannot.

## 7. Arrhenius implementation feasibility

The GH200 memory budget is not the limiting factor for any recommended single model. The principal risk is software portability on the ARM64 host:

- **Low risk:** Morgan/RDKit, MoLFormer through pinned Hugging Face code, ChemBERTa.
- **Low-to-medium risk:** SMI-TED; PyTorch is suitable, but its historical `pytorch-fast-transformers` dependency must be pinned or replaced only after numerical equivalence testing.
- **Medium risk:** MolCLR and GROVER; their published repositories target old PyTorch/PyG or Python, but inference-only architectures are small enough to adapt to the already installed PyG stack.
- **Medium risk:** KERMT; its official checkpoint uses PyTorch 2.x and lists Hopper support, but accelerated molecular graph construction may assume an x86 build.
- **Highest integration risk but still feasible:** MolAI, because the publication stack specifies TensorFlow 2.10 on an ARM64/CUDA-13 system. An isolated reference runtime plus validated inference port is the robust path if no compatible container is available.

Each ecosystem should use a separate immutable Apptainer image or environment; do not mutate the released gMolAI image to accommodate conflicting legacy stacks. Cache model weights under project storage and pin hashes so jobs do not depend on live web state.

## 8. Recommended execution order

1. **Freeze a protocol document first.** Name exact model revisions, output layers, preprocessing, common identity hashes, metrics and speed boundaries.
2. **Build validation-only smoke tests.** Use approximately 100 diverse molecules, including stereochemistry, charges, long strings and uncommon but supported elements. Require deterministic finite vectors and explicit rejection reasons.
3. **Pilot throughput on validation.** Freeze precision, batching and length bucketing. Do not inspect the locked test for tuning.
4. **Export the five MoleculeNet feature matrices.** Verify the existing identity/split hashes and run the unchanged linear-probe evaluator.
5. **Run one-shot locked-test comparison.** Reuse the exact 50k/5k/16,382 identity panels for common representation diagnostics.
6. **Run the scalability benchmark.** Use a large deterministic locked-test sequence, ideally the full 1,088,766 identities for sustained throughput, after all policies are frozen.
7. **Prepare two manuscript tables.** One table for frozen-feature MoleculeNet performance (with descriptor-only and Morgan), and one for coverage/speed/resources. Put locked-test geometry/topology in a separate table or figure.
8. **Optionally run supervised Chemprop.** Only with separate authorization and a separate heading that states neural weights were trained on each endpoint.

## 9. Inputs needed before implementation

- The exact published MolAI checkpoint and encoding script, preferably with the environment/requirements file used for the paper.
- Confirmation that the intended primary MolAI artifact is the epoch-6 512-D encoder from the publication.
- A choice on whether to include KERMT v2 despite its very recent release and not-yet-peer-reviewed v2 model description.
- Authorization for model-weight downloads and separate Apptainer images when execution begins.
- Separate authorization if the optional supervised Chemprop track is desired, because that track trains endpoint-specific models.

No such input is needed to preserve the current gMolAI checkpoint, calibrator or results; this memo proposes only additive comparison work.

## 10. Primary sources consulted

All web sources below were accessed on 2026-08-12. Exact repository commits and model revisions should be pinned at implementation time.

- [RDKit Morgan fingerprint documentation](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints)
- [MolAI peer-reviewed article (JCIM)](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00491) and [official MolAI repository](https://github.com/AnyoLabs/MolAI-Publication)
- [MoLFormer official repository](https://github.com/IBM/molformer), [official frozen-feature model card](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct) and [Nature Machine Intelligence paper](https://www.nature.com/articles/s42256-022-00580-7)
- [SMI-TED peer-reviewed article](https://www.nature.com/articles/s42004-025-01585-0), [official IBM repository](https://github.com/IBM/materials) and [official model card](https://huggingface.co/ibm-research/materials.smi-ted)
- [MolCLR official implementation and paper metadata](https://github.com/yuyangw/MolCLR)
- [KERMT v2 official NGC model card](https://catalog.ngc.nvidia.com/orgs/nvidia/clara/models/kermt-contrastive/2.0) and [official repository](https://github.com/NVIDIA-BioNeMo/KERMT)
- [ChemBERTa-2 paper](https://arxiv.org/abs/2209.01712) and [official DeepChem model](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)
- [GROVER official implementation](https://github.com/tencent-ailab/grover)
- [Chemprop v2 official repository](https://github.com/chemprop/chemprop) and [fingerprint-export documentation](https://chemprop.readthedocs.io/en/latest/mpnn_fingerprints.html)
- [MolMIM official model documentation](https://docs.nvidia.com/bionemo-framework/1.10.1/models/molmim.html) and [embedding endpoint](https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html)
- [Graphormer official repository](https://github.com/microsoft/Graphormer)
- [Uni-Mol official repository](https://github.com/deepmodeling/Uni-Mol) and [GraphMVP official repository](https://github.com/chao1224/GraphMVP), used to verify the 3D exclusions
- [NVIDIA framework-container support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/)

## Bottom line

This is publishable-quality benchmarking if it is framed as a frozen-representation study with identical identities and probes—not as a collage of literature scores. The strongest balance of scientific breadth and practical feasibility is **gMolAI + Morgan + MolAI + MoLFormer + SMI-TED + MolCLR**, with **KERMT v2** added after an ARM64 pilot. The locked internal partition should answer representation-health and geometry questions; the five MoleculeNet datasets should answer endpoint-transfer questions; and speed should be a separately controlled systems benchmark.
