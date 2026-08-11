# Graph Report - gMolAI-retrain  (2026-08-11)

## Corpus Check
- 58 files · ~62,664 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 713 nodes · 1916 edges · 28 communities (23 shown, 5 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 197 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a2934b23`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train.py
- ValueError
- cli.py
- Audit Disposition and Retraining Corrections
- atomic_write_json
- Representation V1 Training Overlay
- Full Promotion Evaluation
- model.py
- data.py
- Combined ZINC-PubChem Retraining Configuration
- Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)
- generate_embeddings.py
- RuntimeError
- run_representation_probes
- build
- audit_step
- common.sh
- Descriptor Schema Configuration
- run_benchmark_in_container.sh
- Representation Screening Configuration (5k steps, frequent validation, no resume)
- submit_pipeline.sh
- gmolai-retrain
- sample_per_graph_negatives
- Manuscript rev3 audit artifacts
- MolecularRepresentationModel
- MolecularVGAE
- _TrainingBatchPrefetcher
- checkpoint.py

## God Nodes (most connected - your core abstractions)
1. `train()` - 36 edges
2. `atomic_write_json()` - 30 edges
3. `build_parser()` - 29 edges
4. `MolecularRepresentationModel` - 29 edges
5. `benchmark_moleculenet()` - 25 edges
6. `Representation V1 Training Overlay` - 25 edges
7. `benchmark_descriptor_control()` - 23 edges
8. `_load()` - 22 edges
9. `_print()` - 21 edges
10. `export_embeddings()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `Training-Time Molecule Acceptance Policy` --semantically_similar_to--> `Molecular Canonicalization Policy`  [INFERRED] [semantically similar]
  inference/README.md → configs/retrain.yaml
- `Explicit 13-Descriptor Contract` --semantically_similar_to--> `Resolved Descriptor Contract`  [INFERRED] [semantically similar]
  AUDIT_RESPONSE.md → README.md
- `Implementation v5 Deterministic Representation` --semantically_similar_to--> `v5 Deterministic Residual GINE Encoder`  [INFERRED] [semantically similar]
  AUDIT_RESPONSE.md → README.md
- `Exact-Denominator Distributed Validation` --semantically_similar_to--> `Exact-Denominator Evaluation`  [INFERRED] [semantically similar]
  AUDIT_RESPONSE.md → REPRESENTATION_REPAIR.md
- `Fail-Closed Representation Promotion` --semantically_similar_to--> `Fail-Closed Promotion Gate`  [INFERRED] [semantically similar]
  AUDIT_RESPONSE.md → REPRESENTATION_REPAIR.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Manuscript Accuracy Correction Set** — graphify_out_memory_query_20260811_093827_are_the_two_manuscript_documents_accurate_relative_cosine_tanimoto_spearman_sampling_protocol, graphify_out_memory_query_20260811_093827_are_the_two_manuscript_documents_accurate_relative_table_1_raw_hybrid_vector_protocol, graphify_out_memory_query_20260811_093827_are_the_two_manuscript_documents_accurate_relative_seed_42_esol_rounding, graphify_out_memory_query_20260811_093827_are_the_two_manuscript_documents_accurate_relative_seed_43_calibration_protocol, graphify_out_memory_query_20260811_093827_are_the_two_manuscript_documents_accurate_relative_separate_edge_decoder_mlps, graphify_out_memory_query_20260811_093827_are_the_two_manuscript_documents_accurate_relative_per_fold_standardscaler_interpretation, graphify_out_memory_query_20260811_093827_are_the_two_manuscript_documents_accurate_relative_manuscript_wording_and_units_corrections [EXTRACTED 1.00]
- **Checkpoint Promotion Evidence Chain** — graphify_out_memory_query_20260811_105025_create_publication_quality_plots_showing_gmolai_tr_figure_2_checkpoint_promotion_composite, graphify_out_memory_query_20260811_110950_one_important_question_has_emerged__at_training_st_step_10k_operational_selection, graphify_out_memory_query_20260811_112119_are_all_retained_checkpoints_from_5k_through_15k_a_seed_42_retained_checkpoint_set, graphify_out_memory_query_20260811_121530_i_want_you_to_conduct_a_full_promotion_evaluation_promotion_gate_trajectory [INFERRED 0.85]
- **Deterministic Public Embedding Contract** — readme_v5_deterministic_graph_encoder, representation_repair_v5_deterministic_graph_encoder, representation_repair_train_only_coordinate_calibrator, review_artifacts_promoted_checkpoint_calibrator_pair [INFERRED 0.85]
- **Fail-Closed Selection Evidence** — audit_response_fail_closed_promotion, readme_fail_closed_representation_promotion, representation_repair_fail_closed_promotion, review_artifacts_retained_checkpoint_selection_audit [INFERRED 0.85]
- **Representation Validation Suite** — representation_repair_exact_denominator_evaluation, representation_repair_external_scaffold_probes, representation_repair_cross_seed_replication, representation_repair_internal_scaffold_hash_test, representation_repair_retained_checkpoint_trajectory [EXTRACTED 1.00]
- **Projector-Space Contrastive Weight and Descriptor-Weight Pilot Sweep** — configs_representation_pilot_contrastive_0005_configuration, configs_representation_pilot_contrastive_001_desc050_configuration, configs_representation_pilot_contrastive_001_seed43_configuration, configs_representation_pilot_contrastive_001_configuration, configs_representation_pilot_contrastive_002_configuration, configs_representation_pilot_contrastive_005_configuration [INFERRED 0.95]
- **Mean-Node Contrastive Descriptor-Weight and Seed Pilot Sweep** — configs_representation_pilot_mean_node_contrastive_001_desc050_seed43_configuration, configs_representation_pilot_mean_node_contrastive_001_desc050_configuration, configs_representation_pilot_mean_node_contrastive_001_desc100_configuration [INFERRED 0.95]
- **Projector-Enabled VICReg Strength, Invariance-Space, and Seed Overlay Family** — configs_representation_projector_standard_strong_seed1337_configuration, configs_representation_projector_standard_strong_seed2027_configuration, configs_representation_projector_standard_strong_configuration, configs_representation_projector_standard_configuration, configs_representation_projector_strong_configuration, configs_representation_projector_configuration [INFERRED 0.95]
- **Representation VICReg Objective Variants** — configs_representation_strong_128d_strong_variance_covariance_objective, configs_representation_strong_vicreg_strong_variance_covariance_objective, configs_representation_variance_only_objective, configs_representation_v1_masked_graph_vicreg [INFERRED 0.85]
- **384-Dimensional Embedding Composition** — configs_representation_v1_dual_latent_spaces, inference_readme_vector_definition, inference_readme_promoted_384_dimensional_vector [INFERRED 0.95]
- **Training-to-Inference Molecule Policy Continuity** — configs_retrain_molecular_canonicalization_policy, inference_readme_molecule_acceptance_policy, inference_readme_row_stable_rejection_policy [INFERRED 0.95]

## Communities (28 total, 5 thin omitted)

### Community 0 - "train.py"
Cohesion: 0.18
Nodes (27): _all_reduce_mean(), _architecture(), _balanced_existence_loss(), _binary_histogram_metrics(), _build_model(), _covariance_diagnostics(), _distributed_context(), evaluate() (+19 more)

### Community 1 - "ValueError"
Cohesion: 0.20
Nodes (29): atomic_copy(), atomic_torch_save(), Path, _automatic_checkpoint_name(), _automatic_representation_calibrator(), _calibrator_expected_identity(), _check_gate(), export_embeddings() (+21 more)

### Community 2 - "cli.py"
Cohesion: 0.09
Nodes (56): _apply_run_directory(), _apply_training_budgets(), build_parser(), command_audit_downstream_overlap(), command_audit_training_exposure(), command_benchmark_descriptor_control(), command_benchmark_downstream(), command_benchmark_training() (+48 more)

### Community 3 - "Audit Disposition and Retraining Corrections"
Cohesion: 0.06
Nodes (57): Atomic Exact Resume Contract, Audit Disposition and Retraining Corrections, Molecular Encoder Capability Boundary, Exact-Denominator Distributed Validation, Explicit 13-Descriptor Contract, Fail-Closed Representation Promotion, Numerical GPU Reproducibility, Per-Graph Negative Sampling (+49 more)

### Community 4 - "atomic_write_json"
Cohesion: 0.06
Nodes (67): Schema, canonicalize(), CanonicalMolecule, _feature_factory(), featurize_molecule(), _hydrogen_bond_flags(), _one_hot(), _position_encoding() (+59 more)

### Community 5 - "Representation V1 Training Overlay"
Cohesion: 0.05
Nodes (45): Representation 128d Configuration, 128-Dimensional Graph Latent Space, Representation Contrastive 0.02 Configuration, NT-Xent Weight 0.02, Representation Contrastive 0.05 Configuration, NT-Xent Weight 0.05, Representation Contrastive 0.10 Configuration, NT-Xent Weight 0.10 (+37 more)

### Community 6 - "Full Promotion Evaluation"
Cohesion: 0.07
Nodes (41): Cosine-Tanimoto Spearman Sampling Protocol, Manuscript Accuracy Audit, Manuscript Wording and Units Corrections, Per-Fold StandardScaler Interpretation, Seed-42 ESOL Rounding, Seed-43 Calibration Protocol, Separate Edge Decoder MLPs, Table 1 Raw Hybrid Vector Protocol (+33 more)

### Community 7 - "model.py"
Cohesion: 0.13
Nodes (13): corrupt_graph_inputs(), CorruptedGraph, DeterministicGINEEncoder, GraphConditionedEdgeDecoder, grouped_feature_loss(), kl_divergence(), Any, A deterministic atom encoder for transferable molecular representations. (+5 more)

### Community 8 - "data.py"
Cohesion: 0.12
Nodes (26): Batch, Data, _balanced_allocation(), finite_batches(), _finite_shard_plan(), _finite_shard_window_plan(), graph_from_shard(), InfiniteGraphBatchIterator (+18 more)

### Community 9 - "Combined ZINC-PubChem Retraining Configuration"
Cohesion: 0.10
Nodes (31): Bemis-Murcko Scaffold Hash Split, Node-Budgeted 500,000-Step Retraining Schedule, Canonical Isomeric SMILES Deduplication, Combined ZINC-PubChem Dataset, Combined ZINC-PubChem Retraining Configuration, Thirteen-Column Descriptor Schema, Four-Layer GINE Latent Model, 8,192-Graph Sharding (+23 more)

### Community 10 - "Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)"
Cohesion: 0.15
Nodes (29): Pilot Contrastive 0.005 Configuration, Low-Contrastive Pilot, Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers), Pilot Projector Contrastive Configuration (weight 0.01), Pilot Projector Contrastive Configuration (weight 0.01, descriptor weight 0.50, 15k steps), Projector-Space Contrastive Masked-Graph Objective with VICReg Terms Disabled, Pilot Projector Contrastive Configuration (weight 0.01, seed 43), Pilot Projector Contrastive Configuration (weight 0.02) (+21 more)

### Community 11 - "generate_embeddings.py"
Cohesion: 0.16
Nodes (27): build_parser(), canonicalize_input(), encode_batch(), fsync_text_handle(), InferenceError, load_json_object(), load_model_bundle(), main() (+19 more)

### Community 12 - "RuntimeError"
Cohesion: 0.09
Nodes (62): RuntimeError, descriptor_names(), audit_pretraining_overlap(), benchmark_descriptor_control(), _dataset_source(), _descriptor_matrix(), _groups_digest(), _identity_digest() (+54 more)

### Community 13 - "run_representation_probes"
Cohesion: 0.27
Nodes (17): _chemical_records(), _embedding_diagnostics(), _held_out_values(), _load_embedding_payload(), _molecules_and_labels(), Any, Mol, ndarray (+9 more)

### Community 14 - "build"
Cohesion: 0.28
Nodes (15): Paragraph, build(), find_paragraph(), fmt(), insert_after(), load_json(), metric_summary(), parse_args() (+7 more)

### Community 15 - "audit_step"
Cohesion: 0.44
Nodes (11): add_check(), audit_step(), finite_number(), format_number(), main(), nested(), Any, Path (+3 more)

### Community 17 - "Descriptor Schema Configuration"
Cohesion: 0.67
Nodes (3): Descriptor Schema Configuration, Thirteen Molecular Descriptor Targets, Train-Split Descriptor Standardization

### Community 22 - "sample_per_graph_negatives"
Cohesion: 0.20
Nodes (14): assert_valid_candidates(), _edge_tensor(), _pair_template(), ndarray, Tensor, Return lexicographic upper-triangle pairs for a graph size., Sample unique undirected negatives independently inside each graph. Easy…, sample_per_graph_negatives() (+6 more)

### Community 23 - "Manuscript rev3 audit artifacts"
Cohesion: 0.29
Nodes (6): Exact downstream/pretraining identity overlap, Exact seed-42 training exposure, Frozen 13-descriptor-only downstream control, Manuscript rev3 audit artifacts, Promotion chronology and terminology, Reproduction commands

### Community 24 - "MolecularRepresentationModel"
Cohesion: 0.15
Nodes (12): MolecularRepresentationModel, nt_xent_loss(), Tensor, Masked graph autoencoder with an explicit deterministic molecule vector. Every…, Return deterministic atom and molecule embeddings., Combine already encoded graph and atom blocks., Concatenate raw graph and mean-atom blocks before train calibration., Apply immutable train-split coordinate statistics to raw vectors. (+4 more)

### Community 25 - "MolecularVGAE"
Cohesion: 0.20
Nodes (16): DistributedDataParallel, MolecularVGAE, NegativeCandidates, Select hard-pool logits while retaining gradients for selected values., select_hard_negative_logits(), _legacy_losses_for_batch(), _losses_for_batch(), Return a plan-scoped seed without changing immutable graph identity. (+8 more)

### Community 26 - "_TrainingBatchPrefetcher"
Cohesion: 0.33
Nodes (5): Future, _prepare_training_batch(), _PreparedTrainingBatch, Prepare one deterministic batch ahead without advancing checkpoint cursors., _TrainingBatchPrefetcher

### Community 27 - "checkpoint.py"
Cohesion: 0.39
Nodes (8): Optimizer, build_checkpoint(), capture_rng_state(), gather_rank_objects(), Any, Module, restore_rng_state(), validate_checkpoint()

## Knowledge Gaps
- **42 isolated node(s):** `gmolai-retrain`, `common.sh script`, `run_benchmark_in_container.sh script`, `submit_pipeline.sh script`, `Exact seed-42 training exposure` (+37 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `Semantic Promotion Suite` (3× useful, score=2.992216627)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MolecularRepresentationModel` connect `MolecularRepresentationModel` to `train.py`, `ValueError`, `atomic_write_json`, `model.py`, `generate_embeddings.py`, `RuntimeError`, `MolecularVGAE`, `_TrainingBatchPrefetcher`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `Representation V1 Training Overlay` connect `Representation V1 Training Overlay` to `Combined ZINC-PubChem Retraining Configuration`, `Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `train()` connect `train.py` to `ValueError`, `cli.py`, `atomic_write_json`, `data.py`, `RuntimeError`, `MolecularVGAE`, `_TrainingBatchPrefetcher`, `checkpoint.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `RuntimeError` (e.g. with `build()` and `find_paragraph()`) actually correct?**
  _`RuntimeError` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `ValueError` (e.g. with `audit_step()` and `main()`) actually correct?**
  _`ValueError` has 46 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `train()` (e.g. with `RuntimeError` and `_request_stop()`) actually correct?**
  _`train()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `build_parser()` (e.g. with `command_audit_downstream_overlap()` and `command_audit_training_exposure()`) actually correct?**
  _`build_parser()` has 22 INFERRED edges - model-reasoned connections that need verification._