# Graph Report - gMolAI-retrain  (2026-08-12)

## Corpus Check
- 72 files · ~87,006 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 799 nodes · 2137 edges · 35 communities (30 shown, 5 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 227 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0a515043`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Any
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
- test_model.py
- train.py
- train
- update_manuscript_rev4.py
- Manuscript rev4: exact downstream-molecule exposure audit
- Q: Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k.
- update_manuscript_rev5.py
- Manuscript rev5: evidence-source reorganization
- Q: Create gmolai-rev5.docx from the current authoritative gmolai-rev4.docx.
- nt_xent_loss

## God Nodes (most connected - your core abstractions)
1. `train()` - 36 edges
2. `atomic_write_json()` - 32 edges
3. `build_parser()` - 30 edges
4. `MolecularRepresentationModel` - 29 edges
5. `benchmark_moleculenet()` - 25 edges
6. `Representation V1 Training Overlay` - 25 edges
7. `_load()` - 23 edges
8. `benchmark_descriptor_control()` - 23 edges
9. `_print()` - 22 edges
10. `audit_downstream_checkpoint_exposure()` - 20 edges

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

## Communities (35 total, 5 thin omitted)

### Community 0 - "Any"
Cohesion: 0.27
Nodes (16): _all_reduce_mean(), _architecture(), _binary_histogram_metrics(), _build_model(), _covariance_diagnostics(), evaluate(), evaluate_saved(), _finalize_group_confusions() (+8 more)

### Community 1 - "ValueError"
Cohesion: 0.10
Nodes (54): atomic_copy(), atomic_torch_save(), Path, benchmark_moleculenet(), _classification_probe(), _encode_molecules(), _inner_group_folds(), _morgan_features() (+46 more)

### Community 2 - "cli.py"
Cohesion: 0.12
Nodes (36): _apply_run_directory(), _apply_training_budgets(), build_parser(), command_audit_downstream_exposure(), command_audit_downstream_overlap(), command_audit_training_exposure(), command_benchmark_descriptor_control(), command_benchmark_downstream() (+28 more)

### Community 3 - "Audit Disposition and Retraining Corrections"
Cohesion: 0.06
Nodes (57): Atomic Exact Resume Contract, Audit Disposition and Retraining Corrections, Molecular Encoder Capability Boundary, Exact-Denominator Distributed Validation, Explicit 13-Descriptor Contract, Fail-Closed Representation Promotion, Numerical GPU Reproducibility, Per-Graph Negative Sampling (+49 more)

### Community 4 - "atomic_write_json"
Cohesion: 0.08
Nodes (58): Schema, apply_training_plan(), canonical_json(), ConfigurationError, _deep_update(), _expand(), load_config(), load_yaml() (+50 more)

### Community 5 - "Representation V1 Training Overlay"
Cohesion: 0.05
Nodes (45): Representation 128d Configuration, 128-Dimensional Graph Latent Space, Representation Contrastive 0.02 Configuration, NT-Xent Weight 0.02, Representation Contrastive 0.05 Configuration, NT-Xent Weight 0.05, Representation Contrastive 0.10 Configuration, NT-Xent Weight 0.10 (+37 more)

### Community 6 - "Full Promotion Evaluation"
Cohesion: 0.07
Nodes (41): Cosine-Tanimoto Spearman Sampling Protocol, Manuscript Accuracy Audit, Manuscript Wording and Units Corrections, Per-Fold StandardScaler Interpretation, Seed-42 ESOL Rounding, Seed-43 Calibration Protocol, Separate Edge Decoder MLPs, Table 1 Raw Hybrid Vector Protocol (+33 more)

### Community 7 - "model.py"
Cohesion: 0.14
Nodes (11): corrupt_graph_inputs(), CorruptedGraph, DeterministicGINEEncoder, GraphConditionedEdgeDecoder, Any, A deterministic atom encoder for transferable molecular representations., Decode an unordered atom pair while forcing use of its graph embedding., ResidualGINEEncoder (+3 more)

### Community 8 - "data.py"
Cohesion: 0.12
Nodes (25): Batch, Data, _balanced_allocation(), finite_batches(), _finite_shard_plan(), _finite_shard_window_plan(), graph_from_shard(), InfiniteGraphBatchIterator (+17 more)

### Community 9 - "Combined ZINC-PubChem Retraining Configuration"
Cohesion: 0.10
Nodes (31): Bemis-Murcko Scaffold Hash Split, Node-Budgeted 500,000-Step Retraining Schedule, Canonical Isomeric SMILES Deduplication, Combined ZINC-PubChem Dataset, Combined ZINC-PubChem Retraining Configuration, Thirteen-Column Descriptor Schema, Four-Layer GINE Latent Model, 8,192-Graph Sharding (+23 more)

### Community 10 - "Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)"
Cohesion: 0.15
Nodes (29): Pilot Contrastive 0.005 Configuration, Low-Contrastive Pilot, Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers), Pilot Projector Contrastive Configuration (weight 0.01), Pilot Projector Contrastive Configuration (weight 0.01, descriptor weight 0.50, 15k steps), Projector-Space Contrastive Masked-Graph Objective with VICReg Terms Disabled, Pilot Projector Contrastive Configuration (weight 0.01, seed 43), Pilot Projector Contrastive Configuration (weight 0.02) (+21 more)

### Community 11 - "generate_embeddings.py"
Cohesion: 0.10
Nodes (42): build_parser(), canonicalize_input(), encode_batch(), fsync_text_handle(), InferenceError, load_json_object(), load_model_bundle(), main() (+34 more)

### Community 12 - "RuntimeError"
Cohesion: 0.10
Nodes (56): RuntimeError, descriptor_names(), audit_pretraining_overlap(), benchmark_descriptor_control(), _dataset_source(), _descriptor_matrix(), _groups_digest(), _identity_digest() (+48 more)

### Community 13 - "run_representation_probes"
Cohesion: 0.27
Nodes (17): _chemical_records(), _embedding_diagnostics(), _held_out_values(), _load_embedding_payload(), _molecules_and_labels(), Any, Mol, ndarray (+9 more)

### Community 14 - "build"
Cohesion: 0.28
Nodes (15): build(), find_paragraph(), fmt(), insert_after(), load_json(), metric_summary(), parse_args(), Any (+7 more)

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
Cohesion: 0.21
Nodes (8): MolecularRepresentationModel, Tensor, Masked graph autoencoder with an explicit deterministic molecule vector. Every…, Return deterministic atom and molecule embeddings., Combine already encoded graph and atom blocks., Concatenate raw graph and mean-atom blocks before train calibration., Apply immutable train-split coordinate statistics to raw vectors., Return the legacy unit-block hybrid molecule vector. The unit-normalized graph…

### Community 25 - "test_model.py"
Cohesion: 0.14
Nodes (21): DistributedDataParallel, MolecularVGAE, _losses_for_batch(), main(), Run directly with: torchrun --standalone --nproc_per_node=2 tests/ddp_smoke.py, valid_features(), Tensor, The descriptor head must see the same representation in train/eval paths. (+13 more)

### Community 26 - "train.py"
Cohesion: 0.14
Nodes (20): Future, grouped_feature_loss(), kl_divergence(), Return invariance, variance-floor, and covariance-redundancy losses., Use categorical CE for one-hot groups and BCE for binary features., vicreg_terms(), NegativeCandidates, Select hard-pool logits while retaining gradients for selected values. (+12 more)

### Community 27 - "train"
Cohesion: 0.18
Nodes (17): Optimizer, build_checkpoint(), capture_rng_state(), gather_rank_objects(), Any, Module, restore_rng_state(), validate_checkpoint() (+9 more)

### Community 28 - "update_manuscript_rev4.py"
Cohesion: 0.27
Nodes (17): build(), _dataset_counts_sentence(), find_paragraph(), _format_seen(), insert_after(), insert_exposure_table(), load_json(), parse_args() (+9 more)

### Community 29 - "Manuscript rev4: exact downstream-molecule exposure audit"
Cohesion: 0.29
Nodes (6): Aggregate pretraining exposure, Checkpoint-resolved downstream exposure, Exact reconstruction and validation, Files, Manuscript rev4: exact downstream-molecule exposure audit, Reproduction

### Community 30 - "Q: Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k., Source Nodes

### Community 31 - "update_manuscript_rev5.py"
Cohesion: 0.22
Nodes (21): Counter, assert_source_contract(), build(), delete_paragraph(), find_exact_paragraph(), find_paragraph(), insert_evidence_roles_table(), new_paragraph_before() (+13 more)

### Community 32 - "Manuscript rev5: evidence-source reorganization"
Cohesion: 0.29
Nodes (6): Experimental chronology made explicit, Interpretation-affecting wording changes, Manuscript rev5: evidence-source reorganization, Old-to-new section mapping, Rebuild, Tables, figures and validation

### Community 33 - "Q: Create gmolai-rev5.docx from the current authoritative gmolai-rev4.docx."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Create gmolai-rev5.docx from the current authoritative gmolai-rev4.docx., Source Nodes

### Community 34 - "nt_xent_loss"
Cohesion: 0.67
Nodes (3): nt_xent_loss(), Symmetric cross-view InfoNCE loss for graph embeddings., test_nt_xent_prefers_correct_cross_view_pairs()

## Knowledge Gaps
- **58 isolated node(s):** `gmolai-retrain`, `common.sh script`, `run_benchmark_in_container.sh script`, `submit_pipeline.sh script`, `Exact seed-42 training exposure` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `Semantic Promotion Suite` (3× useful, score=2.93790641)
- `Manuscript rev4: exact downstream-molecule exposure audit` (2× useful, score=1.989875957)
- `Promotion Integrity Gates` (2× useful, score=1.979643813)
- `Validation Evidence Bundle` (2× useful, score=1.979643813)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MolecularRepresentationModel` connect `MolecularRepresentationModel` to `Any`, `ValueError`, `model.py`, `generate_embeddings.py`, `RuntimeError`, `test_model.py`, `train.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `Representation V1 Training Overlay` connect `Representation V1 Training Overlay` to `Combined ZINC-PubChem Retraining Configuration`, `Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Why does `train()` connect `train` to `Any`, `ValueError`, `cli.py`, `atomic_write_json`, `data.py`, `RuntimeError`, `test_model.py`, `train.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `RuntimeError` (e.g. with `build()` and `find_paragraph()`) actually correct?**
  _`RuntimeError` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 51 inferred relationships involving `ValueError` (e.g. with `audit_step()` and `main()`) actually correct?**
  _`ValueError` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `train()` (e.g. with `RuntimeError` and `_request_stop()`) actually correct?**
  _`train()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `build_parser()` (e.g. with `command_audit_downstream_exposure()` and `command_audit_downstream_overlap()`) actually correct?**
  _`build_parser()` has 23 INFERRED edges - model-reasoned connections that need verification._