# Graph Report - .  (2026-08-12)

## Corpus Check
- 66 files · ~103,452 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 893 nodes · 2264 edges · 39 communities (31 shown, 8 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 166 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Data and Artifact Utilities
- Frozen Encoder Benchmark
- Downstream Evaluation Audits
- Scientific Audit and Repair
- CLI and Audit Commands
- Representation Training Design
- Promotion Evidence and Figures
- Graph Data Loading
- Training and Evaluation
- Corpus and Representation Contract
- Embedding Export and Promotion
- Contrastive Training Configurations
- Standalone Inference
- Manuscript Revision 5
- Training Loss Computation
- Graph Encoder Architecture
- Molecular Representation Model
- Representation Probes
- Manuscript Revision 4
- Model Tests
- Negative Sampling
- Manuscript Revision 3
- Checkpoint Management
- Promotion Trajectory Summary
- Training Prefetch Pipeline
- Revision 3 Audit Bundle
- Revision 4 Exposure Audit
- Revision 5 Evidence Structure
- Exposure Audit Memory
- Revision 5 Memory
- Arrhenius Container Helpers
- Descriptor Schema
- Container Benchmark Runner
- Screening Configuration
- Pipeline Submission
- Package Root
- Argument Parsing
- Module Namespace
- Tensor Namespace

## God Nodes (most connected - your core abstractions)
1. `train()` - 33 edges
2. `atomic_write_json()` - 32 edges
3. `build_parser()` - 30 edges
4. `MolecularRepresentationModel` - 25 edges
5. `Representation V1 Training Overlay` - 25 edges
6. `_print()` - 22 edges
7. `atomic_write_json()` - 21 edges
8. `_load()` - 21 edges
9. `benchmark_moleculenet()` - 21 edges
10. `benchmark_descriptor_control()` - 21 edges

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

## Communities (39 total, 8 thin omitted)

### Community 0 - "Data and Artifact Utilities"
Cohesion: 0.06
Nodes (74): Schema, canonicalize(), CanonicalMolecule, _feature_factory(), featurize_molecule(), _hydrogen_bond_flags(), _one_hot(), _position_encoding() (+66 more)

### Community 1 - "Frozen Encoder Benchmark"
Cohesion: 0.07
Nodes (76): Exception, load_kermt(), load_molai(), load_molclr(), load_molformer(), load_morgan(), load_smi_ted(), main() (+68 more)

### Community 2 - "Downstream Evaluation Audits"
Cohesion: 0.07
Nodes (78): Module, audit_pretraining_overlap(), benchmark_descriptor_control(), _dataset_source(), _descriptor_matrix(), _groups_digest(), _identity_digest(), _indices_digest() (+70 more)

### Community 3 - "Scientific Audit and Repair"
Cohesion: 0.06
Nodes (57): Atomic Exact Resume Contract, Audit Disposition and Retraining Corrections, Molecular Encoder Capability Boundary, Exact-Denominator Distributed Validation, Explicit 13-Descriptor Contract, Fail-Closed Representation Promotion, Numerical GPU Reproducibility, Per-Graph Negative Sampling (+49 more)

### Community 4 - "CLI and Audit Commands"
Cohesion: 0.12
Nodes (36): ArgumentParser, _apply_run_directory(), _apply_training_budgets(), build_parser(), command_audit_downstream_exposure(), command_audit_downstream_overlap(), command_audit_training_exposure(), command_benchmark_descriptor_control() (+28 more)

### Community 5 - "Representation Training Design"
Cohesion: 0.05
Nodes (45): Representation 128d Configuration, 128-Dimensional Graph Latent Space, Representation Contrastive 0.02 Configuration, NT-Xent Weight 0.02, Representation Contrastive 0.05 Configuration, NT-Xent Weight 0.05, Representation Contrastive 0.10 Configuration, NT-Xent Weight 0.10 (+37 more)

### Community 6 - "Promotion Evidence and Figures"
Cohesion: 0.07
Nodes (41): Cosine-Tanimoto Spearman Sampling Protocol, Manuscript Accuracy Audit, Manuscript Wording and Units Corrections, Per-Fold StandardScaler Interpretation, Seed-42 ESOL Rounding, Seed-43 Calibration Protocol, Separate Edge Decoder MLPs, Table 1 Raw Hybrid Vector Protocol (+33 more)

### Community 7 - "Graph Data Loading"
Cohesion: 0.13
Nodes (24): Batch, Data, _balanced_allocation(), finite_batches(), _finite_shard_plan(), _finite_shard_window_plan(), graph_from_shard(), InfiniteGraphBatchIterator (+16 more)

### Community 8 - "Training and Evaluation"
Cohesion: 0.17
Nodes (30): load_graph_manifest(), Path, validate_feature_schema(), _architecture(), _binary_histogram_metrics(), _build_model(), _covariance_diagnostics(), _distributed_context() (+22 more)

### Community 9 - "Corpus and Representation Contract"
Cohesion: 0.10
Nodes (31): Bemis-Murcko Scaffold Hash Split, Node-Budgeted 500,000-Step Retraining Schedule, Canonical Isomeric SMILES Deduplication, Combined ZINC-PubChem Dataset, Combined ZINC-PubChem Retraining Configuration, Thirteen-Column Descriptor Schema, Four-Layer GINE Latent Model, 8,192-Graph Sharding (+23 more)

### Community 10 - "Embedding Export and Promotion"
Cohesion: 0.17
Nodes (29): atomic_torch_save(), Path, MolecularRepresentationModel, MolecularVGAE, _automatic_checkpoint_name(), _automatic_representation_calibrator(), _calibrator_expected_identity(), _check_gate() (+21 more)

### Community 11 - "Contrastive Training Configurations"
Cohesion: 0.15
Nodes (29): Pilot Contrastive 0.005 Configuration, Low-Contrastive Pilot, Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers), Pilot Projector Contrastive Configuration (weight 0.01), Pilot Projector Contrastive Configuration (weight 0.01, descriptor weight 0.50, 15k steps), Projector-Space Contrastive Masked-Graph Objective with VICReg Terms Disabled, Pilot Projector Contrastive Configuration (weight 0.01, seed 43), Pilot Projector Contrastive Configuration (weight 0.02) (+21 more)

### Community 12 - "Standalone Inference"
Cohesion: 0.16
Nodes (27): build_parser(), canonicalize_input(), encode_batch(), fsync_text_handle(), InferenceError, load_json_object(), load_model_bundle(), main() (+19 more)

### Community 13 - "Manuscript Revision 5"
Cohesion: 0.22
Nodes (21): Counter, assert_source_contract(), build(), delete_paragraph(), find_exact_paragraph(), find_paragraph(), insert_evidence_roles_table(), new_paragraph_before() (+13 more)

### Community 14 - "Training Loss Computation"
Cohesion: 0.16
Nodes (19): DistributedDataParallel, RuntimeError, grouped_feature_loss(), Use categorical CE for one-hot groups and BCE for binary features., NegativeCandidates, Select hard-pool logits while retaining gradients for selected values., select_hard_negative_logits(), _all_reduce_mean() (+11 more)

### Community 15 - "Graph Encoder Architecture"
Cohesion: 0.14
Nodes (11): corrupt_graph_inputs(), CorruptedGraph, DeterministicGINEEncoder, GraphConditionedEdgeDecoder, kl_divergence(), Any, A deterministic atom encoder for transferable molecular representations., Decode an unordered atom pair while forcing use of its graph embedding. (+3 more)

### Community 16 - "Molecular Representation Model"
Cohesion: 0.17
Nodes (14): MolecularRepresentationModel, nt_xent_loss(), Tensor, Masked graph autoencoder with an explicit deterministic molecule vector. Every…, Return deterministic atom and molecule embeddings., Combine already encoded graph and atom blocks., Concatenate raw graph and mean-atom blocks before train calibration., Apply immutable train-split coordinate statistics to raw vectors. (+6 more)

### Community 17 - "Representation Probes"
Cohesion: 0.27
Nodes (17): _chemical_records(), _embedding_diagnostics(), _held_out_values(), _load_embedding_payload(), _molecules_and_labels(), Any, Mol, ndarray (+9 more)

### Community 18 - "Manuscript Revision 4"
Cohesion: 0.27
Nodes (17): build(), _dataset_counts_sentence(), find_paragraph(), _format_seen(), insert_after(), insert_exposure_table(), load_json(), parse_args() (+9 more)

### Community 19 - "Model Tests"
Cohesion: 0.20
Nodes (14): MolecularVGAE, _losses_for_batch(), Tensor, The descriptor head must see the same representation in train/eval paths., _representation_model_and_batch(), test_complete_corrected_objective_backpropagates(), test_contrastive_objective_can_target_canonical_mean_node_block(), test_descriptor_prediction_is_independent_of_posterior_sampling() (+6 more)

### Community 20 - "Negative Sampling"
Cohesion: 0.20
Nodes (14): assert_valid_candidates(), _edge_tensor(), _pair_template(), ndarray, Tensor, Return lexicographic upper-triangle pairs for a graph size., Sample unique undirected negatives independently inside each graph. Easy…, sample_per_graph_negatives() (+6 more)

### Community 21 - "Manuscript Revision 3"
Cohesion: 0.28
Nodes (15): build(), find_paragraph(), fmt(), insert_after(), load_json(), metric_summary(), parse_args(), Any (+7 more)

### Community 22 - "Checkpoint Management"
Cohesion: 0.29
Nodes (11): Optimizer, atomic_copy(), atomic_torch_save(), build_checkpoint(), capture_rng_state(), gather_rank_objects(), Any, Module (+3 more)

### Community 23 - "Promotion Trajectory Summary"
Cohesion: 0.44
Nodes (11): add_check(), audit_step(), finite_number(), format_number(), main(), nested(), Any, Path (+3 more)

### Community 24 - "Training Prefetch Pipeline"
Cohesion: 0.29
Nodes (5): Future, _prepare_training_batch(), _PreparedTrainingBatch, Prepare one deterministic batch ahead without advancing checkpoint cursors., _TrainingBatchPrefetcher

### Community 25 - "Revision 3 Audit Bundle"
Cohesion: 0.29
Nodes (6): Exact downstream/pretraining identity overlap, Exact seed-42 training exposure, Frozen 13-descriptor-only downstream control, Manuscript rev3 audit artifacts, Promotion chronology and terminology, Reproduction commands

### Community 26 - "Revision 4 Exposure Audit"
Cohesion: 0.29
Nodes (6): Aggregate pretraining exposure, Checkpoint-resolved downstream exposure, Exact reconstruction and validation, Files, Manuscript rev4: exact downstream-molecule exposure audit, Reproduction

### Community 27 - "Revision 5 Evidence Structure"
Cohesion: 0.29
Nodes (6): Experimental chronology made explicit, Interpretation-affecting wording changes, Manuscript rev5: evidence-source reorganization, Old-to-new section mapping, Rebuild, Tables, figures and validation

### Community 28 - "Exposure Audit Memory"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k., Source Nodes

### Community 29 - "Revision 5 Memory"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Create gmolai-rev5.docx from the current authoritative gmolai-rev4.docx., Source Nodes

### Community 31 - "Descriptor Schema"
Cohesion: 0.67
Nodes (3): Descriptor Schema Configuration, Thirteen Molecular Descriptor Targets, Train-Split Descriptor Standardization

## Knowledge Gaps
- **58 isolated node(s):** `common.sh script`, `run_benchmark_in_container.sh script`, `submit_pipeline.sh script`, `Exact seed-42 training exposure`, `Exact downstream/pretraining identity overlap` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_write_json()` connect `Data and Artifact Utilities` to `Training and Evaluation`, `Representation Probes`, `Downstream Evaluation Audits`, `Embedding Export and Promotion`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `run_representation_probes()` connect `Representation Probes` to `Data and Artifact Utilities`, `Frozen Encoder Benchmark`, `CLI and Audit Commands`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Why does `main()` connect `Frozen Encoder Benchmark` to `Representation Probes`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `train()` (e.g. with `RuntimeError` and `_request_stop()`) actually correct?**
  _`train()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `build_parser()` (e.g. with `command_audit_downstream_exposure()` and `command_audit_downstream_overlap()`) actually correct?**
  _`build_parser()` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `MolecularRepresentationModel` (e.g. with `InferenceError` and `ModelBundle`) actually correct?**
  _`MolecularRepresentationModel` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `common.sh script`, `run_benchmark_in_container.sh script`, `submit_pipeline.sh script` to the rest of the system?**
  _58 weakly-connected nodes found - possible documentation gaps or missing edges._