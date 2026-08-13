# Graph Report - gMolAI-retrain  (2026-08-13)

## Corpus Check
- 178 files · ~282,777 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1294 nodes · 3450 edges · 67 communities (57 shown, 10 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 359 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `333abda9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- chem.py
- adapter.py
- RuntimeError
- Scientific Audit and Repair
- cli.py
- Representation V1 Training Overlay
- Promotion Evidence and Figures
- InfiniteGraphBatchIterator
- train.py
- Self-Contained Inference Entry Point
- ValueError
- Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)
- generate_embeddings.py
- update_manuscript_rev5.py
- OptimizedSmilesEncoder
- Tensor
- tune.py
- run_representation_probes
- update_manuscript_rev4.py
- model.py
- NegativeCandidates
- build
- checkpoint.py
- audit_step
- MolecularRepresentationModel
- Manuscript rev3 audit artifacts
- Manuscript rev4: exact downstream-molecule exposure audit
- Manuscript rev5: evidence-source reorganization
- Q: Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k.
- Q: Create gmolai-rev5.docx from the current authoritative gmolai-rev4.docx.
- common.sh
- Descriptor Schema Configuration
- run_benchmark_in_container.sh
- Representation Screening Configuration (5k steps, frequent validation, no resume)
- submit_pipeline.sh
- gmolai-retrain
- load_protocol
- gmolai_retrain/fast_graph.py
- speed_adapter.py
- gMolAI comparator and encoding-speed benchmark: feasibility memo
- MoleculeNet development panel plus HIV: completed results
- update_manuscript_rev6.py
- data.py
- test_fast_inference.py
- Locked internal test-partition encoder benchmark: completed results
- Q: Explore and fully understand this repository. Expanded query tokens: architecture, pipeline, data, model, training, inference, evaluation, validation, checkpoints, reproducibility, promotion, audit.
- Strong VICReg Variance-Covariance Objective
- Q: Complete the optimized single-GPU speed benchmark for gMolAI, Morgan, and competitor models at batch sizes 64, 128, 256, and 512 with the qualified worker count, then replace old speed artifacts and update documentation.
- Combined ZINC-PubChem Dataset
- Frozen protocol: locked internal test-partition benchmark
- Combined ZINC-PubChem Retraining Configuration
- Frozen speed-benchmark protocol
- Raw Molecule Data Flow
- gMolAI inference speed optimization
- KERMT v2 native batch-dependence audit
- Q: Audit the completed MoleculeNet plus HIV benchmark implementation and artifact flow
- Q: Calculate the embedding time for each model and Morgan from the MoleculeNet plus HIV benchmark
- Q: done
- Q: gmolai is the smallest model but it is much slower compare to others. why is it?
- Common locked-test encoding-speed benchmark
- speed/RESULTS.md
- Invariance-Only Configuration
- run_encode_example.sh
- run_final.sh
- Q: is the use face folder 'inference' current? I mean is it using the speed-optimized pipeline instead of old slow one?
- extra-benchmark/README.md
- Graph-256 and Node-128 Latent Spaces

## God Nodes (most connected - your core abstractions)
1. `MolecularRepresentationModel` - 42 edges
2. `train()` - 36 edges
3. `atomic_write_json()` - 32 edges
4. `build_parser()` - 30 edges
5. `benchmark_moleculenet()` - 25 edges
6. `Representation V1 Training Overlay` - 25 edges
7. `featurize_molecule()` - 23 edges
8. `_load()` - 23 edges
9. `benchmark_descriptor_control()` - 23 edges
10. `_print()` - 22 edges

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

## Communities (67 total, 10 thin omitted)

### Community 0 - "chem.py"
Cohesion: 0.15
Nodes (23): canonicalize(), CanonicalMolecule, _feature_factory(), featurize_molecule(), _hydrogen_bond_flags(), _one_hot(), _position_encoding(), Any (+15 more)

### Community 1 - "adapter.py"
Cohesion: 0.07
Nodes (78): Exception, load_kermt(), load_molai(), load_molclr(), load_molformer(), load_morgan(), load_smi_ted(), main() (+70 more)

### Community 2 - "RuntimeError"
Cohesion: 0.08
Nodes (71): RuntimeError, descriptor_names(), audit_pretraining_overlap(), benchmark_descriptor_control(), _dataset_source(), _descriptor_matrix(), _groups_digest(), _identity_digest() (+63 more)

### Community 3 - "Scientific Audit and Repair"
Cohesion: 0.06
Nodes (57): Atomic Exact Resume Contract, Audit Disposition and Retraining Corrections, Molecular Encoder Capability Boundary, Exact-Denominator Distributed Validation, Explicit 13-Descriptor Contract, Fail-Closed Representation Promotion, Numerical GPU Reproducibility, Per-Graph Negative Sampling (+49 more)

### Community 4 - "cli.py"
Cohesion: 0.05
Nodes (92): Schema, _apply_run_directory(), _apply_training_budgets(), build_parser(), command_audit_downstream_exposure(), command_audit_downstream_overlap(), command_audit_training_exposure(), command_benchmark_descriptor_control() (+84 more)

### Community 5 - "Representation V1 Training Overlay"
Cohesion: 0.07
Nodes (29): Representation Contrastive 0.02 Configuration, NT-Xent Weight 0.02, Representation Contrastive 0.05 Configuration, NT-Xent Weight 0.05, Representation Contrastive 0.10 Configuration, NT-Xent Weight 0.10, Representation Contrastive 0.20 Configuration, NT-Xent Weight 0.20 (+21 more)

### Community 6 - "Promotion Evidence and Figures"
Cohesion: 0.07
Nodes (41): Cosine-Tanimoto Spearman Sampling Protocol, Manuscript Accuracy Audit, Manuscript Wording and Units Corrections, Per-Fold StandardScaler Interpretation, Seed-42 ESOL Rounding, Seed-43 Calibration Protocol, Separate Edge Decoder MLPs, Table 1 Raw Hybrid Vector Protocol (+33 more)

### Community 7 - "InfiniteGraphBatchIterator"
Cohesion: 0.21
Nodes (6): Batch, Future, InfiniteGraphBatchIterator, Deterministic shard-exclusive stream with an exactly serializable cursor., _prepare_training_batch(), _PreparedTrainingBatch

### Community 8 - "train.py"
Cohesion: 0.13
Nodes (40): DistributedDataParallel, grouped_feature_loss(), Use categorical CE for one-hot groups and BCE for binary features., _all_reduce_mean(), _architecture(), _balanced_existence_loss(), _binary_histogram_metrics(), _build_model() (+32 more)

### Community 9 - "Self-Contained Inference Entry Point"
Cohesion: 0.27
Nodes (11): CSV Molecular Embedding Inference, Embedding Output Contract, Checkpoint and Calibrator Model Asset Bundle, SHA-256 and Model Identity Validation, Training-Time Molecule Acceptance Policy, Hardware-Conditional Numerical Determinism Contract, Promoted gMolAI 384-Dimensional Molecular Vector, Embedding Provenance Metadata Sidecar (+3 more)

### Community 10 - "ValueError"
Cohesion: 0.17
Nodes (35): atomic_copy(), atomic_torch_save(), Path, benchmark_moleculenet(), _morgan_features(), Select the public vector without depending on diagnostic feature exports., _select_representation_embedding(), _automatic_checkpoint_name() (+27 more)

### Community 11 - "Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)"
Cohesion: 0.15
Nodes (29): Pilot Contrastive 0.005 Configuration, Low-Contrastive Pilot, Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers), Pilot Projector Contrastive Configuration (weight 0.01), Pilot Projector Contrastive Configuration (weight 0.01, descriptor weight 0.50, 15k steps), Projector-Space Contrastive Masked-Graph Objective with VICReg Terms Disabled, Pilot Projector Contrastive Configuration (weight 0.01, seed 43), Pilot Projector Contrastive Configuration (weight 0.02) (+21 more)

### Community 12 - "generate_embeddings.py"
Cohesion: 0.16
Nodes (27): build_parser(), canonicalize_input(), encode_batch(), fsync_text_handle(), InferenceError, load_json_object(), load_model_bundle(), main() (+19 more)

### Community 13 - "update_manuscript_rev5.py"
Cohesion: 0.22
Nodes (21): assert_source_contract(), build(), delete_paragraph(), find_exact_paragraph(), find_paragraph(), insert_evidence_roles_table(), new_paragraph_before(), next_table_element() (+13 more)

### Community 14 - "OptimizedSmilesEncoder"
Cohesion: 0.17
Nodes (8): initialize_worker(), Warm immutable RDKit state and prevent nested CPU oversubscription., smiles_tasks(), calibrated_embedding_numpy(), OptimizedSmilesEncoder, inference_mode, Multiprocess RDKit + direct packed-array + equivalent GINE inference., test_smiles_tasks_respect_graph_and_node_budgets()

### Community 15 - "Tensor"
Cohesion: 0.10
Nodes (14): DeterministicGINEEncoder, GraphConditionedEdgeDecoder, Any, Tensor, A deterministic atom encoder for transferable molecular representations., Decode an unordered atom pair while forcing use of its graph embedding., Return deterministic atom and molecule embeddings., Combine already encoded graph and atom blocks. (+6 more)

### Community 16 - "tune.py"
Cohesion: 0.08
Nodes (58): ProcessPoolExecutor, main(), main(), Path, sha256_file(), _category(), fast_featurize_molecule(), _hbond_factory() (+50 more)

### Community 17 - "run_representation_probes"
Cohesion: 0.27
Nodes (17): _chemical_records(), _embedding_diagnostics(), _held_out_values(), _load_embedding_payload(), _molecules_and_labels(), Any, Mol, ndarray (+9 more)

### Community 18 - "update_manuscript_rev4.py"
Cohesion: 0.27
Nodes (17): build(), _dataset_counts_sentence(), find_paragraph(), _format_seen(), insert_after(), insert_exposure_table(), load_json(), parse_args() (+9 more)

### Community 19 - "model.py"
Cohesion: 0.11
Nodes (24): corrupt_graph_inputs(), CorruptedGraph, kl_divergence(), MolecularVGAE, nt_xent_loss(), Return invariance, variance-floor, and covariance-redundancy losses., Symmetric cross-view InfoNCE loss for graph embeddings., vicreg_terms() (+16 more)

### Community 20 - "NegativeCandidates"
Cohesion: 0.15
Nodes (18): assert_valid_candidates(), _edge_tensor(), NegativeCandidates, _pair_template(), ndarray, Tensor, Select hard-pool logits while retaining gradients for selected values., Return lexicographic upper-triangle pairs for a graph size. (+10 more)

### Community 21 - "build"
Cohesion: 0.28
Nodes (15): build(), find_paragraph(), fmt(), insert_after(), load_json(), metric_summary(), parse_args(), Any (+7 more)

### Community 22 - "checkpoint.py"
Cohesion: 0.39
Nodes (8): Optimizer, build_checkpoint(), capture_rng_state(), gather_rank_objects(), Any, Module, restore_rng_state(), validate_checkpoint()

### Community 23 - "audit_step"
Cohesion: 0.44
Nodes (11): add_check(), audit_step(), finite_number(), format_number(), main(), nested(), Any, Path (+3 more)

### Community 24 - "MolecularRepresentationModel"
Cohesion: 0.16
Nodes (21): build_smiles_encoder(), _calibrator_arrays(), compare_embedding_matrices(), optimized_representation_blocks(), OptimizedRawCore, packed_to_device(), device, Module (+13 more)

### Community 25 - "Manuscript rev3 audit artifacts"
Cohesion: 0.29
Nodes (6): Exact downstream/pretraining identity overlap, Exact seed-42 training exposure, Frozen 13-descriptor-only downstream control, Manuscript rev3 audit artifacts, Promotion chronology and terminology, Reproduction commands

### Community 26 - "Manuscript rev4: exact downstream-molecule exposure audit"
Cohesion: 0.29
Nodes (6): Aggregate pretraining exposure, Checkpoint-resolved downstream exposure, Exact reconstruction and validation, Files, Manuscript rev4: exact downstream-molecule exposure audit, Reproduction

### Community 27 - "Manuscript rev5: evidence-source reorganization"
Cohesion: 0.29
Nodes (6): Experimental chronology made explicit, Interpretation-affecting wording changes, Manuscript rev5: evidence-source reorganization, Old-to-new section mapping, Rebuild, Tables, figures and validation

### Community 28 - "Q: Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k., Source Nodes

### Community 29 - "Q: Create gmolai-rev5.docx from the current authoritative gmolai-rev4.docx."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Create gmolai-rev5.docx from the current authoritative gmolai-rev4.docx., Source Nodes

### Community 31 - "Descriptor Schema Configuration"
Cohesion: 0.67
Nodes (3): Descriptor Schema Configuration, Thirteen Molecular Descriptor Targets, Train-Split Descriptor Standardization

### Community 36 - "load_protocol"
Cohesion: 0.11
Nodes (55): main(), Any, ndarray, remap(), role_digest(), validate_split(), atomic_savez(), atomic_write_csv() (+47 more)

### Community 37 - "gmolai_retrain/fast_graph.py"
Cohesion: 0.25
Nodes (12): _category(), fast_featurize_molecule(), _hbond_factory(), pack_feature_arrays(), pack_molecules(), pack_smiles_task(), PackedBatch, Mol (+4 more)

### Community 38 - "speed_adapter.py"
Cohesion: 0.12
Nodes (43): atomic_write_json(), atomic_write_text(), load_json(), load_protocol(), protocol_digest(), Any, Path, read_panel_tsv() (+35 more)

### Community 39 - "gMolAI comparator and encoding-speed benchmark: feasibility memo"
Cohesion: 0.08
Nodes (23): 10. Primary sources consulted, 1.1 Locked internal pretraining test partition, 1.2 Current MoleculeNet development/promotion panel, 1. What can validly be compared in each partition?, 2. Recommended headline encoders, 3. Other possible tools and how to treat them, 4.1 Freeze the degrees of freedom before locked-test access, 4.2 Keep raw native representation dimensions (+15 more)

### Community 40 - "MoleculeNet development panel plus HIV: completed results"
Cohesion: 0.09
Nodes (19): Common-coverage rule, Evidence scope, Execution chronology and integrity, Frozen protocol: MoleculeNet development panel plus HIV confirmation, Frozen representations, Identical downstream probes, Molecular preparation and split inheritance, Timing scope (+11 more)

### Community 41 - "update_manuscript_rev6.py"
Cohesion: 0.21
Nodes (21): build(), find_exact_paragraph(), find_paragraph(), insert_parameter_table(), load_benchmark(), load_parameter_counts(), old_text_is_subsequence(), omml_hashes() (+13 more)

### Community 42 - "data.py"
Cohesion: 0.16
Nodes (22): Data, _balanced_allocation(), finite_batches(), _finite_shard_plan(), _finite_shard_window_plan(), graph_from_shard(), load_graph_manifest(), _load_shard() (+14 more)

### Community 43 - "test_fast_inference.py"
Cohesion: 0.42
Nodes (9): molecules(), Mol, small_representation_model(), test_direct_packing_matches_pyg_batch_exactly(), test_downstream_consumer_uses_equivalent_optimized_blocks(), test_fast_features_are_exact_on_curated_chemistry(), test_optimized_raw_core_matches_authoritative_model_encode(), test_optimized_smiles_encoder_preserves_reference_values_and_order() (+1 more)

### Community 44 - "Locked internal test-partition encoder benchmark: completed results"
Cohesion: 0.18
Nodes (9): Audit status, Launch on Arrhenius, Locked internal test-partition encoder benchmark, Bounded interpretation, Common representation diagnostics, Execution and integrity, Locked internal test-partition encoder benchmark: completed results, Realized common coverage (+1 more)

### Community 45 - "Q: Explore and fully understand this repository. Expanded query tokens: architecture, pipeline, data, model, training, inference, evaluation, validation, checkpoints, reproducibility, promotion, audit."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Explore and fully understand this repository. Expanded query tokens: architecture, pipeline, data, model, training, inference, evaluation, validation, checkpoints, reproducibility, promotion, audit., Source Nodes

### Community 46 - "Strong VICReg Variance-Covariance Objective"
Cohesion: 0.18
Nodes (11): Representation 128d Configuration, 128-Dimensional Graph Latent Space, Representation Strong 128d Configuration, Strong Variance-Covariance Objective, Representation Strong VICReg Configuration, Strong VICReg Variance-Covariance Objective, Representation Collapse Prevention, Masked Graph Reconstruction Objective (+3 more)

### Community 47 - "Q: Complete the optimized single-GPU speed benchmark for gMolAI, Morgan, and competitor models at batch sizes 64, 128, 256, and 512 with the qualified worker count, then replace old speed artifacts and update documentation."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Complete the optimized single-GPU speed benchmark for gMolAI, Morgan, and competitor models at batch sizes 64, 128, 256, and 512 with the qualified worker count, then replace old speed artifacts and update documentation., Source Nodes

### Community 48 - "Combined ZINC-PubChem Dataset"
Cohesion: 0.32
Nodes (8): Bemis-Murcko Scaffold Hash Split, Canonical Isomeric SMILES Deduplication, Combined ZINC-PubChem Dataset, 8,192-Graph Sharding, Molecular Canonicalization Policy, PubChem Molecular Source, Explicit Versioned Molecular Feature Schema, ZINC Molecular Source

### Community 49 - "Frozen protocol: locked internal test-partition benchmark"
Cohesion: 0.25
Nodes (7): Common diagnostics, Frozen comparator panel, Frozen protocol: locked internal test-partition benchmark, Input and coverage policy, Prohibitions, Scientific scope, Timing scope in this initiated run

### Community 50 - "Combined ZINC-PubChem Retraining Configuration"
Cohesion: 0.53
Nodes (6): Node-Budgeted 500,000-Step Retraining Schedule, Combined ZINC-PubChem Retraining Configuration, Thirteen-Column Descriptor Schema, Four-Layer GINE Latent Model, KL Warmup Regularization, Masked Graph and Descriptor Pretraining Objective

### Community 51 - "Frozen speed-benchmark protocol"
Cohesion: 0.20
Nodes (9): Common conditions, Frozen speed-benchmark protocol, Integrity gates, KERMT v2 limitation, Optimized gMolAI path, Reporting limits, Status and scope, Superseded execution (+1 more)

### Community 52 - "Raw Molecule Data Flow"
Cohesion: 0.60
Nodes (6): Deterministic V5 Molecular Encoder, gMolAI Retraining Pipeline, Promoted gMolAI 384-Dimensional Molecular Vector, Raw Molecule Data Flow, Semantic Promotion Suite, Verified ZINC and PubChem Source Ingestion

### Community 53 - "gMolAI inference speed optimization"
Cohesion: 0.33
Nodes (5): Files, gMolAI inference speed optimization, Outcome, Production use, Scientific equivalence

### Community 54 - "KERMT v2 native batch-dependence audit"
Cohesion: 0.40
Nodes (4): Benchmark treatment, Finding, KERMT v2 native batch-dependence audit, Root cause

### Community 55 - "Q: Audit the completed MoleculeNet plus HIV benchmark implementation and artifact flow"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Audit the completed MoleculeNet plus HIV benchmark implementation and artifact flow, Source Nodes

### Community 56 - "Q: Calculate the embedding time for each model and Morgan from the MoleculeNet plus HIV benchmark"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Calculate the embedding time for each model and Morgan from the MoleculeNet plus HIV benchmark, Source Nodes

### Community 57 - "Q: done"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: done, Source Nodes

### Community 58 - "Q: gmolai is the smallest model but it is much slower compare to others. why is it?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: gmolai is the smallest model but it is much slower compare to others. why is it?, Source Nodes

### Community 59 - "Common locked-test encoding-speed benchmark"
Cohesion: 0.33
Nodes (5): Artifacts, Common locked-test encoding-speed benchmark, Completed execution, Reproduce on Arrhenius, Timing and integrity

### Community 64 - "Q: is the use face folder 'inference' current? I mean is it using the speed-optimized pipeline instead of old slow one?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: is the use face folder 'inference' current? I mean is it using the speed-optimized pipeline instead of old slow one?, Source Nodes

### Community 66 - "Graph-256 and Node-128 Latent Spaces"
Cohesion: 0.67
Nodes (3): Graph-256 and Node-128 Latent Spaces, Masked Graph VICReg Architecture, Standardized Graph-256 Plus Weighted Mean-Node-128 Vector Definition

## Knowledge Gaps
- **149 isolated node(s):** `gmolai-retrain`, `common.sh script`, `run_benchmark_in_container.sh script`, `submit_pipeline.sh script`, `run_encode_example.sh script` (+144 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `Semantic Promotion Suite` (3× useful, score=2.867628143)
- `Manuscript rev4: exact downstream-molecule exposure audit` (2× useful, score=1.942275722)
- `Promotion Integrity Gates` (2× useful, score=1.932288344)
- `Validation Evidence Bundle` (2× useful, score=1.932288344)

**Known dead ends** — questions that led nowhere; don't re-derive.
- "done" -> `benchmark_io.py`, `finalize.py`, `adapter.py`

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_representation_probes()` connect `run_representation_probes` to `adapter.py`, `RuntimeError`, `ValueError`, `cli.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Why does `MolecularRepresentationModel` connect `MolecularRepresentationModel` to `RuntimeError`, `InfiniteGraphBatchIterator`, `train.py`, `ValueError`, `test_fast_inference.py`, `generate_embeddings.py`, `OptimizedSmilesEncoder`, `Tensor`, `model.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `main()` connect `adapter.py` to `RuntimeError`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Are the 127 inferred relationships involving `RuntimeError` (e.g. with `main()` and `validate_split()`) actually correct?**
  _`RuntimeError` has 127 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `ValueError` (e.g. with `main()` and `run_encode()`) actually correct?**
  _`ValueError` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `MolecularRepresentationModel` (e.g. with `InferenceError` and `ModelBundle`) actually correct?**
  _`MolecularRepresentationModel` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `train()` (e.g. with `RuntimeError` and `_request_stop()`) actually correct?**
  _`train()` has 2 INFERRED edges - model-reasoned connections that need verification._