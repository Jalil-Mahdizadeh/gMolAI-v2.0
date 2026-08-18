# Graph Report - gMolAI-retrain  (2026-08-18)

## Corpus Check
- 595 files · ~2,183,322 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3295 nodes · 7958 edges · 297 communities (268 shown, 29 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 828 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `45d0d51c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- config.py
- adapter.py
- downstream_audit.py
- Scientific Audit and Repair
- cli.py
- Representation V1 Training Overlay
- Checkpoint 10k vs 15k Promotion Review
- data.py
- train.py
- Self-Contained Inference Entry Point
- ValueError
- Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)
- gmolai.py
- update_manuscript_rev5.py
- study_common.py
- MolecularRepresentationModel
- tune.py
- sha256_file
- update_manuscript_rev4.py
- feature_schema
- sample_per_graph_negatives
- build
- downstream.py
- audit_step
- fast_inference.py
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
- scaled_common.py
- speed_adapter.py
- gMolAI comparator and encoding-speed benchmark: feasibility memo
- MoleculeNet development panel plus HIV: completed results
- update_manuscript_rev6.py
- verify.py
- evaluate_panel.py
- Locked internal test-partition encoder benchmark: completed results
- Q: Explore and fully understand this repository. Expanded query tokens: architecture, pipeline, data, model, training, inference, evaluation, validation, checkpoints, reproducibility, promotion, audit.
- Strong VICReg Variance-Covariance Objective
- Q: Complete the optimized single-GPU speed benchmark for gMolAI, Morgan, and competitor models at batch sizes 64, 128, 256, and 512 with the qualified worker count, then replace old speed artifacts and update documentation.
- Combined ZINC-PubChem Retraining Configuration
- Frozen protocol: locked internal test-partition benchmark
- main
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
- day1_study.py
- run_encode_example.sh
- run_final.sh
- Q: is the use face folder 'inference' current? I mean is it using the speed-optimized pipeline instead of old slow one?
- extra-benchmark/README.md
- deriv-gen/README.md
- files
- files
- main
- update_manuscript_rev9.py
- Frozen generation baseline v1
- Step 2b results
- RuntimeError
- load_json
- atomic_write_json
- Q: Which existing decoder and sampling strategy define the frozen deriv-gen generation baseline?
- sha256_file
- step-02c-chemical-characterization/inputs/manifest.json
- analyze_phase.py
- external_step2b_common_source
- Final protocol: latent geometry and derivative retrieval
- Frozen protocol: scaled MMP mining and latent control-space selection
- Frozen protocol: decoder feasibility from released gMolAI vectors
- step-02d-generation-scaling/inputs/manifest.json
- generate_deriv_gen_manuscript_figures.py
- test_representations.py
- files
- main
- step-02b-candidate-reranking/inputs/manifest.json
- files
- Step 2c results: chemical characterization of frozen candidate sets
- step-02-decoder-feasibility/inputs/manifest.json
- Frozen protocol: Step 2d candidate-generation scaling
- Frozen protocol: MoleculeNet development panel plus HIV confirmation
- Manuscript Accuracy Audit
- Full Promotion Evaluation
- files
- files
- Frozen protocol: Step 2c chemical characterization
- Step 2d post-completion synthetic-accessibility extension
- Step 2b protocol: frozen-decoder candidate search and latent reranking
- Step 2d results: frozen decoder candidate scaling
- Publication-Quality Training Plots
- Retained Checkpoint Availability Audit
- Derivative-generation checkpoint
- Protocol amendment 01: repeated-transformation support
- step-01b-scaled-space-selection/scripts/run_study.sh
- step-02-decoder-feasibility/scripts/run_study.sh
- Step 2d results: frozen decoder candidate scaling
- step-02d-generation-scaling/scripts/run_study.sh
- Q: Update gmolai-rev6.docx to rev7 using the completed test-partition, MoleculeNet plus HIV, and speed benchmarks; stage the speed plots in the manuscript figures directory, embed no figures, and leave rev6 untouched.
- Q: The root extra-benchmark/README.md still says that moleculenet has not been executed while the MoleculeNet run is complete, and moleculenet/RESULTS.md says a controlled throughput benchmark is still needed even though speed is complete. Why were these missed, and can I rely on the documentation review?
- Q: the results are very promising and I accept your decission. Proceed to the next step only: train and validate a molecular decoder conditioned on the frozen released gMolAI representation.
- Q: Step 2b frozen-decoder candidate generation and frozen-gMolAI latent reranking on a fresh validation panel
- Q: Move to Step 2c: chemical characterization of the frozen Step-2b 50-candidate correct-condition sets using the Step-1b one-cut MMP rules, without training, regeneration, or latent perturbation.
- Q: I do not see the valid plot (blue) in quality_locality_diversity_scaling.png
- Q: if feasible and easy do synthetic accessibility as part of Step 2d and create proper plots; then checkpoint for the next agent
- Step 1b: scaled latent-space selection
- external_chemistry_policy
- optimized_inference
- external_step1b_protocol
- external_step2b_candidates
- TDC ADMET frozen-representation benchmark: completed results
- external_step2b_complete
- external_step2b_evaluation_source
- external_step2b_final_seal
- external_step2b_generation_stats
- external_step2b_manifest
- external_step2b_panel
- external_step2b_policy_seal
- external_step2b_verification
- external_validation_molecules
- source_audit
- source_audit_core
- source_common
- source_component_tests
- main
- source_registration
- source_report
- source_runner
- source_verify
- step-02c-chemical-characterization/scripts/run_study.sh
- moleculenet/README.md
- Q: Step 2d frozen decoder candidate-generation scaling: what strategy and budget are supported?
- Step 1: latent geometry and derivative retrieval
- base_config
- atomic_write_json
- checkpoint
- chemistry_policy
- container
- inference_entrypoint
- model_definition
- optimized_inference
- packaged_calibrator
- packaged_checkpoint
- representation_selection
- step1b_complete
- step1b_decision
- train_molecules
- train_raw_embeddings
- validation_embeddings
- validation_molecules
- Step 2: frozen-representation decoder feasibility
- chemistry_policy
- decoder_checkpoint
- decoder_inference_export
- gmolai_calibrator
- gmolai_model_definition
- gmolai_resolved_config
- inference_entrypoint
- Scaled latent-space selection results
- sha256_file
- packaged_checkpoint
- packaged_resolved_config
- representation_selection
- step2_common_source
- step2_complete
- step2_decoder_model_source
- step2_manifest
- step2_original_final_panel
- step2_split_indices
- step2_training_complete
- train_molecules
- train_raw_embeddings
- validation_embeddings
- validation_molecules
- metrics_common.py
- Frozen scientific protocol
- container
- Frozen benchmark protocol
- decoder_inference_export
- gmolai_calibrator
- gmolai_checkpoint
- gmolai_resolved_config
- inference_entrypoint
- optimized_inference
- packaged_calibrator
- packaged_checkpoint
- packaged_resolved_config
- representation_selection
- step1b_complete
- step1b_fragment_source
- step1b_protocol
- step2_common_source
- step2_complete
- step2_decoder_model_source
- step2_manifest
- step2_original_final_panel
- step2_split_indices
- step2_training_complete
- step2b_common_source_frozen
- step2b_development_panel
- step2b_final_panel
- step2b_protocol
- step2b_verification
- step2c_audit_core_source
- step2c_complete
- step2c_protocol
- step2c_verification
- train_molecules
- train_raw_embeddings
- validation_embeddings
- validation_molecules
- Step 2d: frozen decoder generation scaling
- Shared contracts
- run_day1.sh
- step-02-decoder-feasibility/DESIGN.md
- step-02b-candidate-reranking/DESIGN.md
- step-02b-candidate-reranking/README.md
- step-02b-candidate-reranking/scripts/run_study.sh
- Day-1 results: latent geometry and derivative retrieval
- step-02c-chemical-characterization/DESIGN.md
- step-02c-chemical-characterization/README.md
- step-02d-generation-scaling/DESIGN.md
- sa_extension_base_DECISION.md
- sa_extension_base_README.md
- run_sa_extension.sh
- checkpoint.py
- tdc-admet/RESULTS.md
- Decoder feasibility results
- gMolAI derivative generation studies
- Latent control-space decision
- Frozen strategic direction
- Q: explore the current repo and fully understand it. you can use graphify skill if that helps.
- Q: Update the deriv-gen project direction based on the completed evidence.
- Q: generate publication quality radar plots for MoleculeNet plus HIV and TDC ADMET
- Q: update the manuscript (both 'Materials and Methods' and 'Results and Discussion' when applicable) from rev7 to rev8 (the complete extra-benchmark added) and from rev8 to rev9 (step-01b, step-02, step-02b, step-02c, step-02d added). Notes: 1-keep older rev1-7 on the disk. 2-do not include the figures in the docx files. 3-create relevant and well-kept ordered headings/sub-headings. 4-use the same math and formula editing tool if there need to add math/formula. 5-the text must be scientific style and clear. 7-for rev9, feel free to create images/plots from the main results of the step-01b, step-02, step-02b, step-02c, step-02d. Do your best.
- external_step1b_fragment_source
- gmolai_model_definition
- run_representation_probes
- Frozen minimal clustering benchmark protocol
- read_panel_tsv
- main
- collect
- QMugs: Quantum Mechanical Properties of Drug-like Molecules
- Q: There is a folder in the root named 'inference' which is the user-faced SMILES -> gMolAI embeddings hydride x3. I want to re-structure it so that: it takes a list of SMILES and generates the gMolAI release hybride x3 embedings in .npz or .npy format (you decide whic one). I also want some nice CLI flags for when user want to generate candidates SMILES from the corresponding embeddings, each in a separate CSV files (a single CSV for each seed molecule). The candidates must be valid and unique canonical SMILES with a column for Morgan/Tanimoto similarity values. Run the example_smiles.csv both embeddings and 1000 derivatives for each smiles. Ensure that the this user-faced worflow uses the correct ckeckpoints, sampling strategy, etc. rename '/inference/model/' to '/inference/models/' and place the correct decoder checkpoint there and validate all check points and calibrator. Update the README.md accordingly, particularly explain the CLI flags for both encode (embeddings generation) and decoder (candidate generation from embeddings). Do it professionaly and user friendly.
- Q: Review three findings: missing PyTorch and PyTorch Geometric package dependencies, missing GitHub Actions CI, and collection-sized memory accumulation in the public encoder; address only valid findings.
- external_resolved_config
- resolved_config
- test_downstream_audit.py
- Frozen protocol: Step 2b-style analysis of Step 2d candidates
- extra-step-02b-style/inputs/manifest.json
- reencoding
- atomic_write_csv
- assemble_qmugs
- main
- immutable_inputs
- Q: explore the repo and fully understanf it. Then I ask for some questions.
- analysis_config
- analysis_protocol
- candidate_characterization
- conditions
- encoder_calibrator
- encoder_checkpoint
- encoder_cli
- encoder_config
- encoder_input
- encoder_model_source
- encoder_selection
- optimized_inference_source
- panel
- seed_budget_metrics
- step2d_final_analysis_summary
- step2d_verification
- unique_molecules
- Step 2b-style search and latent-reranking analysis of Step 2d
- run_analysis.sh
- sitecustomize.py
- run_benchmark.sh
- container
- decoder_checkpoint
- main
- main
- extra-step-02b-style/README.md
- clustering/README.md
- scripts/__init__.py

## God Nodes (most connected - your core abstractions)
1. `device` - 48 edges
2. `MolecularRepresentationModel` - 42 edges
3. `files` - 40 edges
4. `train()` - 36 edges
5. `load_json()` - 35 edges
6. `main()` - 34 edges
7. `sha256_file()` - 34 edges
8. `main()` - 33 edges
9. `atomic_write_json()` - 33 edges
10. `atomic_write_json()` - 33 edges

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

## Communities (297 total, 29 thin omitted)

### Community 0 - "config.py"
Cohesion: 0.20
Nodes (25): apply_training_plan(), canonical_json(), ConfigurationError, _deep_update(), descriptor_names(), _expand(), load_config(), load_yaml() (+17 more)

### Community 1 - "adapter.py"
Cohesion: 0.07
Nodes (78): Exception, load_kermt(), load_molai(), load_molclr(), load_molformer(), load_morgan(), load_smi_ted(), main() (+70 more)

### Community 2 - "downstream_audit.py"
Cohesion: 0.12
Nodes (45): join_corpus(), main(), audit_pretraining_overlap(), benchmark_descriptor_control(), _dataset_source(), _descriptor_matrix(), _groups_digest(), _identity_digest() (+37 more)

### Community 3 - "Scientific Audit and Repair"
Cohesion: 0.06
Nodes (57): Atomic Exact Resume Contract, Audit Disposition and Retraining Corrections, Molecular Encoder Capability Boundary, Exact-Denominator Distributed Validation, Explicit 13-Descriptor Contract, Fail-Closed Representation Promotion, Numerical GPU Reproducibility, Per-Graph Negative Sampling (+49 more)

### Community 4 - "cli.py"
Cohesion: 0.20
Nodes (32): _apply_run_directory(), _apply_training_budgets(), build_parser(), command_audit_downstream_exposure(), command_audit_downstream_overlap(), command_audit_training_exposure(), command_benchmark_descriptor_control(), command_benchmark_downstream() (+24 more)

### Community 5 - "Representation V1 Training Overlay"
Cohesion: 0.06
Nodes (34): Representation Contrastive 0.02 Configuration, NT-Xent Weight 0.02, Representation Contrastive 0.05 Configuration, NT-Xent Weight 0.05, Representation Contrastive 0.10 Configuration, NT-Xent Weight 0.10, Representation Contrastive 0.20 Configuration, NT-Xent Weight 0.20 (+26 more)

### Community 6 - "Checkpoint 10k vs 15k Promotion Review"
Cohesion: 0.19
Nodes (13): _check_gate(), Checkpoint 10k vs 15k Promotion Review, Fixed Promotion Threshold Wording, FreeSolv 1.30 RMSE Gate, Promotion Integrity Gates, Reviewer-Proof Follow-Up Evaluation, Statistical Evidence Limit, Step 10k Operational Selection (+5 more)

### Community 7 - "data.py"
Cohesion: 0.09
Nodes (30): Batch, Data, Future, _balanced_allocation(), finite_batches(), _finite_shard_plan(), _finite_shard_window_plan(), graph_from_shard() (+22 more)

### Community 8 - "train.py"
Cohesion: 0.12
Nodes (45): device, DistributedDataParallel, validate_checkpoint(), grouped_feature_loss(), Use categorical CE for one-hot groups and BCE for binary features., NegativeCandidates, Select hard-pool logits while retaining gradients for selected values., select_hard_negative_logits() (+37 more)

### Community 9 - "Self-Contained Inference Entry Point"
Cohesion: 0.27
Nodes (11): CSV Molecular Embedding Inference, Embedding Output Contract, Checkpoint and Calibrator Model Asset Bundle, SHA-256 and Model Identity Validation, Training-Time Molecule Acceptance Policy, Hardware-Conditional Numerical Determinism Contract, Promoted gMolAI 384-Dimensional Molecular Vector, Embedding Provenance Metadata Sidecar (+3 more)

### Community 10 - "ValueError"
Cohesion: 0.21
Nodes (28): load_gmolai(), atomic_copy(), atomic_torch_save(), Path, _automatic_checkpoint_name(), _automatic_representation_calibrator(), _calibrator_expected_identity(), _check_gate() (+20 more)

### Community 11 - "Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)"
Cohesion: 0.15
Nodes (29): Pilot Contrastive 0.005 Configuration, Low-Contrastive Pilot, Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers), Pilot Projector Contrastive Configuration (weight 0.01), Pilot Projector Contrastive Configuration (weight 0.01, descriptor weight 0.50, 15k steps), Projector-Space Contrastive Masked-Graph Objective with VICReg Terms Disabled, Pilot Projector Contrastive Configuration (weight 0.01, seed 43), Pilot Projector Contrastive Configuration (weight 0.02) (+21 more)

### Community 12 - "gmolai.py"
Cohesion: 0.05
Nodes (95): Connection, beam_order(), ConditionalSmilesTransformer, decode_tokens(), decoder_parameter_count(), generate_beam_pool(), generate_seeded_sample_pool(), proportional_merge() (+87 more)

### Community 13 - "update_manuscript_rev5.py"
Cohesion: 0.22
Nodes (21): assert_source_contract(), build(), delete_paragraph(), find_exact_paragraph(), find_paragraph(), insert_evidence_roles_table(), new_paragraph_before(), next_table_element() (+13 more)

### Community 14 - "study_common.py"
Cohesion: 0.05
Nodes (100): atomic_copy(), epoch_row(), main(), DataFrame, Path, Series, utc_now(), ConditionalSmilesTransformer (+92 more)

### Community 15 - "MolecularRepresentationModel"
Cohesion: 0.08
Nodes (25): corrupt_graph_inputs(), CorruptedGraph, DeterministicGINEEncoder, GraphConditionedEdgeDecoder, kl_divergence(), MolecularRepresentationModel, MolecularVGAE, nt_xent_loss() (+17 more)

### Community 16 - "tune.py"
Cohesion: 0.09
Nodes (57): ProcessPoolExecutor, main(), main(), Path, sha256_file(), _category(), fast_featurize_molecule(), _hbond_factory() (+49 more)

### Community 17 - "sha256_file"
Cohesion: 0.08
Nodes (75): main(), metric_defined(), ndarray, remap(), split_key(), main(), pearson(), Path (+67 more)

### Community 18 - "update_manuscript_rev4.py"
Cohesion: 0.27
Nodes (17): build(), _dataset_counts_sentence(), find_paragraph(), _format_seen(), insert_after(), insert_exposure_table(), load_json(), parse_args() (+9 more)

### Community 19 - "feature_schema"
Cohesion: 0.13
Nodes (23): categorical_width(), dimensions(), feature_schema(), FeatureDimensions, Tensor, The descriptor head must see the same representation in train/eval paths., _representation_model_and_batch(), test_complete_corrected_objective_backpropagates() (+15 more)

### Community 20 - "sample_per_graph_negatives"
Cohesion: 0.20
Nodes (14): assert_valid_candidates(), _edge_tensor(), _pair_template(), ndarray, Tensor, Return lexicographic upper-triangle pairs for a graph size., Sample unique undirected negatives independently inside each graph. Easy…, sample_per_graph_negatives() (+6 more)

### Community 21 - "build"
Cohesion: 0.28
Nodes (15): build(), find_paragraph(), fmt(), insert_after(), load_json(), metric_summary(), parse_args(), Any (+7 more)

### Community 22 - "downstream.py"
Cohesion: 0.17
Nodes (25): benchmark_moleculenet(), _classification_probe(), _encode_molecules(), _inner_group_folds(), _morgan_features(), _prepare_dataset(), _prepare_dataset_records(), Any (+17 more)

### Community 23 - "audit_step"
Cohesion: 0.44
Nodes (11): add_check(), audit_step(), finite_number(), format_number(), main(), nested(), Any, Path (+3 more)

### Community 24 - "fast_inference.py"
Cohesion: 0.07
Nodes (47): _category(), fast_featurize_molecule(), _hbond_factory(), initialize_worker(), pack_feature_arrays(), pack_molecules(), pack_smiles_task(), PackedBatch (+39 more)

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
Nodes (54): main(), Any, ndarray, remap(), role_digest(), validate_split(), atomic_savez(), atomic_write_csv() (+46 more)

### Community 37 - "scaled_common.py"
Cohesion: 0.08
Nodes (79): add_observation_ids(), assign_mismatched_transforms(), average_replicates(), by_transformation(), evaluate_alignment(), fit_directions(), hierarchical_bootstrap(), Any (+71 more)

### Community 38 - "speed_adapter.py"
Cohesion: 0.12
Nodes (42): atomic_write_json(), atomic_write_text(), load_json(), load_protocol(), protocol_digest(), Any, Path, read_panel_tsv() (+34 more)

### Community 39 - "gMolAI comparator and encoding-speed benchmark: feasibility memo"
Cohesion: 0.08
Nodes (23): 10. Primary sources consulted, 1.1 Locked internal pretraining test partition, 1.2 Current MoleculeNet development/promotion panel, 1. What can validly be compared in each partition?, 2. Recommended headline encoders, 3. Other possible tools and how to treat them, 4.1 Freeze the degrees of freedom before locked-test access, 4.2 Keep raw native representation dimensions (+15 more)

### Community 40 - "MoleculeNet development panel plus HIV: completed results"
Cohesion: 0.22
Nodes (9): Bounded conclusions, Completion and integrity, Descriptor-only context, Development-panel interpretation, HIV post-selection confirmation, MoleculeNet development panel plus HIV: completed results, Observed representation-export timings, Primary paired endpoint results (+1 more)

### Community 41 - "update_manuscript_rev6.py"
Cohesion: 0.21
Nodes (21): build(), find_exact_paragraph(), find_paragraph(), insert_parameter_table(), load_benchmark(), load_parameter_counts(), old_text_is_subsequence(), omml_hashes() (+13 more)

### Community 42 - "verify.py"
Cohesion: 0.53
Nodes (9): find_named(), load_json(), main(), Any, Path, require(), resolve_bound_path(), sha256_file() (+1 more)

### Community 43 - "evaluate_panel.py"
Cohesion: 0.08
Nodes (66): generate_beam_pool(), generate_sample_pool(), ConditionalSmilesTransformer, dtype, inference_mode, Tensor, Inference-only candidate generators for the frozen Step-2 decoder., Return fixed-seed samples and their untempered decoder scores. (+58 more)

### Community 44 - "Locked internal test-partition encoder benchmark: completed results"
Cohesion: 0.33
Nodes (6): Bounded interpretation, Common representation diagnostics, Execution and integrity, Locked internal test-partition encoder benchmark: completed results, Realized common coverage, Versioned audit artifacts

### Community 45 - "Q: Explore and fully understand this repository. Expanded query tokens: architecture, pipeline, data, model, training, inference, evaluation, validation, checkpoints, reproducibility, promotion, audit."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Explore and fully understand this repository. Expanded query tokens: architecture, pipeline, data, model, training, inference, evaluation, validation, checkpoints, reproducibility, promotion, audit., Source Nodes

### Community 46 - "Strong VICReg Variance-Covariance Objective"
Cohesion: 0.18
Nodes (11): Representation 128d Configuration, 128-Dimensional Graph Latent Space, Representation Strong 128d Configuration, Strong Variance-Covariance Objective, Representation Strong VICReg Configuration, Strong VICReg Variance-Covariance Objective, Representation Collapse Prevention, Masked Graph Reconstruction Objective (+3 more)

### Community 47 - "Q: Complete the optimized single-GPU speed benchmark for gMolAI, Morgan, and competitor models at batch sizes 64, 128, 256, and 512 with the qualified worker count, then replace old speed artifacts and update documentation."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Complete the optimized single-GPU speed benchmark for gMolAI, Morgan, and competitor models at batch sizes 64, 128, 256, and 512 with the qualified worker count, then replace old speed artifacts and update documentation., Source Nodes

### Community 48 - "Combined ZINC-PubChem Retraining Configuration"
Cohesion: 0.21
Nodes (14): Bemis-Murcko Scaffold Hash Split, Node-Budgeted 500,000-Step Retraining Schedule, Canonical Isomeric SMILES Deduplication, Combined ZINC-PubChem Dataset, Combined ZINC-PubChem Retraining Configuration, Thirteen-Column Descriptor Schema, Four-Layer GINE Latent Model, 8,192-Graph Sharding (+6 more)

### Community 49 - "Frozen protocol: locked internal test-partition benchmark"
Cohesion: 0.22
Nodes (7): Common diagnostics, Frozen comparator panel, Frozen protocol: locked internal test-partition benchmark, Input and coverage policy, Prohibitions, Scientific scope, Timing scope and subsequent controlled benchmark

### Community 50 - "main"
Cohesion: 0.09
Nodes (64): assert_equal_series(), element_delta(), main(), Any, DataFrame, ndarray, Series, scalar_metric_rows() (+56 more)

### Community 51 - "Frozen speed-benchmark protocol"
Cohesion: 0.22
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

### Community 60 - "speed/RESULTS.md"
Cohesion: 0.32
Nodes (4): Encoding-speed benchmark results, Audit status, Launch on Arrhenius, Locked internal test-partition encoder benchmark

### Community 61 - "day1_study.py"
Cohesion: 0.11
Nodes (60): atomic_save_npz(), atomic_write_csv(), atomic_write_json(), atomic_write_parquet(), atomic_write_text(), bootstrap_mean_ci(), build_mmp_pairs(), core_sets() (+52 more)

### Community 64 - "Q: is the use face folder 'inference' current? I mean is it using the speed-optimized pipeline instead of old slow one?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: is the use face folder 'inference' current? I mean is it using the speed-optimized pipeline instead of old slow one?, Source Nodes

### Community 66 - "deriv-gen/README.md"
Cohesion: 0.17
Nodes (6): Decoder decision, NO-GO, Step 2b decision, Step 2c bounded decision, Post-completion SA-score context, Step 2d decision

### Community 67 - "files"
Cohesion: 0.05
Nodes (40): path, sha256, path, sha256, path, sha256, path, sha256 (+32 more)

### Community 68 - "files"
Cohesion: 0.06
Nodes (32): path, sha256, path, sha256, path, sha256, path, sha256 (+24 more)

### Community 69 - "main"
Cohesion: 0.11
Nodes (29): configure_determinism(), ensure_inside(), load_decoder(), ConditionalSmilesTransformer, Path, Shared integrity and I/O utilities for the frozen Step 2d study., require_one_gpu(), resolve_manifest_inputs() (+21 more)

### Community 70 - "update_manuscript_rev9.py"
Cohesion: 0.08
Nodes (71): assert_archive_has_no_images(), build(), endpoint_table_matrix(), find_exact_paragraph(), find_paragraph(), insert_table_before(), load_endpoint_results(), load_speed_results() (+63 more)

### Community 71 - "Frozen generation baseline v1"
Cohesion: 0.40
Nodes (4): Change control, Frozen generation baseline v1, Verification, What is frozen

### Community 72 - "Step 2b results"
Cohesion: 0.40
Nodes (4): Condition-use controls, Interpretation, Outcome, Step 2b results

### Community 73 - "RuntimeError"
Cohesion: 0.26
Nodes (25): analyze_extension(), _append_section(), assert_base_protected(), assert_source_hashes(), _bootstrap_seed_macro(), _chunks(), component_test(), extension_config() (+17 more)

### Community 74 - "load_json"
Cohesion: 0.14
Nodes (27): atomic_write_json(), main(), main(), main(), main(), deterministic_subset(), load_json(), protocol() (+19 more)

### Community 75 - "atomic_write_json"
Cohesion: 0.10
Nodes (42): Schema, canonicalize(), CanonicalMolecule, _feature_factory(), featurize_molecule(), _hydrogen_bond_flags(), _one_hot(), _position_encoding() (+34 more)

### Community 76 - "Q: Which existing decoder and sampling strategy define the frozen deriv-gen generation baseline?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Which existing decoder and sampling strategy define the frozen deriv-gen generation baseline?, Source Nodes

### Community 77 - "sha256_file"
Cohesion: 0.13
Nodes (43): Axes, atomic_write_csv(), atomic_write_parquet(), atomic_write_text(), ensure_inside(), input_paths(), load_config(), load_json() (+35 more)

### Community 78 - "step-02c-chemical-characterization/inputs/manifest.json"
Cohesion: 0.12
Nodes (15): candidate_regeneration, endpoint_labels_used, latent_perturbation, locked_test_rows, mmp_directed_generation, model_execution, registered_at, schema_version (+7 more)

### Community 79 - "analyze_phase.py"
Cohesion: 0.17
Nodes (24): aggregate_tables(), audit_and_extract(), characterize_pairs(), main(), phase_strategies(), proposal_view(), Any, DataFrame (+16 more)

### Community 80 - "external_step2b_common_source"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_common_source

### Community 81 - "Final protocol: latent geometry and derivative retrieval"
Cohesion: 0.18
Nodes (10): Analyses, Coordinate spaces, Data separation, Final protocol: latent geometry and derivative retrieval, Perturbation and retrieval controls, Predeclared feasibility gates, Prospective support amendment, Question (+2 more)

### Community 82 - "Frozen protocol: scaled MMP mining and latent control-space selection"
Cohesion: 0.18
Nodes (10): Candidate spaces, Direction and null controls, Frozen control-space decision rule, Frozen protocol: scaled MMP mining and latent control-space selection, Metrics, Populations and separation, Required decisions, Scalable matched-molecular-pair mining (+2 more)

### Community 83 - "Frozen protocol: decoder feasibility from released gMolAI vectors"
Cohesion: 0.18
Nodes (10): Boundary, Decoder, Development-only decode selection, Development-only duration safeguard, Final controls, Frozen GO / NO-GO rule, Frozen protocol: decoder feasibility from released gMolAI vectors, Metrics (+2 more)

### Community 84 - "step-02d-generation-scaling/inputs/manifest.json"
Cohesion: 0.18
Nodes (10): embedding_space, forbidden_inputs, locked internal test partition, MoleculeNet or HIV endpoint labels, policy, schema_version, study_id, latent perturbations (+2 more)

### Community 85 - "generate_deriv_gen_manuscript_figures.py"
Cohesion: 0.40
Nodes (16): configure_style(), figure10(), figure7(), figure8(), figure9(), finish_axis(), main(), panel_label() (+8 more)

### Community 86 - "test_representations.py"
Cohesion: 0.17
Nodes (4): Corrected and reproducible gMolAI retraining pipeline., test_downstream_cli_accepts_calibrated_embedding_definition(), test_embed_cli_accepts_independent_sampling_seed(), test_promotion_requires_identical_checkpoint_and_calibrator()

### Community 87 - "files"
Cohesion: 0.20
Nodes (10): path, sha256, files, chemistry_policy, step2b_candidate_source, step2b_complete, path, sha256 (+2 more)

### Community 88 - "main"
Cohesion: 0.16
Nodes (26): atomic_write_json(), atomic_write_text(), load_json(), Any, Path, ratio_bootstrap(), sha256_file(), utc_now() (+18 more)

### Community 89 - "step-02b-candidate-reranking/inputs/manifest.json"
Cohesion: 0.22
Nodes (8): embedding_space, forbidden_inputs, locked internal test partition, MoleculeNet or HIV endpoint labels, policy, schema_version, study_id, target structural quantities as ranking features

### Community 90 - "files"
Cohesion: 0.22
Nodes (9): path, sha256, size_bytes, files, external_step2b_protocol, source_protocol, path, sha256 (+1 more)

### Community 91 - "Step 2c results: chemical characterization of frozen candidate sets"
Cohesion: 0.22
Nodes (8): Artifact map, Bounded conclusion, Denominators, validity, and uniqueness, One-cut MMP derivatives, Primary chemical classification, Scaffold preservation and chemical proximity, Step 2c results: chemical characterization of frozen candidate sets, Within-set diversity and non-MMP graph changes

### Community 92 - "step-02-decoder-feasibility/inputs/manifest.json"
Cohesion: 0.25
Nodes (7): forbidden_inputs, HIV endpoint labels, locked internal test partition, MoleculeNet endpoint labels, policy, schema_version, selected_embedding_space

### Community 93 - "Frozen protocol: Step 2d candidate-generation scaling"
Cohesion: 0.25
Nodes (7): Budget and candidate definitions, Chemistry and novelty, Development and final separation, Development selection, Frozen protocol: Step 2d candidate-generation scaling, Scaling and decision rules, Scientific boundary

### Community 94 - "Frozen protocol: MoleculeNet development panel plus HIV confirmation"
Cohesion: 0.25
Nodes (8): Common-coverage rule, Evidence scope, Execution chronology and integrity, Frozen protocol: MoleculeNet development panel plus HIV confirmation, Frozen representations, Identical downstream probes, Molecular preparation and split inheritance, Timing scope

### Community 95 - "Manuscript Accuracy Audit"
Cohesion: 0.32
Nodes (8): Cosine-Tanimoto Spearman Sampling Protocol, Manuscript Accuracy Audit, Manuscript Wording and Units Corrections, Per-Fold StandardScaler Interpretation, Seed-42 ESOL Rounding, Seed-43 Calibration Protocol, Separate Edge Decoder MLPs, Table 1 Raw Hybrid Vector Protocol

### Community 96 - "Full Promotion Evaluation"
Cohesion: 0.39
Nodes (8): benchmark_moleculenet(), Checkpoint-Specific Calibration and Evaluation Panel, Full Promotion Evaluation, Promotion Gate Trajectory, Promotion Trajectory Audit, run_representation_probes(), Table 5 Full Promotion Protocol, Table 5 Promotion Trajectory CSV

### Community 97 - "files"
Cohesion: 0.29
Nodes (7): path, sha256, files, calibrator, packaged_resolved_config, path, sha256

### Community 98 - "files"
Cohesion: 0.29
Nodes (7): files, gmolai_checkpoint, packaged_calibrator, path, sha256, path, sha256

### Community 99 - "Frozen protocol: Step 2c chemical characterization"
Cohesion: 0.29
Nodes (6): Candidate denominators, Frozen protocol: Step 2c chemical characterization, Metrics, Molecular identity and categories, One-cut MMP definition, Question and boundary

### Community 100 - "Step 2d post-completion synthetic-accessibility extension"
Cohesion: 0.29
Nodes (6): Frozen summaries, Frozen visualizations, Interpretation boundary, Population and score, Status and scope, Step 2d post-completion synthetic-accessibility extension

### Community 101 - "Step 2b protocol: frozen-decoder candidate search and latent reranking"
Cohesion: 0.33
Nodes (5): Development and final discipline, Frozen boundary, Metrics, Question, Step 2b protocol: frozen-decoder candidate search and latent reranking

### Community 102 - "Step 2d results: frozen decoder candidate scaling"
Cohesion: 0.33
Nodes (5): Main result, Post-completion synthetic-accessibility comparison, Reproducibility notes, Scaling decision, Step 2d results: frozen decoder candidate scaling

### Community 103 - "Publication-Quality Training Plots"
Cohesion: 0.67
Nodes (6): Figure 1 Training and Validation Composite, Figure 2 Checkpoint Promotion Composite, Independent Plot Data Audit, Publication Export and Provenance Bundle, Publication-Quality Training Plots, Two-Seed Training Metrics Logs

### Community 104 - "Retained Checkpoint Availability Audit"
Cohesion: 0.53
Nodes (6): Checkpoint Archive Integrity, Evaluation Coverage Limit, Retained Checkpoint Availability Audit, Seed-42 Retained Checkpoint Set, Seed-43 Retained Checkpoint Set, Seed-42 Five-Checkpoint Set

### Community 105 - "Derivative-generation checkpoint"
Cohesion: 0.33
Nodes (6): Derivative-generation checkpoint, Frozen strategic decision, Future-work design (not an active objective), Integrity and outputs, SA extension, Terminal state

### Community 106 - "Protocol amendment 01: repeated-transformation support"
Cohesion: 0.40
Nodes (4): Amendment, Audit, Protocol amendment 01: repeated-transformation support, Trigger

### Community 107 - "step-01b-scaled-space-selection/scripts/run_study.sh"
Cohesion: 0.50
Nodes (4): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, run_logged(), run_study.sh script

### Community 108 - "step-02-decoder-feasibility/scripts/run_study.sh"
Cohesion: 0.50
Nodes (4): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, run_logged(), run_study.sh script

### Community 109 - "Step 2d results: frozen decoder candidate scaling"
Cohesion: 0.40
Nodes (4): Main result, Reproducibility notes, Scaling decision, Step 2d results: frozen decoder candidate scaling

### Community 110 - "step-02d-generation-scaling/scripts/run_study.sh"
Cohesion: 0.50
Nodes (4): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, run_gpu_phase(), run_study.sh script

### Community 111 - "Q: Update gmolai-rev6.docx to rev7 using the completed test-partition, MoleculeNet plus HIV, and speed benchmarks; stage the speed plots in the manuscript figures directory, embed no figures, and leave rev6 untouched."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Update gmolai-rev6.docx to rev7 using the completed test-partition, MoleculeNet plus HIV, and speed benchmarks; stage the speed plots in the manuscript figures directory, embed no figures, and leave rev6 untouched., Source Nodes

### Community 112 - "Q: The root extra-benchmark/README.md still says that moleculenet has not been executed while the MoleculeNet run is complete, and moleculenet/RESULTS.md says a controlled throughput benchmark is still needed even though speed is complete. Why were these missed, and can I rely on the documentation review?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: The root extra-benchmark/README.md still says that moleculenet has not been executed while the MoleculeNet run is complete, and moleculenet/RESULTS.md says a controlled throughput benchmark is still needed even though speed is complete. Why were these missed, and can I rely on the documentation review?, Source Nodes

### Community 113 - "Q: the results are very promising and I accept your decission. Proceed to the next step only: train and validate a molecular decoder conditioned on the frozen released gMolAI representation."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: the results are very promising and I accept your decission. Proceed to the next step only: train and validate a molecular decoder conditioned on the frozen released gMolAI representation., Source Nodes

### Community 114 - "Q: Step 2b frozen-decoder candidate generation and frozen-gMolAI latent reranking on a fresh validation panel"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Step 2b frozen-decoder candidate generation and frozen-gMolAI latent reranking on a fresh validation panel, Source Nodes

### Community 115 - "Q: Move to Step 2c: chemical characterization of the frozen Step-2b 50-candidate correct-condition sets using the Step-1b one-cut MMP rules, without training, regeneration, or latent perturbation."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Move to Step 2c: chemical characterization of the frozen Step-2b 50-candidate correct-condition sets using the Step-1b one-cut MMP rules, without training, regeneration, or latent perturbation., Source Nodes

### Community 116 - "Q: I do not see the valid plot (blue) in quality_locality_diversity_scaling.png"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: I do not see the valid plot (blue) in quality_locality_diversity_scaling.png, Source Nodes

### Community 117 - "Q: if feasible and easy do synthetic accessibility as part of Step 2d and create proper plots; then checkpoint for the next agent"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: if feasible and easy do synthetic accessibility as part of Step 2d and create proper plots; then checkpoint for the next agent, Source Nodes

### Community 118 - "Step 1b: scaled latent-space selection"
Cohesion: 0.50
Nodes (3): Contents, Status, Step 1b: scaled latent-space selection

### Community 119 - "external_chemistry_policy"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_chemistry_policy

### Community 120 - "optimized_inference"
Cohesion: 0.67
Nodes (3): optimized_inference, path, sha256

### Community 121 - "external_step1b_protocol"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step1b_protocol

### Community 122 - "external_step2b_candidates"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_candidates

### Community 123 - "TDC ADMET frozen-representation benchmark: completed results"
Cohesion: 0.20
Nodes (10): Bounded conclusions, Category-balanced summary, Completion and integrity, Descriptor diagnostic, gMolAI interpretation, Primary paired endpoint results, Realized common population, Scientific result (+2 more)

### Community 124 - "external_step2b_complete"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_complete

### Community 125 - "external_step2b_evaluation_source"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_evaluation_source

### Community 126 - "external_step2b_final_seal"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_final_seal

### Community 127 - "external_step2b_generation_stats"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_generation_stats

### Community 128 - "external_step2b_manifest"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_manifest

### Community 129 - "external_step2b_panel"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_panel

### Community 130 - "external_step2b_policy_seal"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_policy_seal

### Community 131 - "external_step2b_verification"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step2b_verification

### Community 132 - "external_validation_molecules"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_validation_molecules

### Community 133 - "source_audit"
Cohesion: 0.50
Nodes (4): source_audit, path, sha256, size_bytes

### Community 134 - "source_audit_core"
Cohesion: 0.50
Nodes (4): source_audit_core, path, sha256, size_bytes

### Community 135 - "source_common"
Cohesion: 0.50
Nodes (4): source_common, path, sha256, size_bytes

### Community 136 - "source_component_tests"
Cohesion: 0.50
Nodes (4): source_component_tests, path, sha256, size_bytes

### Community 137 - "main"
Cohesion: 0.18
Nodes (21): metric_estimate(), MetricSpec, paired_bootstrap(), DataFrame, Pure metric selection and paired-bootstrap functions., Select one candidate per seed with the frozen target-blind ordering., Return deterministic paired seed-bootstrap intervals for metric specs., select_reranked() (+13 more)

### Community 138 - "source_registration"
Cohesion: 0.50
Nodes (4): source_registration, path, sha256, size_bytes

### Community 139 - "source_report"
Cohesion: 0.50
Nodes (4): source_report, path, sha256, size_bytes

### Community 140 - "source_runner"
Cohesion: 0.50
Nodes (4): source_runner, path, sha256, size_bytes

### Community 141 - "source_verify"
Cohesion: 0.50
Nodes (4): source_verify, path, sha256, size_bytes

### Community 142 - "step-02c-chemical-characterization/scripts/run_study.sh"
Cohesion: 0.50
Nodes (3): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, run_study.sh script

### Community 144 - "Q: Step 2d frozen decoder candidate-generation scaling: what strategy and budget are supported?"
Cohesion: 0.50
Nodes (3): Answer, Outcome, Q: Step 2d frozen decoder candidate-generation scaling: what strategy and budget are supported?

### Community 146 - "base_config"
Cohesion: 0.67
Nodes (3): path, sha256, base_config

### Community 147 - "atomic_write_json"
Cohesion: 0.16
Nodes (17): load_screens(), Path, atomic_write_json(), atomic_write_text(), _json_default(), load_json(), load_protocol(), protocol_digest() (+9 more)

### Community 148 - "checkpoint"
Cohesion: 0.67
Nodes (3): path, sha256, checkpoint

### Community 149 - "chemistry_policy"
Cohesion: 0.67
Nodes (3): path, sha256, chemistry_policy

### Community 150 - "container"
Cohesion: 0.67
Nodes (3): path, sha256, container

### Community 151 - "inference_entrypoint"
Cohesion: 0.67
Nodes (3): inference_entrypoint, path, sha256

### Community 152 - "model_definition"
Cohesion: 0.67
Nodes (3): model_definition, path, sha256

### Community 153 - "optimized_inference"
Cohesion: 0.67
Nodes (3): optimized_inference, path, sha256

### Community 154 - "packaged_calibrator"
Cohesion: 0.67
Nodes (3): packaged_calibrator, path, sha256

### Community 155 - "packaged_checkpoint"
Cohesion: 0.67
Nodes (3): packaged_checkpoint, path, sha256

### Community 156 - "representation_selection"
Cohesion: 0.67
Nodes (3): representation_selection, path, sha256

### Community 157 - "step1b_complete"
Cohesion: 0.67
Nodes (3): step1b_complete, path, sha256

### Community 158 - "step1b_decision"
Cohesion: 0.67
Nodes (3): step1b_decision, path, sha256

### Community 159 - "train_molecules"
Cohesion: 0.67
Nodes (3): train_molecules, path, sha256

### Community 160 - "train_raw_embeddings"
Cohesion: 0.67
Nodes (3): train_raw_embeddings, path, sha256

### Community 161 - "validation_embeddings"
Cohesion: 0.67
Nodes (3): validation_embeddings, path, sha256

### Community 162 - "validation_molecules"
Cohesion: 0.67
Nodes (3): validation_molecules, path, sha256

### Community 164 - "chemistry_policy"
Cohesion: 0.67
Nodes (3): path, sha256, chemistry_policy

### Community 165 - "decoder_checkpoint"
Cohesion: 0.67
Nodes (3): path, sha256, decoder_checkpoint

### Community 166 - "decoder_inference_export"
Cohesion: 0.67
Nodes (3): path, sha256, decoder_inference_export

### Community 167 - "gmolai_calibrator"
Cohesion: 0.67
Nodes (3): gmolai_calibrator, path, sha256

### Community 168 - "gmolai_model_definition"
Cohesion: 0.67
Nodes (3): gmolai_model_definition, path, sha256

### Community 169 - "gmolai_resolved_config"
Cohesion: 0.67
Nodes (3): gmolai_resolved_config, path, sha256

### Community 170 - "inference_entrypoint"
Cohesion: 0.67
Nodes (3): inference_entrypoint, path, sha256

### Community 171 - "Scaled latent-space selection results"
Cohesion: 0.18
Nodes (10): Derivative retrieval, Directional transfer, Frozen primary comparison, Mean-node assessment, MMP scale and support, Output map, Required answers, Scaled latent-space selection results (+2 more)

### Community 172 - "sha256_file"
Cohesion: 0.19
Nodes (19): assemble_classyfire(), canonicalize_external(), panel_columns(), Path, Remove source-explicit H atoms, then apply the repository identity policy., relative_to_repository(), require_regular_file(), sha256_file() (+11 more)

### Community 173 - "packaged_checkpoint"
Cohesion: 0.67
Nodes (3): packaged_checkpoint, path, sha256

### Community 174 - "packaged_resolved_config"
Cohesion: 0.67
Nodes (3): packaged_resolved_config, path, sha256

### Community 175 - "representation_selection"
Cohesion: 0.67
Nodes (3): representation_selection, path, sha256

### Community 176 - "step2_common_source"
Cohesion: 0.67
Nodes (3): step2_common_source, path, sha256

### Community 177 - "step2_complete"
Cohesion: 0.67
Nodes (3): step2_complete, path, sha256

### Community 178 - "step2_decoder_model_source"
Cohesion: 0.67
Nodes (3): step2_decoder_model_source, path, sha256

### Community 179 - "step2_manifest"
Cohesion: 0.67
Nodes (3): step2_manifest, path, sha256

### Community 180 - "step2_original_final_panel"
Cohesion: 0.67
Nodes (3): step2_original_final_panel, path, sha256

### Community 181 - "step2_split_indices"
Cohesion: 0.67
Nodes (3): step2_split_indices, path, sha256

### Community 182 - "step2_training_complete"
Cohesion: 0.67
Nodes (3): step2_training_complete, path, sha256

### Community 183 - "train_molecules"
Cohesion: 0.67
Nodes (3): train_molecules, path, sha256

### Community 184 - "train_raw_embeddings"
Cohesion: 0.67
Nodes (3): train_raw_embeddings, path, sha256

### Community 185 - "validation_embeddings"
Cohesion: 0.67
Nodes (3): validation_embeddings, path, sha256

### Community 186 - "validation_molecules"
Cohesion: 0.67
Nodes (3): validation_molecules, path, sha256

### Community 187 - "metrics_common.py"
Cohesion: 0.18
Nodes (18): main(), ndarray, row_overlap(), main(), atomic_parquet(), atomic_save_npy(), atomic_save_npz(), exact_knn() (+10 more)

### Community 188 - "Frozen scientific protocol"
Cohesion: 0.22
Nodes (9): Data and roles, Frozen diagnostic-only runtime amendment, Frozen scientific protocol, Leakage and inference constraints, Molecular policy and common panel, Objective, Outcomes, Probe fitting (+1 more)

### Community 189 - "container"
Cohesion: 0.67
Nodes (3): path, sha256, container

### Community 190 - "Frozen benchmark protocol"
Cohesion: 0.12
Nodes (14): Decoder, Frozen benchmark protocol, Inputs, Metrics, Question, Timing regions, Uncertainty and interpretation, Reproduction (+6 more)

### Community 191 - "decoder_inference_export"
Cohesion: 0.67
Nodes (3): path, sha256, decoder_inference_export

### Community 192 - "gmolai_calibrator"
Cohesion: 0.67
Nodes (3): gmolai_calibrator, path, sha256

### Community 193 - "gmolai_checkpoint"
Cohesion: 0.67
Nodes (3): gmolai_checkpoint, path, sha256

### Community 194 - "gmolai_resolved_config"
Cohesion: 0.67
Nodes (3): gmolai_resolved_config, path, sha256

### Community 195 - "inference_entrypoint"
Cohesion: 0.67
Nodes (3): inference_entrypoint, path, sha256

### Community 196 - "optimized_inference"
Cohesion: 0.67
Nodes (3): optimized_inference, path, sha256

### Community 197 - "packaged_calibrator"
Cohesion: 0.67
Nodes (3): packaged_calibrator, path, sha256

### Community 198 - "packaged_checkpoint"
Cohesion: 0.67
Nodes (3): packaged_checkpoint, path, sha256

### Community 199 - "packaged_resolved_config"
Cohesion: 0.67
Nodes (3): packaged_resolved_config, path, sha256

### Community 200 - "representation_selection"
Cohesion: 0.67
Nodes (3): representation_selection, path, sha256

### Community 201 - "step1b_complete"
Cohesion: 0.67
Nodes (3): step1b_complete, path, sha256

### Community 202 - "step1b_fragment_source"
Cohesion: 0.67
Nodes (3): step1b_fragment_source, path, sha256

### Community 203 - "step1b_protocol"
Cohesion: 0.67
Nodes (3): step1b_protocol, path, sha256

### Community 204 - "step2_common_source"
Cohesion: 0.67
Nodes (3): step2_common_source, path, sha256

### Community 205 - "step2_complete"
Cohesion: 0.67
Nodes (3): step2_complete, path, sha256

### Community 206 - "step2_decoder_model_source"
Cohesion: 0.67
Nodes (3): step2_decoder_model_source, path, sha256

### Community 207 - "step2_manifest"
Cohesion: 0.67
Nodes (3): step2_manifest, path, sha256

### Community 208 - "step2_original_final_panel"
Cohesion: 0.67
Nodes (3): step2_original_final_panel, path, sha256

### Community 209 - "step2_split_indices"
Cohesion: 0.67
Nodes (3): step2_split_indices, path, sha256

### Community 210 - "step2_training_complete"
Cohesion: 0.67
Nodes (3): step2_training_complete, path, sha256

### Community 211 - "step2b_common_source_frozen"
Cohesion: 0.67
Nodes (3): step2b_common_source_frozen, path, sha256

### Community 212 - "step2b_development_panel"
Cohesion: 0.67
Nodes (3): step2b_development_panel, path, sha256

### Community 213 - "step2b_final_panel"
Cohesion: 0.67
Nodes (3): step2b_final_panel, path, sha256

### Community 214 - "step2b_protocol"
Cohesion: 0.67
Nodes (3): step2b_protocol, path, sha256

### Community 215 - "step2b_verification"
Cohesion: 0.67
Nodes (3): step2b_verification, path, sha256

### Community 216 - "step2c_audit_core_source"
Cohesion: 0.67
Nodes (3): step2c_audit_core_source, path, sha256

### Community 217 - "step2c_complete"
Cohesion: 0.67
Nodes (3): step2c_complete, path, sha256

### Community 218 - "step2c_protocol"
Cohesion: 0.67
Nodes (3): step2c_protocol, path, sha256

### Community 219 - "step2c_verification"
Cohesion: 0.67
Nodes (3): step2c_verification, path, sha256

### Community 220 - "train_molecules"
Cohesion: 0.67
Nodes (3): train_molecules, path, sha256

### Community 221 - "train_raw_embeddings"
Cohesion: 0.67
Nodes (3): train_raw_embeddings, path, sha256

### Community 222 - "validation_embeddings"
Cohesion: 0.67
Nodes (3): validation_embeddings, path, sha256

### Community 223 - "validation_molecules"
Cohesion: 0.67
Nodes (3): validation_molecules, path, sha256

### Community 231 - "Day-1 results: latent geometry and derivative retrieval"
Cohesion: 0.20
Nodes (9): Artifacts, Audited data, Bounded conclusion, Day-1 results: latent geometry and derivative retrieval, Geometry, Hybrid-384 retrieval, Outcome, Predeclared gates (+1 more)

### Community 238 - "checkpoint.py"
Cohesion: 0.43
Nodes (7): build_checkpoint(), capture_rng_state(), gather_rank_objects(), Any, Module, Optimizer, restore_rng_state()

### Community 239 - "tdc-admet/RESULTS.md"
Cohesion: 0.33
Nodes (4): Completed result, Interpretation boundary, TDC ADMET frozen-representation benchmark, Verification and reproduction

### Community 240 - "Decoder feasibility results"
Cohesion: 0.29
Nodes (6): Decoder feasibility results, Frozen decision audit, Frozen latent consistency and chemistry, Outcome, Reconstruction and explicit condition-use controls, Teacher-forced controls

### Community 241 - "gMolAI derivative generation studies"
Cohesion: 0.40
Nodes (5): Artifact policy, Closure and future-work boundary, gMolAI derivative generation studies, Reproducing Step 1, Study layout

### Community 242 - "Latent control-space decision"
Cohesion: 0.40
Nodes (4): Directional-transfer result, Latent control-space decision, Mean-node result, Weighting result

### Community 243 - "Frozen strategic direction"
Cohesion: 0.29
Nodes (7): Closed generation scope, Decision, Evidence retained permanently, Frozen generation baseline, Frozen strategic direction, Future work, not current execution, Retired placeholders

### Community 244 - "Q: explore the current repo and fully understand it. you can use graphify skill if that helps."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: explore the current repo and fully understand it. you can use graphify skill if that helps., Source Nodes

### Community 245 - "Q: Update the deriv-gen project direction based on the completed evidence."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Update the deriv-gen project direction based on the completed evidence., Source Nodes

### Community 246 - "Q: generate publication quality radar plots for MoleculeNet plus HIV and TDC ADMET"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: generate publication quality radar plots for MoleculeNet plus HIV and TDC ADMET, Source Nodes

### Community 247 - "Q: update the manuscript (both 'Materials and Methods' and 'Results and Discussion' when applicable) from rev7 to rev8 (the complete extra-benchmark added) and from rev8 to rev9 (step-01b, step-02, step-02b, step-02c, step-02d added). Notes: 1-keep older rev1-7 on the disk. 2-do not include the figures in the docx files. 3-create relevant and well-kept ordered headings/sub-headings. 4-use the same math and formula editing tool if there need to add math/formula. 5-the text must be scientific style and clear. 7-for rev9, feel free to create images/plots from the main results of the step-01b, step-02, step-02b, step-02c, step-02d. Do your best."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: update the manuscript (both 'Materials and Methods' and 'Results and Discussion' when applicable) from rev7 to rev8 (the complete extra-benchmark added) and from rev8 to rev9 (step-01b, step-02, step-02b, step-02c, step-02d added). Notes: 1-keep older rev1-7 on the disk. 2-do not include the figures in the docx files. 3-create relevant and well-kept ordered headings/sub-headings. 4-use the same math and formula editing tool if there need to add math/formula. 5-the text must be scientific style and clear. 7-for rev9, feel free to create images/plots from the main results of the step-01b, step-02, step-02b, step-02c, step-02d. Do your best., Source Nodes

### Community 248 - "external_step1b_fragment_source"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_step1b_fragment_source

### Community 249 - "gmolai_model_definition"
Cohesion: 0.67
Nodes (3): gmolai_model_definition, path, sha256

### Community 250 - "run_representation_probes"
Cohesion: 0.27
Nodes (17): _chemical_records(), _embedding_diagnostics(), _held_out_values(), _load_embedding_payload(), _molecules_and_labels(), Any, Mol, ndarray (+9 more)

### Community 251 - "Frozen minimal clustering benchmark protocol"
Cohesion: 0.13
Nodes (13): Common support, Frozen minimal clustering benchmark protocol, Identity policy, Property endpoint, Scientific scope, Statistics and visualization, Structural endpoint, ClassyFire-25 structural organization (+5 more)

### Community 252 - "read_panel_tsv"
Cohesion: 0.21
Nodes (11): read_panel_tsv(), write_csv(), main(), reason_category(), main(), property_space(), structural(), main() (+3 more)

### Community 253 - "main"
Cohesion: 0.38
Nodes (12): classyfire_pca(), configure(), coverage_figure(), decile_figure(), diagnostic_figure(), main(), metric_point_figure(), native_figure() (+4 more)

### Community 254 - "collect"
Cohesion: 0.36
Nodes (9): collect(), CollectArgs, propagate(), Tensor, NamedTuple, OptPairTensor, OptTensor, Size (+1 more)

### Community 255 - "QMugs: Quantum Mechanical Properties of Drug-like Molecules"
Cohesion: 0.18
Nodes (10): Bioactivity and pharmacological information, Calculating electron densities from the wavefunctions, Changelog, ChEMBL information, Content of QMugs and file structures:, Converting .wfn files into .fchk files, Downloading the dataset, QMugs: Quantum Mechanical Properties of Drug-like Molecules (+2 more)

### Community 256 - "Q: There is a folder in the root named 'inference' which is the user-faced SMILES -> gMolAI embeddings hydride x3. I want to re-structure it so that: it takes a list of SMILES and generates the gMolAI release hybride x3 embedings in .npz or .npy format (you decide whic one). I also want some nice CLI flags for when user want to generate candidates SMILES from the corresponding embeddings, each in a separate CSV files (a single CSV for each seed molecule). The candidates must be valid and unique canonical SMILES with a column for Morgan/Tanimoto similarity values. Run the example_smiles.csv both embeddings and 1000 derivatives for each smiles. Ensure that the this user-faced worflow uses the correct ckeckpoints, sampling strategy, etc. rename '/inference/model/' to '/inference/models/' and place the correct decoder checkpoint there and validate all check points and calibrator. Update the README.md accordingly, particularly explain the CLI flags for both encode (embeddings generation) and decoder (candidate generation from embeddings). Do it professionaly and user friendly."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: There is a folder in the root named 'inference' which is the user-faced SMILES -> gMolAI embeddings hydride x3. I want to re-structure it so that: it takes a list of SMILES and generates the gMolAI release hybride x3 embedings in .npz or .npy format (you decide whic one). I also want some nice CLI flags for when user want to generate candidates SMILES from the corresponding embeddings, each in a separate CSV files (a single CSV for each seed molecule). The candidates must be valid and unique canonical SMILES with a column for Morgan/Tanimoto similarity values. Run the example_smiles.csv both embeddings and 1000 derivatives for each smiles. Ensure that the this user-faced worflow uses the correct ckeckpoints, sampling strategy, etc. rename '/inference/model/' to '/inference/models/' and place the correct decoder checkpoint there and validate all check points and calibrator. Update the README.md accordingly, particularly explain the CLI flags for both encode (embeddings generation) and decoder (candidate generation from embeddings). Do it professionaly and user friendly., Source Nodes

### Community 257 - "Q: Review three findings: missing PyTorch and PyTorch Geometric package dependencies, missing GitHub Actions CI, and collection-sized memory accumulation in the public encoder; address only valid findings."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Review three findings: missing PyTorch and PyTorch Geometric package dependencies, missing GitHub Actions CI, and collection-sized memory accumulation in the public encoder; address only valid findings., Source Nodes

### Community 258 - "external_resolved_config"
Cohesion: 0.50
Nodes (4): path, sha256, size_bytes, external_resolved_config

### Community 259 - "resolved_config"
Cohesion: 0.67
Nodes (3): resolved_config, path, sha256

### Community 260 - "test_downstream_audit.py"
Cohesion: 0.27
Nodes (10): _cfg(), _hash(), test_atomic_csv_artifacts_use_lf_line_endings(), test_bucket_join_returns_exact_corpus_split_and_frozen_descriptors(), test_cycle_zero_seen_boundary_is_strict(), test_descriptor_matrix_uses_the_frozen_13_feature_order(), test_identity_metadata_reader_and_exact_target_locations(), test_no_training_audit_commands_are_registered() (+2 more)

### Community 261 - "Frozen protocol: Step 2b-style analysis of Step 2d candidates"
Cohesion: 0.25
Nodes (7): Direct search and structural metrics, Exclusions, Frozen inputs, Frozen-latent reranking, Frozen protocol: Step 2b-style analysis of Step 2d candidates, Scope, Uncertainty and figures

### Community 262 - "extra-step-02b-style/inputs/manifest.json"
Cohesion: 0.29
Nodes (6): created_at, encoder_input_order, encoder_input_rows, schema_version, study_id, writes_allowed_only_under

### Community 263 - "reencoding"
Cohesion: 0.29
Nodes (7): reencoding, backend, batch_size, embedding_dimensions, node_budget, threads, workers

### Community 264 - "atomic_write_csv"
Cohesion: 0.43
Nodes (6): atomic_write_csv(), main(), DataFrame, Figure, Path, save_figure()

### Community 265 - "assemble_qmugs"
Cohesion: 0.48
Nodes (6): assemble_qmugs(), coverage_rows(), main(), parse_args(), Any, Namespace

### Community 266 - "main"
Cohesion: 0.57
Nodes (6): contrast(), interval(), main(), metric_table(), DataFrame, rank_text()

### Community 267 - "immutable_inputs"
Cohesion: 0.33
Nodes (6): immutable_inputs, policy_audit, raw_proposals, bytes, path, sha256

### Community 268 - "Q: explore the repo and fully understanf it. Then I ask for some questions."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: explore the repo and fully understanf it. Then I ask for some questions., Source Nodes

### Community 269 - "analysis_config"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, analysis_config

### Community 270 - "analysis_protocol"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, analysis_protocol

### Community 271 - "candidate_characterization"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, candidate_characterization

### Community 272 - "conditions"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, conditions

### Community 273 - "encoder_calibrator"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, encoder_calibrator

### Community 274 - "encoder_checkpoint"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, encoder_checkpoint

### Community 275 - "encoder_cli"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, encoder_cli

### Community 276 - "encoder_config"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, encoder_config

### Community 277 - "encoder_input"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, encoder_input

### Community 278 - "encoder_model_source"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, encoder_model_source

### Community 279 - "encoder_selection"
Cohesion: 0.50
Nodes (4): bytes, path, sha256, encoder_selection

### Community 280 - "optimized_inference_source"
Cohesion: 0.50
Nodes (4): optimized_inference_source, bytes, path, sha256

### Community 281 - "panel"
Cohesion: 0.50
Nodes (4): panel, bytes, path, sha256

### Community 282 - "seed_budget_metrics"
Cohesion: 0.50
Nodes (4): seed_budget_metrics, bytes, path, sha256

### Community 283 - "step2d_final_analysis_summary"
Cohesion: 0.50
Nodes (4): step2d_final_analysis_summary, bytes, path, sha256

### Community 284 - "step2d_verification"
Cohesion: 0.50
Nodes (4): step2d_verification, bytes, path, sha256

### Community 285 - "unique_molecules"
Cohesion: 0.50
Nodes (4): unique_molecules, bytes, path, sha256

### Community 286 - "Step 2b-style search and latent-reranking analysis of Step 2d"
Cohesion: 0.50
Nodes (3): Figures, Main estimates, Step 2b-style search and latent-reranking analysis of Step 2d

### Community 287 - "run_analysis.sh"
Cohesion: 0.50
Nodes (3): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, run_analysis.sh script

### Community 288 - "sitecustomize.py"
Cohesion: 0.50
Nodes (3): load_config_with_effective_backend(), Path, Narrow compatibility shim for the release encoder's effective backend label.…

### Community 289 - "run_benchmark.sh"
Cohesion: 0.50
Nodes (3): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, run_benchmark.sh script

### Community 290 - "container"
Cohesion: 0.67
Nodes (3): path, sha256, container

### Community 291 - "decoder_checkpoint"
Cohesion: 0.67
Nodes (3): path, sha256, decoder_checkpoint

## Knowledge Gaps
- **797 isolated node(s):** `schema_version`, `policy`, `path`, `sha256`, `split` (+792 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **29 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `OptimizedSmilesEncoder` (6× useful, score=5.423540888)
- `Molecular Canonicalization Policy` (3× useful, score=2.726600102)
- `generate_embeddings.py` (3× useful, score=2.717337618)
- `Semantic Promotion Suite` (3× useful, score=2.560063333)
- `gmolai.py` (2× useful, score=1.843277033)
- `GraphConditionedEdgeDecoder` (2× useful, score=1.805688903)
- `MolecularRepresentationModel` (2× useful, score=1.802871328)
- `speed_adapter.py` (2× useful, score=1.786648986)
- `gMolAI Retraining Pipeline` (2× useful, score=1.763588575)
- `Manuscript rev4: exact downstream-molecule exposure audit` (2× useful, score=1.733958733)

**Known dead ends** — questions that led nowhere; don't re-derive.
- "done" -> `benchmark_io.py`, `finalize.py`, `adapter.py`

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `device` connect `train.py` to `load_protocol`, `scaled_common.py`, `main`, `reencoding`, `ValueError`, `evaluate_panel.py`, `gmolai.py`, `study_common.py`, `tune.py`, `sha256_file`, `downstream.py`, `fast_inference.py`, `day1_study.py`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `reencoding` connect `reencoding` to `train.py`, `extra-step-02b-style/inputs/manifest.json`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `immutable_inputs` connect `immutable_inputs` to `extra-step-02b-style/inputs/manifest.json`, `analysis_config`, `analysis_protocol`, `candidate_characterization`, `conditions`, `encoder_calibrator`, `encoder_checkpoint`, `encoder_cli`, `encoder_config`, `encoder_input`, `encoder_model_source`, `encoder_selection`, `optimized_inference_source`, `panel`, `seed_budget_metrics`, `step2d_final_analysis_summary`, `step2d_verification`, `unique_molecules`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Are the 327 inferred relationships involving `RuntimeError` (e.g. with `load_json()` and `require()`) actually correct?**
  _`RuntimeError` has 327 INFERRED edges - model-reasoned connections that need verification._
- **Are the 120 inferred relationships involving `ValueError` (e.g. with `effective_rank()` and `make_fingerprints()`) actually correct?**
  _`ValueError` has 120 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `MolecularRepresentationModel` (e.g. with `InferenceError` and `ModelBundle`) actually correct?**
  _`MolecularRepresentationModel` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `schema_version`, `policy`, `path` to the rest of the system?**
  _797 weakly-connected nodes found - possible documentation gaps or missing edges._