# Frozen protocol: decoder feasibility from released gMolAI vectors

## Boundary

Only a new molecular decoder is trainable. The seed-42 step-10,000 gMolAI
checkpoint, train-only coordinate calibrator, and released 384-dimensional
`[graph_std, 3 * mean_node_std]` vector are immutable. No gMolAI parameter,
checkpoint, calibration statistic, or public embedding definition is modified.

This step tests zero-perturbation inversion only. It does not fit or apply MMP
directions and does not generate derivatives. Locked-test molecules and all
endpoint labels are forbidden.

## Populations

The decoder sees the deterministic 1,000,000-molecule pretraining-train sample
exported and sealed in Step 1b. A deterministic scaffold-group holdout of about
20,000 molecules inside that sample is used only for checkpoint selection; no
scaffold group crosses its decoder-train/development boundary.

The independent 50,000-molecule pretraining-validation export is untouched
during training and checkpoint selection. Final teacher-forced likelihood is
measured on all 50,000 rows. Autoregressive chemistry metrics and all controls
use one predeclared deterministic 10,000-molecule validation panel. Train and
validation molecule hashes and nonempty Bemis-Murcko scaffolds must be disjoint.

## Sequence representation and chemistry

Targets are the canonical isomeric SMILES already produced by gMolAI's
canonicalization policy: disconnected inputs rejected, supported elements only,
2-256 atoms, and stereochemistry retained when present.

The decoder uses a lossless fixed ASCII-byte vocabulary: PAD, BOS, EOS, and all
128 ASCII bytes. It therefore has no validation OOV tokens and can express
bracket atoms, charges, ring syntax, and `@`, `/`, and `\` stereochemistry
without changing the chemistry policy. The observed 1M train maximum is 112
bytes and the validation maximum is 87; the frozen generation ceiling is 128.

## Decoder

The primary design is a six-layer autoregressive Transformer decoder
(`d_model=512`, eight heads, FFN width 2,048). The frozen 384-D vector is
projected to four cross-attention memory tokens and to an additive token-stream
bias, exposing the condition at every decoding layer and time step. This is a
new decoder only; the gMolAI encoder is never part of the optimizer.

Training minimizes teacher-forced next-byte cross entropy. For a deterministic
25% sub-batch, an auxiliary hinge requires the correct condition to give at
least 0.10 nats/token lower loss than a within-batch wrong condition. This
trains condition dependence without changing gMolAI. Checkpoints are selected
only from the train-partition development holdout using correct reconstruction,
condition-control deterioration, and teacher-forced NLL.

## Final controls

Every final-panel target is greedily decoded with identical settings under:

1. its correct released embedding;
2. a deterministic shuffled-panel derangement;
3. the all-zero vector;
4. the nearest non-self validation molecule's released embedding, a hard wrong
   condition under Euclidean distance.

For shuffled and hard-wrong controls, metrics are computed both against the
original target and against the molecule that supplied the condition.

## Metrics

Report RDKit valid-SMILES rate, gMolAI-policy acceptance, exact canonical
reconstruction after applying the frozen chemistry policy, canonical
molecular-hash recovery, Morgan Tanimoto to target, Bemis-Murcko scaffold
recovery, and source-identity recovery for controls. Byte-for-byte equality of
the emitted SMILES to the canonical target is retained as a separate diagnostic.
Invalid outputs count as failures and have zero all-row Morgan similarity.

Every policy-accepted decoded molecule is re-encoded using the packaged frozen
optimized gMolAI inference pipeline. Report cosine similarity, RMSE, L2, and
relative L2 to both the original target vector and the actually supplied
condition. Stereochemical recovery is reported when stereochemical targets
exist; if the frozen panel contains none, it is explicitly not estimable.

Uncertainty for core proportions and means uses 2,000 deterministic bootstrap
replicates on the fixed molecule panel.

## Frozen GO / NO-GO rule

A GO requires all of:

- at least 10,000 final autoregressive validation targets;
- correct-condition valid-SMILES rate at least 0.95;
- correct-condition molecular identity recovery at least 0.80;
- correct-condition scaffold recovery at least 0.90;
- correct-condition mean all-row Morgan similarity at least 0.90;
- median correct-condition re-encoded cosine at least 0.95;
- correct target-identity recovery at least 0.50 above the best shuffled, zero,
  or hard-wrong target recovery;
- hard-wrong supplied-source identity recovery at least 0.60;
- shuffled supplied-source identity recovery at least 0.60.

Failure of any gate yields NO-GO. GO means the decoder learned a faithful,
condition-dependent inverse on held-out validation molecules; it is not
permission to begin latent perturbation in this step.


## Development-only duration safeguard

The baseline optimizer schedule runs for 12 epochs. Before epoch 12 completed and
before any final validation generation, a bounded extension rule was registered
in `config/development_extension.json`. If epoch-12 correct identity remains
below 0.80 but improves by at least 0.01 from epoch 9, correct NLL also improves,
and shuffled/zero target identity remains at most 0.05, the same decoder and
optimizer may continue to at most epoch 24 at the 4e-5 learning-rate floor with
the same checkpoint score and three-epoch early stopping. The epoch-12
checkpoint and summary must be preserved.

This safeguard may change training duration using only the train-partition
development split. It cannot change the model, data, final GO/NO-GO gates,
decoding method, or any frozen gMolAI input, and final validation remains
untouched until one checkpoint is frozen.


## Development-only decode selection

Before final validation, the fixed 2,048-molecule train-partition development
panel compares greedy decoding with deterministic width-4 beam search
(length penalty 0.6). Beam decoding is selected only if it gains at least 0.02
correct identity, loses at most 0.01 validity, keeps shuffled/zero target
identity at most 0.05, and retains shuffled supplied-source identity within
0.01 of correct identity. Otherwise greedy remains the fallback. The selected
method is then frozen and applied identically to all four final validation
controls. The registered rule is in `config/decode_selection.json`.
