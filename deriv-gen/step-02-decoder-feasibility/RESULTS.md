# Decoder feasibility results

## Outcome

**NO-GO.** The decoder learned clear condition dependence, but it did not meet every predeclared fidelity criterion for a faithful inverse.

A new 28,316,160-parameter autoregressive decoder was trained; zero gMolAI parameters entered its optimizer. Fitting used 980,000 train-partition molecules, checkpoint selection used 20,000 scaffold-disjoint train-partition molecules, correct-condition teacher forcing used all 50,000 validation molecules, and all four autoregressive controls used a fixed 10,000-molecule validation panel. Locked-test rows and endpoint labels used: zero.

The registered train-development-only extension activated after epoch 12; the same decoder and optimizer continued at the fixed learning-rate floor, and the epoch-12 state was archived. The frozen checkpoint was selected at epoch 17; training stopped at epoch 20, without final-validation model selection. The deterministic decode method selected on the same development panel was greedy.

## Reconstruction and explicit condition-use controls

| Condition | Valid | Policy | Exact canonical | Identity | Scaffold | Morgan target | Latent cosine target | Source identity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Correct | 0.9752 | 0.9752 | 0.6390 | 0.6390 | 0.7470 | 0.8495 | 0.9938 | 0.6390 |
| Shuffled | 0.9753 | 0.9753 | 0.0000 | 0.0000 | 0.0053 | 0.1126 | 0.0087 | 0.6392 |
| Zero | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1471 | -0.0080 | NA |
| Nearest wrong | 0.9750 | 0.9750 | 0.0007 | 0.0007 | 0.5015 | 0.3918 | 0.8853 | 0.6612 |

Deterministic 2,000-bootstrap 95% confidence intervals are in `outputs/tables/metrics_by_control.csv`. Correct target identity was 0.6390; the best wrong-condition target identity was 0.0007; the absolute condition-use gap was 0.6383. Shuffled and nearest-wrong supplied-source identity recovery were 0.6392 and 0.6612.

Correct valid-SMILES was 0.9752 [0.9720, 0.9783]; exact canonical reconstruction was 0.6390 [0.6296, 0.6485]; all-row Morgan similarity was 0.8495 [0.8446, 0.8542].

## Teacher-forced controls

| Condition | NLL | Token accuracy |
|---|---:|---:|
| Correct | 0.0512 | 0.9827 |
| Shuffled | 6.9938 | 0.4521 |
| Zero | 5.6977 | 0.5423 |
| Nearest wrong | 2.1182 | 0.8098 |

All-validation correct-condition NLL was 0.0519 and token accuracy was 0.9823.

## Frozen latent consistency and chemistry

Correct policy-accepted outputs had median re-encoded cosine 1.0000 and mean relative L2 error 0.0491 to the supplied released vector. Re-encoding used the immutable packaged checkpoint, calibrator, optimized inference path, and released x3 hybrid.

The final panel contained 0 stereochemical targets, so stereochemical recovery is not estimable for this dataset. Invalid outputs count as reconstruction failure and zero all-row Morgan similarity.

## Frozen decision audit

Failed gates: correct_identity (0.6390 < 0.8000); correct_scaffold (0.7470 < 0.9000); correct_morgan (0.8495 < 0.9000)

This step ends at zero-perturbation reconstruction. No MMP-direction or derivative generation was performed.
