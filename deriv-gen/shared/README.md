# Shared contracts

Only cross-step, versioned contracts belong here. Step-specific scripts,
intermediate data, outputs, logs, and conclusions remain in the corresponding
numbered step directory so each experiment can be audited independently.

## Frozen contracts

- [`frozen-generation-v1/`](frozen-generation-v1/): permanently binds the
  existing Step-2 decoder bytes and the Step-2d
  `hybrid_b500_s500_t120`/1,000-proposal sampling strategy as the closed
  project's archival generation baseline and for any separately preregistered
  future candidate-library work. It is not an active optimization protocol.

Contracts are immutable. A scientific decision that changes a bound artifact
or policy must create a new version rather than editing an existing contract.
