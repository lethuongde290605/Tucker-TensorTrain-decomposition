I want to implement a Tucker-based variant of Eigen Attention for OPT.

Please first inspect the existing repository carefully, especially the current OPT Eigen Attention implementation, decoder-layer replacement logic, calibration flow, rank-search loop, and any existing Tucker/Tensor Train utilities. Do not blindly add a new class or force a new API. Reuse and adapt the existing code structure, naming style, and module boundaries wherever appropriate.

Background and goal:
The current Eigen Attention method collects calibration activations, builds a low-rank subspace for Q/K and V, merges the projection into the attention weights, and then chooses the compression level using decoder-layer output error. I want to implement the same high-level idea, but replace the SVD/eigenbasis with Tucker-based projection factors.

Very important clarification:
The current Eigen Attention code does not select ranks using direct reconstruction error of the decomposed matrix. Its reported error is the normalized output error of the decoder layer, comparing the original decoder layer output against the compressed decoder layer output on calibration inputs.

For the Tucker version, follow the same principle:

* Rank/threshold selection must be based on decoder-layer output error.
* Direct Tucker reconstruction error can be logged for debugging only.
* Do not accept or reject Tucker ranks based on reconstruction error.
* Do not confuse “tensor reconstruction quality” with “compressed decoder-layer approximation quality”.

Main objective:
Build a Tucker-only version of Eigen Attention. Do not use Tensor Train. Remove, bypass, or disable the Tensor Train path for this experiment.

Algorithmic requirements:

1. Calibration activation collection

Follow the existing Eigen Attention calibration flow. For each decoder layer, collect the Q, K, and V activations from calibration samples.

For OPT, the hidden dimension is usually 768. Conceptually, the calibration activations have shape:

* Q: `(num_samples, seq_len, hidden_dim)`
* K: `(num_samples, seq_len, hidden_dim)`
* V: `(num_samples, seq_len, hidden_dim)`

Flatten the sample and sequence dimensions into one calibration-token dimension, analogous to how Eigen Attention builds representation matrices.

Then tensorize the hidden dimension using configurable factor dimensions, for example:

* `hidden_dim = 768`
* `factor_dims = [8, 8, 12]`

Do not hard-code `[8, 8, 12]`. The factorization must come from configuration/arguments, and the implementation must validate that the product of `factor_dims` equals `hidden_dim`.

2. Shared Tucker subspace for Q and K

Like Eigen Attention uses a shared low-rank basis for Q and K, the Tucker version should use a shared Tucker projection for Q and K.

Conceptually:

* Tensorize Q.
* Tensorize K.
* Concatenate Q and K along the calibration-token axis.
* Perform Tucker decomposition on this concatenated Q/K tensor.

This produces the Tucker factors used to project both Q and K into the same low-dimensional Tucker subspace.

3. Separate Tucker subspace for V

Build a separate Tucker projection for V from the calibration V activations, analogous to Eigen Attention using a separate value basis.

4. Tucker decomposition modes

Apply Tucker only on the feature modes created by reshaping the hidden dimension.

For example, if the tensor shape is:

`(calibration_tokens, 8, 8, 12)`

then Tucker should decompose the feature modes corresponding to `8, 8, 12`, not the calibration-token mode.

The calibration-token mode should not be compressed by default. Only allow compressing it if explicitly configured.

5. Tucker rank selection

Support threshold-based rank selection for each Tucker mode.

For each feature mode:

* unfold the tensor along that mode,
* analyze the singular-value energy of that unfolding,
* choose the smallest mode-rank that reaches the target retained-energy threshold.

The rank list may look like:

`[r1, r2, r3]`

The Tucker latent dimension is:

`r1 * r2 * r3`

Important:
Do not treat this product rank as equivalent to SVD rank. Tucker product rank is a structured multilinear latent dimension, not a freely chosen matrix rank.

6. Tucker projection interpretation

The Tucker factors define a structured projection from the original hidden dimension into a lower-dimensional latent Tucker space.

Conceptually, if the factors are:

* A1: maps mode 1 to rank r1
* A2: maps mode 2 to rank r2
* A3: maps mode 3 to rank r3

then together they define a product/Kronecker-style projection from:

`hidden_dim`

to:

`r1 * r2 * r3`

The implementation may materialize this projection for correctness first, or apply the factors directly using tensor operations. Prioritize correctness and consistency with the existing codebase over speed.

7. Merge Tucker projections into attention weights

Follow the same principle as Eigen Attention: projections should be folded into the attention projection weights offline, so that inference produces low-dimensional Q, K, and V directly.

The merged Tucker attention should behave conceptually as follows:

* Q projection outputs low-dimensional Q in the shared Q/K Tucker space.
* K projection outputs low-dimensional K in the same shared Q/K Tucker space.
* V projection outputs low-dimensional V in the V Tucker space.
* K and V stored in the KV cache are low-dimensional.
* Attention scores are computed using low-dimensional Q and K.
* Attention aggregation uses low-dimensional V.
* The output projection maps the low-dimensional V-space result back to the model hidden dimension.

Preserve the same attention semantics, masking behavior, causal behavior, residual connections, layer norms, MLP, dropout, and output format as the original OPT decoder layer and the existing Eigen Attention implementation.

8. Decoder-layer output error is the selection metric

Implement Tucker threshold/rank search using the same criterion as Eigen Attention.

For each layer:

* Compute the original decoder layer output on calibration inputs.
* Build a Tucker-compressed version of that layer for a candidate threshold/rank configuration.
* Compute the compressed decoder layer output on the same calibration inputs.
* Measure normalized output error between original and compressed decoder outputs.
* If the output error is within the allowed error budget, keep this configuration as the best valid one and try a more aggressive compression.
* If the output error exceeds the budget, stop and revert to the last valid configuration.

Do not use direct Tucker reconstruction error for rank acceptance.

9. Rank/threshold search behavior

Use the existing Eigen Attention threshold-search style as the template, but make it robust.

The search should:

* start from a conservative high threshold,
* progressively lower the threshold to increase compression,
* track the last valid Tucker configuration,
* stop when the decoder-layer output error exceeds the error budget,
* return the last valid configuration.

Avoid fragile logic such as blindly adjusting the threshold after the loop. Store the last valid configuration explicitly.

10. Configuration requirements

Add or reuse configuration options for Tucker, including:

* enable/disable Tucker path,
* factor dimensions for tensorizing hidden dimension,
* initial threshold,
* threshold step size,
* minimum threshold,
* optional manual ranks for Q/K,
* optional manual ranks for V,
* decomposed modes,
* whether to log reconstruction error,
* whether to materialize the full projection or apply factors directly.

The exact names and location should follow the existing repository style.

11. Logging requirements

For each layer, log enough information to diagnose the compression behavior:

* layer index,
* Tucker factor dimensions,
* decomposed modes,
* threshold,
* Q/K Tucker ranks,
* V Tucker ranks,
* Q/K latent dimension,
* V latent dimension,
* Q/K per-mode retained energy,
* V per-mode retained energy,
* decoder-layer output error,
* compression ratio,
* optional Tucker reconstruction error.

The logs must clearly distinguish:

* `output_error`: decoder-layer output error used for rank selection,
* `reconstruction_error`: direct Tucker tensor reconstruction error, debug only, not used for rank selection.

12. Sanity checks

Add minimal checks to verify correctness:

* Full Tucker ranks should give near-original behavior.
* The compressed layer output shape must match the original decoder layer output shape.
* Low-dimensional Q and K must have the same latent dimension.
* Low-dimensional V and the input dimension of the modified output projection must match.
* The selected configuration must be the last one satisfying the decoder-layer output error budget.
* Tensor Train must not be used in the Tucker-only experiment.

13. Evaluation

After implementation, run the same kind of layer-wise comparison currently used by Eigen Attention.

Report:

* selected ranks per layer,
* output error per layer,
* compression ratio per layer,
* reconstruction error only as debug information,
* any downstream metric already supported by the repo.

Expected result:
The Tucker implementation should be comparable to Eigen Attention in workflow, but not necessarily in performance. The important goal is to make the comparison fair: both SVD/Eigen Attention and Tucker Attention should choose compression level based on decoder-layer output error, not direct reconstruction error.

Key warning:
Do not compare EigenAttention SVD rank directly with Tucker product rank as if they were equivalent. Compare by decoder-layer output error, compression ratio, and downstream model performance.
