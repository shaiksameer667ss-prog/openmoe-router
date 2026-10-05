# FWR Multi-Step Run Record — Step10, Forced Pair (3,4)

Exact step10 checkpoint:
`e69fb593e6b158db1b5db16ffe26c75da0113bcc8ee9fa9d9c8abd8077823041`

Exact replay:
`de9136168116970af6051e73e64b208ee776ac127367f52862daabcaf65da8ed`

Protocol:
- 715 replay examples
- 45,760 aligned tokens
- T4 fixed batch = first 128 Task4 train examples, shuffle=False
- forced pair = (3,4)
- eta = 0.001
- 20 sequential direct-SGD steps
- router observation disabled
- eval mode
- no optimizer state

## Main result

Spearman rho over all 20 steps = **+1.000000**.
Sign agreement = **20/20**.
Median relative first-order error = **4.113310e-4**.
Maximum relative error = **7.337738e-3**.
Cumulative residual = **-1.346006e-6**.

There is no observed early-valid / late-collapse regime: prefix rho is +1.0 for every prefix n=3..20, including the late 10 steps.

See the CSV and JSON files in this directory for the exact per-step numerical record.

## Technical fix preserved by this checkpoint

The earlier implementation failure came from comparing image-level L1 h rows with token-level router gate rows. The correct alignment is:

- L1 h: [B,64,512]
- flatten to: [B*64,512]
- dense router gates: [B*64,8]
- exact replay: 715*64 = 45,760 tokens.

The forced-pair patch is applied to the L1 `ContinualRouter`, matching the Cell-1 helper signature `(router, pair)`.

## Scientific interpretation

This run establishes **multi-step predictor validity for FWR on the forced (3,4) trajectory**. It does not yet establish task-level retention benefit or justify router training.

Current decision remains:

**DO NOT TRAIN THE ROUTER YET.**
