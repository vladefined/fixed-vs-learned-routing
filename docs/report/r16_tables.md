# Round 16: books-shared-attractor, fine-tuned from A_s0 (5k steps, beta 1 -> 30)

| run | val CE | − A (2 seeds) | − A + 5k |
|---|---|---|---|
| A (2 seeds) | 1.3981 | — | |
| attractor, soft (beta 30) | 1.9741 | +0.5760 | — |
| attractor, hard (arg max) | 6.3616 | +4.9635 | — |

## soft

- symbol change share by block 1-8 (block 1: vs the embedding's symbol): 0.999 0.951 0.979 0.979 0.944 0.962 0.975 0.980
- changes per token 62.1 of 64; overall share 0.971
- book vectors used per group (of K, blocks 1-8): 1020 1004 1023 995 1022 1015 1018 1023
- distinct vectors per block (mean over groups): 986 460 241 155 215 471 410 499
- perplexity per block: 492 123 53 30 49 149 112 128
- mean max softmax weight per block: 0.28 0.11 0.09 0.06 0.07 0.09 0.05 0.04
- rho(changes, A loss) 0.073, within token 0.043; by A-loss quintile: 61.8 / 62.2 / 62.2 / 62.3 / 62.3

## hard

- symbol change share by block 1-8 (block 1: vs the embedding's symbol): 0.999 0.079 0.005 0.004 0.002 0.001 0.003 0.020
- changes per token 8.9 of 64; overall share 0.139
- book vectors used per group (of K, blocks 1-8): 1011 983 1018 968 1020 1012 987 1018
- distinct vectors per block (mean over groups): 986 994 996 996 997 997 997 999
- perplexity per block: 492 506 507 507 507 507 508 506
- rho(changes, A loss) 0.058, within token 0.019; by A-loss quintile: 8.7 / 8.9 / 9.1 / 8.8 / 9.0
