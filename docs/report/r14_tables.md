# Round 14: VQ state (no grid, no gate, no cost)

| run | val CE | − A (mean of 2 seeds) |
|---|---|---|
| A (2 seeds) | 1.3981 ± 0.0017 | — |
| books-shared | 2.5875 | +1.1894 |
| books-delta | 2.1114 | +0.7133 |
| shared-norm | 2.5090 | +1.1109 |
| delta-norm | 1.8245 | +0.4264 |
| shared-norm ft A (k-means) | 3.6474 | +2.2493 |
| delta-norm ft A (k-means) | 2.3020 | +0.9038 |
| A + 5k, lr 1e-4 (control) | 1.3737 | -0.0245 |

## books-shared (K = 1024)

- symbol changes: 0.569 of (token, block, group); 36.4 of 64 per token; tokens with no change 0.000
- change share by block: 1.000 0.062 0.094 0.365 0.772 0.722 0.549 0.992
- changes per group over the 8 blocks (0..8): 0.000 0.001 0.044 0.151 0.265 0.323 0.177 0.035 0.003
- vectors used per group (of K): 1024 1024 1024 1023 1024 1024 1024 1024
- distinct vectors per depth (mean over groups; emb, blocks 1-8): 100 402 421 442 509 675 778 814 430
- perplexity per depth: 76 163 180 197 245 329 420 436 236
- vectors used at 2+ depths: 0.822
- relative quantisation error ||z - q||^2 / ||z||^2 per state (emb, blocks 1-8): 0.39 0.07 0.02 0.03 0.07 0.14 0.18 0.17 0.14
- RMS of z per state (emb, blocks 1-8): 0.06 11.16 10.61 10.68 11.07 11.23 11.10 10.82 9.95
- difficulty: rho(changes, A loss) 0.093, within token -0.001, with own loss 0.124; R2 of the token itself 0.641
- rho by block: nan 0.06 0.05 0.01 0.01 0.02 0.10 0.08
- changes per token by A-loss quintile: 35.7 / 36.4 / 36.5 / 36.9 / 36.8

## books-delta (K = 256)

- symbol changes: 0.993 of (token, block, group); 63.6 of 64 per token; tokens with no change 0.000
- change share by block: 0.999 0.993 0.990 0.989 0.989 0.991 0.996 1.000
- changes per group over the 8 blocks (0..8): 0.000 0.000 0.000 0.000 0.000 0.000 0.003 0.048 0.950
- vectors used per block (mean over groups, of K): 256 256 256 256 256 256 256 256
- perplexity per block: 149 162 168 167 185 197 203 197
- zero update share per block: 0.001 0.007 0.010 0.011 0.011 0.009 0.004 0.000
- relative quantisation error ||z - q||^2 / ||z||^2 per block: 0.24 0.40 0.57 0.62 0.62 0.60 0.52 0.22
- RMS of z per block: 8.00 4.44 4.17 3.97 3.72 3.43 3.55 5.48
- difficulty: rho(changes, A loss) 0.074, within token 0.003, with own loss 0.122; R2 of the token itself 0.272
- rho by block: 0.00 -0.02 0.01 -0.02 0.05 0.14 0.09 -0.00
- changes per token by A-loss quintile: 63.4 / 63.6 / 63.6 / 63.6 / 63.6

## shared-norm (K = 1024)

- symbol changes: 0.715 of (token, block, group); 45.8 of 64 per token; tokens with no change 0.000
- change share by block: 0.999 0.512 0.549 0.602 0.662 0.658 0.744 0.993
- changes per group over the 8 blocks (0..8): 0.000 0.001 0.028 0.081 0.122 0.176 0.228 0.231 0.133
- vectors used per group (of K): 1024 1024 1024 1024 1024 1024 1024 1024
- distinct vectors per depth (mean over groups; emb, blocks 1-8): 542 1001 1020 1023 1024 1024 1024 1024 960
- perplexity per depth: 168 413 487 548 579 606 627 528 177
- vectors used at 2+ depths: 1.000
- direction error |u - e|^2 = 2 - 2 cos per state (emb, blocks 1-8): 0.27 0.73 0.70 0.76 0.79 0.82 0.85 0.87 0.76
- mean length |z_g| per state (emb, blocks 1-8): 0.37 6.00 9.23 15.20 27.88 54.97 111.26 279.52 1443.00
- difficulty: rho(changes, A loss) 0.093, within token 0.000, with own loss 0.123; R2 of the token itself 0.520
- rho by block: 0.00 0.06 0.08 0.04 0.04 0.07 0.04 -0.00
- changes per token by A-loss quintile: 44.6 / 45.4 / 45.9 / 46.5 / 46.4

## delta-norm (K = 256)

- symbol changes: 1.000 of (token, block, group); 64.0 of 64 per token; tokens with no change 0.000
- change share by block: 1.000 1.000 1.000 1.000 1.000 0.999 1.000 0.999
- changes per group over the 8 blocks (0..8): 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.002 0.998
- vectors used per block (mean over groups, of K): 256 256 256 256 256 256 256 256
- perplexity per block: 240 242 246 249 250 249 248 247
- direction error |u - e|^2 = 2 - 2 cos per block: 0.90 1.03 1.07 1.10 1.10 1.10 1.07 1.00
- mean length |z_g| per block: 15.36 8.68 8.06 7.46 8.26 8.69 9.65 9.69
- update / state length |dx_g|/|x_g| per block: mean 56.337 0.601 0.440 0.390 0.372 0.364 0.358 0.236; median 48.419 0.561 0.433 0.385 0.367 0.358 0.352 0.209
- share of updates with |dx|/|x| >= 0.01 per block: 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000
- share of updates with |dx|/|x| >= 0.05 per block: 1.000 1.000 1.000 1.000 1.000 0.999 1.000 0.999
- share of updates with |dx|/|x| >= 0.1 per block: 1.000 1.000 1.000 0.997 1.000 0.996 0.998 0.966
- difficulty, continuous (sum of |dx|/|x|): rho with A loss -0.055, within token -0.013; by A-loss quintile 491.64 / 470.52 / 484.86 / 452.35 / 464.67
- difficulty: rho(changes, A loss) -0.042, within token -0.027, with own loss -0.016; R2 of the token itself 0.391
- rho by block: nan nan nan nan nan -0.01 nan -0.05
- changes per token by A-loss quintile: 64.0 / 64.0 / 64.0 / 64.0 / 64.0

## shared-norm ft A (k-means) (K = 1024)

- symbol changes: 0.377 of (token, block, group); 24.1 of 64 per token; tokens with no change 0.000
- change share by block: 0.992 0.309 0.241 0.184 0.190 0.176 0.318 0.606
- changes per group over the 8 blocks (0..8): 0.001 0.099 0.268 0.303 0.207 0.094 0.025 0.004 0.000
- vectors used per group (of K): 1024 1024 1024 1024 1024 1024 1024 1024
- distinct vectors per depth (mean over groups; emb, blocks 1-8): 332 1012 1022 1024 1024 1024 1024 1024 1024
- perplexity per depth: 117 377 444 517 584 656 701 697 465
- vectors used at 2+ depths: 1.000
- direction error |u - e|^2 = 2 - 2 cos per state (emb, blocks 1-8): 0.32 0.84 0.78 0.77 0.77 0.75 0.75 0.77 0.66
- mean length |z_g| per state (emb, blocks 1-8): 0.28 1.71 4.01 7.33 13.32 23.55 41.69 75.97 175.84
- difficulty: rho(changes, A loss) 0.063, within token 0.016, with own loss 0.128; R2 of the token itself 0.445
- rho by block: 0.04 0.07 0.03 0.04 0.05 0.00 0.01 0.05
- changes per token by A-loss quintile: 23.4 / 24.0 / 24.3 / 24.5 / 24.4

## delta-norm ft A (k-means) (K = 256)

- symbol changes: 1.000 of (token, block, group); 64.0 of 64 per token; tokens with no change 0.000
- change share by block: 1.000 1.000 1.000 1.000 1.000 0.999 1.000 1.000
- changes per group over the 8 blocks (0..8): 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.999
- vectors used per block (mean over groups, of K): 256 256 256 256 256 256 256 256
- perplexity per block: 224 233 235 244 244 242 242 234
- direction error |u - e|^2 = 2 - 2 cos per block: 0.79 0.90 0.96 1.02 1.04 1.05 1.04 1.00
- mean length |z_g| per block: 7.68 5.43 6.07 6.73 7.98 9.27 12.67 11.71
- update / state length |dx_g|/|x_g| per block: mean 29.224 0.770 0.588 0.531 0.512 0.491 0.562 0.393; median 25.583 0.722 0.556 0.506 0.491 0.483 0.538 0.367
- share of updates with |dx|/|x| >= 0.01 per block: 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000
- share of updates with |dx|/|x| >= 0.05 per block: 1.000 1.000 1.000 1.000 1.000 0.999 1.000 1.000
- share of updates with |dx|/|x| >= 0.1 per block: 1.000 1.000 0.999 0.994 0.995 0.995 0.999 1.000
- difficulty, continuous (sum of |dx|/|x|): rho with A loss 0.013, within token -0.001; by A-loss quintile 262.49 / 266.84 / 271.04 / 259.61 / 262.88
- difficulty: rho(changes, A loss) -0.006, within token nan, with own loss 0.030; R2 of the token itself 1.000
- rho by block: nan nan nan nan nan -0.01 nan nan
- changes per token by A-loss quintile: 64.0 / 64.0 / 64.0 / 64.0 / 64.0
