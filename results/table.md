| direction | context | val loss | loss@last | d1 lift | d2 lift | d1 lift per pair | d1 slant lift | coverage | same spelling | seen word pair | copied lines |
|---|---|---|---|---|---|---|---|---|---|---|---|
| forward | 16 | 1.680 | 1.545 | +0.0 ± 0.5 | -0.1 ± 0.3 | +0.0 ± 0.3 | -0.2 ± 1.9 | 56% | 22% | 52% | 0.0% |
| forward | 32 | 1.552 | 1.462 | +0.0 ± 0.2 | -0.3 ± 0.2 | +0.0 ± 0.1 | -0.3 ± 1.4 | 66% | 9% | 47% | 0.6% |
| forward | 64 | 1.471 | 1.417 | +0.3 ± 0.2 | +0.0 ± 0.3 | +0.2 ± 0.1 | +0.3 ± 0.5 | 67% | 19% | 61% | 0.7% |
| forward | 128 | 1.414 | 1.363 | +0.4 ± 0.7 | +0.3 ± 0.2 | +0.3 ± 0.5 | +1.7 ± 1.5 | 70% | 10% | 52% | 1.7% |
| forward | 256 | 1.401 | 1.353 | +0.4 ± 0.3 | +0.3 ± 0.2 | +0.3 ± 0.2 | +1.2 ± 1.1 | 70% | 22% | 51% | 1.4% |
| forward | 512 | 1.360 | 1.348 | +0.3 ± 0.3 | +0.2 ± 0.4 | +0.1 ± 0.2 | +1.0 ± 0.5 | 64% | 15% | 61% | 1.6% |
| reversed | 16 | 1.689 | 1.582 | -0.1 ± 0.3 | -0.2 ± 0.2 | -0.0 ± 0.2 | -0.8 ± 0.8 | 58% | 0% | 63% | 0.1% |
| reversed | 32 | 1.564 | 1.480 | +0.1 ± 0.4 | -0.5 ± 0.2 | +0.0 ± 0.2 | -0.6 ± 0.2 | 59% | 25% | 57% | 0.9% |
| reversed | 64 | 1.469 | 1.400 | +2.3 ± 0.6 | +0.5 ± 0.2 | +1.5 ± 0.4 | +3.1 ± 0.5 | 66% | 35% | 52% | 1.4% |
| reversed | 128 | 1.413 | 1.365 | +4.0 ± 0.4 | +2.6 ± 0.9 | +2.5 ± 0.3 | +6.9 ± 0.8 | 63% | 53% | 43% | 1.4% |
| reversed | 256 | 1.385 | 1.371 | +6.5 ± 0.4 | +4.4 ± 0.4 | +3.6 ± 0.3 | +10.5 ± 1.1 | 58% | 53% | 43% | 1.3% |
| reversed | 512 | 1.348 | 1.367 | +4.2 ± 1.6 | +4.2 ± 0.9 | +2.4 ± 0.9 | +7.9 ± 0.9 | 58% | 61% | 38% | 1.5% |
train_d1 {'coverage': 0.804, 'perfect': 0.105, 'slant': 0.371, 'perfect_per_pair': 0.084, 'chance_perfect': 0.006, 'chance_slant': 0.093, 'lift_perfect': 0.099, 'lift_slant': 0.277, 'lift_per_pair': 0.079, 'same_spelling_share': 0.385}
train_d2 {'coverage': 0.819, 'perfect': 0.062, 'slant': 0.281, 'perfect_per_pair': 0.051, 'chance_perfect': 0.006, 'chance_slant': 0.093, 'lift_perfect': 0.056, 'lift_slant': 0.188, 'lift_per_pair': 0.045, 'same_spelling_share': 0.342}
val_d1 {'coverage': 0.8, 'perfect': 0.083, 'slant': 0.331, 'perfect_per_pair': 0.066, 'chance_perfect': 0.006, 'chance_slant': 0.088, 'lift_perfect': 0.077, 'lift_slant': 0.243, 'lift_per_pair': 0.061, 'same_spelling_share': 0.356, 'seen_pair_share': 0.337}
val_d2 {'coverage': 0.808, 'perfect': 0.045, 'slant': 0.229, 'perfect_per_pair': 0.036, 'chance_perfect': 0.006, 'chance_slant': 0.091, 'lift_perfect': 0.039, 'lift_slant': 0.138, 'lift_per_pair': 0.031, 'same_spelling_share': 0.315, 'seen_pair_share': 0.333}
