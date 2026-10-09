---
title: ELEC5305 Project
---

# Streaming urban sound event detection on RK3588

**Mingze Li | Student ID 550374130 | ELEC5305**

I am investigating how pretrained YAMNet scores can be converted into stable urban sound event detections. The target is an RK3588 microphone prototype that reports sound classes and event intervals, including multiple classes at the same time. The main question is how temporal processing affects false alarms, fragmentation and detection delay.

## Initial implementation and testing

**Progress update: 9 October 2026.** YAMNet inference is working on Windows CPU and RK3588 CPU/NPU. A new offline Windows CPU pilot now processes complete audio scenes, compares independent thresholding with causal smoothing, and saves scores, nominal intervals and plots.

[Read the implementation report]({{ '/docs/initial_implementation.html' | relative_url }}) | [Download the submission PDF]({{ '/docs/submission/ELEC5305_Initial_Implementation_Mingze_Li_550374130.pdf' | relative_url }}) | [View the code](https://github.com/kolentossa/elec5305-project-550374130/tree/main/scripts)

The pilot uses three ESC-50 recordings of dog barking, a siren and a car horn, chosen before inference as the first filename alphabetically within fold 4 for each category. They are reused to construct six scenes totalling **56 seconds and 119 score patches**. Source insertion times are known, but audible event boundaries are unannotated.

A threshold of **0.2** is applied independently to each class, with and without a causal exponential moving average at **alpha 0.5**. These are fixed illustrative parameters and have not been tuned.

| Scene | Duration | Raw intervals | Smoothed intervals | Outcome |
| --- | ---: | ---: | ---: | --- |
| Dog only | 8 s | 1 | 1 | Dog-bark output crossed the threshold |
| Siren only | 8 s | 1 | 1 | Siren output crossed the threshold |
| Horn only | 8 s | 1 | 1 | Horn output crossed the threshold |
| Dog then siren | 14 s | 2 | 2 | Separate dog-bark and siren intervals |
| Mixed siren and horn | 10 s | 1 | 1 | Siren only; horn stayed below the threshold |
| Silence | 8 s | 0 | 0 | No output crossed the threshold |

### Sequential sounds

The dog and siren form separate nominal intervals. Smoothing extended their endings by one 0.48-second patch bin; interval counts stayed unchanged, so this run does not show reduced fragmentation.

![Sequential dog and siren scene with raw and smoothed scores and nominal intervals]({{ '/results/initial_pilot/sequential.png' | relative_url }})

### A mixed-scene failure

The horn's maximum score fell from **0.872814** alone to **0.079715** in the mixed scene, below the 0.2 threshold. The siren still crossed the threshold. Both mixed sources used gain 0.5, compared with 1.0 alone; this comparison does not isolate the effects of mixing and amplitude.

![Mixed siren and horn scene in which the horn stays below the threshold]({{ '/results/initial_pilot/overlap.png' | relative_url }})

[Recorded parameters and results]({{ '/results/initial_pilot/run.json' | relative_url }}) | [Nominal event intervals]({{ '/results/initial_pilot/events.csv' | relative_url }}) | [Source manifest]({{ '/results/initial_pilot/manifest.json' | relative_url }}) | [All score files and plots](https://github.com/kolentossa/elec5305-project-550374130/tree/main/results/initial_pilot)

## What these results establish

The implementation can process scenes longer than the model's three-second input and turn selected class scores into intervals. Four unit checks passed, covering event grouping, causal smoothing, window coverage and tail padding.

The intervals use patch-start bins. Each score needs 0.975 seconds of waveform context, and the runner uses three-second windows. These times are not annotated event boundaries or measured live emission times. The six scenes reuse only three source recordings, digital silence is the only background control, and only three classes are mapped. Accuracy, F1 and live latency have not been measured.

The earlier RK3588 NPU smoke test retained all **6 x 521 scores**. Its mean absolute difference from Windows CPU was **0.001982**, with top-1 agreement on five of six patches. This is a single-sample numerical diagnostic. The new pilot has been run on Windows CPU only.

[First inference report]({{ '/docs/first_run.html' | relative_url }}) | [RK3588 setup and reproduction]({{ '/docs/rk3588_setup.html' | relative_url }})

## Next work and feedback

Next I will add annotated URBAN-SED evaluation and the remaining class mapping, compare class-specific thresholds and hysteresis, and build the microphone streaming prototype on RK3588. Parameters will be selected on the validation set. Additional mixed scenes and an amplitude control will help investigate the horn failure.

I would appreciate teaching-staff feedback on this starting scope, the temporal methods, and the planned evaluation of fragmentation and detection delay.

## Project files

[GitHub repository](https://github.com/kolentossa/elec5305-project-550374130) | [Experiment reproduction and source credits]({{ '/docs/initial_implementation.html' | relative_url }})

The original CNN training proposal is retained as a historical document. The current direction focuses on temporal decisions using pretrained YAMNet.

[Original proposal PDF]({{ '/docs/ELEC5305_Project_Proposal_Mingze_Li.pdf' | relative_url }})
