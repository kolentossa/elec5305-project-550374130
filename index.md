---
title: ELEC5305 Project
---

# Urban sound event detection on RK3588 using pretrained YAMNet

**Mingze Li | Student ID 550374130 | ELEC5305**

I am investigating how pretrained YAMNet scores can be converted into stable urban sound event detections. The implementation tests prerecorded WAV files on RK3588 and reports sound classes and nominal event intervals. The main question is how temporal processing affects false alarms, missed labels, fragmentation and event boundary error.

## Initial implementation and testing

**Progress update: 9 October 2026.** The same six prerecorded audio scenes have now been tested on the **RK3588 NPU (LubanCat-5 V2)** and Windows CPU. The implementation processes complete WAV files, compares independent thresholding with causal smoothing, and saves scores, nominal intervals and plots.

[Read the implementation report]({{ '/docs/initial_implementation.html' | relative_url }}) | [Download the submission PDF]({{ '/docs/submission/ELEC5305_Initial_Implementation_Mingze_Li_550374130.pdf' | relative_url }}) | [View the code](https://github.com/kolentossa/elec5305-project-550374130/tree/main/scripts)

The pilot uses three ESC-50 recordings of dog barking, a siren and a car horn, chosen before inference as the first filename alphabetically within fold 4 for each category. They are reused to construct six scenes totalling **56 seconds and 119 score patches**. Source insertion times are known, but audible event boundaries are unannotated.

Both backends produced the interval counts below. A threshold of **0.2** is applied independently to each class, with and without a causal exponential moving average at **alpha 0.5**. These are fixed illustrative parameters and have not been tuned.

| Scene | Duration | Raw intervals | Smoothed intervals | Outcome |
| --- | ---: | ---: | ---: | --- |
| Dog only | 8 s | 1 | 1 | Dog-bark output crossed the threshold |
| Siren only | 8 s | 1 | 1 | Siren output crossed the threshold |
| Horn only | 8 s | 1 | 1 | Horn output crossed the threshold |
| Dog then siren | 14 s | 2 | 2 | Separate dog-bark and siren intervals |
| Mixed siren and horn | 10 s | 1 | 1 | Siren only; horn stayed below the threshold |
| Silence | 8 s | 0 | 0 | No output crossed the threshold |

### Recording-level results

On both the RK3588 NPU and Windows CPU, raw thresholding and EMA each gave **micro precision 100.0% and recall 85.7%**: six true positive labels, zero false positive labels and one missed horn label. There are 18 scene-label decisions per backend. A label is expected if its source file was inserted, and predicted if it produces at least one interval anywhere in that scene. These are presence metrics for three reused source recordings, not an evaluation of event boundaries.

[NPU metrics]({{ '/results/initial_pilot_rk3588_npu/metrics.json' | relative_url }}) | [CPU metrics]({{ '/results/initial_pilot/metrics.json' | relative_url }})

### Sequential sounds

The dog and siren form separate nominal intervals. Smoothing extended their endings by one 0.48-second patch bin; interval counts stayed unchanged, so this run does not show reduced fragmentation.

![RK3588 NPU scores and nominal intervals for sequential dog and siren recordings]({{ '/results/initial_pilot_rk3588_npu/sequential.png' | relative_url }})

### A mixed-scene failure

On NPU, the horn's maximum score was **0.876465** alone and **0.077698** in the mixed scene, below the 0.2 threshold. The CPU reference also missed the mixed horn (0.872814 alone and 0.079715 mixed). The siren still crossed the threshold. Both mixed sources used gain 0.5, compared with 1.0 alone; this comparison does not isolate the effects of mixing and amplitude.

![RK3588 NPU scores for a mixed siren and horn recording in which the horn stays below the threshold]({{ '/results/initial_pilot_rk3588_npu/overlap.png' | relative_url }})

[NPU run record]({{ '/results/initial_pilot_rk3588_npu/run.json' | relative_url }}) | [NPU event intervals]({{ '/results/initial_pilot_rk3588_npu/events.csv' | relative_url }}) | [Source manifest]({{ '/results/initial_pilot_rk3588_npu/manifest.json' | relative_url }}) | [NPU execution log]({{ '/results/initial_pilot_rk3588_npu/inference.log' | relative_url }}) | [All NPU results](https://github.com/kolentossa/elec5305-project-550374130/tree/main/results/initial_pilot_rk3588_npu)

## What these results establish

The implementation can process scenes longer than the model's three-second input and turn selected class scores into intervals. Six unit checks passed on Windows and RK3588, covering event grouping, causal smoothing, window coverage, tail padding and metric counts. The recorded NPU run used core 0 with RKNN Runtime 2.3.2 and driver 0.9.8.

The intervals use patch-start bins. Each score needs 0.975 seconds of waveform context, and the runner uses three-second windows. These times describe score bins rather than annotated event boundaries. The six scenes reuse only three source recordings, digital silence is the only background control, and only three classes are mapped. Strongly annotated event detection accuracy and boundary errors have not been measured.

The earlier RK3588 NPU smoke test retained all **6 x 521 scores**. Its mean absolute difference from Windows CPU was **0.001982**, with top-1 agreement on five of six patches. This is a single-sample numerical diagnostic. The new recorded-audio pilot provides six-scene NPU testing and a matching CPU reference; the nominal intervals agree on these scenes.

[First inference report]({{ '/docs/first_run.html' | relative_url }}) | [RK3588 setup and reproduction]({{ '/docs/rk3588_setup.html' | relative_url }})

## Next work and feedback

Next I will add annotated URBAN-SED evaluation and the remaining class mapping, compare class-specific thresholds and hysteresis, and test the selected method on prerecorded WAV files on RK3588. Parameters will be selected on the validation set. Additional mixed scenes and an amplitude control will help investigate the horn failure.

I would appreciate teaching-staff feedback on this starting scope, the temporal methods, and the planned evaluation of fragmentation and event boundary error.

## Project files

[GitHub repository](https://github.com/kolentossa/elec5305-project-550374130) | [Experiment reproduction and source credits]({{ '/docs/initial_implementation.html' | relative_url }})

The original CNN training proposal is retained as a historical document. The current direction focuses on temporal decisions using pretrained YAMNet.

[Original proposal PDF]({{ '/docs/ELEC5305_Project_Proposal_Mingze_Li.pdf' | relative_url }})
