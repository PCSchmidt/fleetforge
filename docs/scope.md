# FleetForge scope (draft, 2026-10-21)

Edge MLOps: ship a model to many devices as a signed artifact, roll it out by canary, decide on-device whether to promote or roll back. This page fixes what to build first, what the laptop can do alone, and what is unknown. Nothing is built.

## Question the project answers

How quickly and how safely can a fleet adopt a new model, and how few frames does a device need to see before it is justified to promote it? The measured outputs are time to full rollout, time to automatic rollback after an injected bad model, bytes per update, and the error rates of the promotion test.

## Milestones

| # | Milestone | Needs | Done when |
| --- | --- | --- | --- |
| 0.1 (done in simulation, see sequential-test.md) | Sequential shadow-evaluation test: candidate and production models score the same frames; a sequential test decides promote, hold or roll back | Laptop | Type I and II error measured by simulation against the nominal levels over many injected improvements and regressions; average frames to decision reported against a fixed-sample test |
| 0.2 (done, see artifact.md) | Signed artifact format: model file plus manifest (version, hashes, target, metrics) signed; verify refuses unsigned, tampered, wrong-target and older-than-floor artifacts | Laptop | Attack tests pass; the verify path reuses IronBoot's rollback-floor logic where it fits |
| 0.3 (done, see agent.md) | Device agent: pull-based updates over a link that drops and truncates, A/B model slots, resume, and a cut-at-every-step test as in IronBoot | Laptop | Exhaustive interruption sweep with no state where the device has no valid model |
| 0.4 (done in simulation, see fleet.md) | Control plane and fleet simulation: groups, canary percentages, rollback triggers, many emulated devices in containers | Docker | Rollout and rollback timings measured on a few hundred emulated devices |
| 0.5 | Real targets: build a model for the desktop GPU (TensorRT) with an accuracy-regression gate; then Jetson and a microcontroller artifact | Jetson, Pico 2 | Same pipeline produces and rolls out artifacts for at least two real device types |
| 0.6 | Delta updates: bytes saved against full shipping | Laptop | Table of bytes per update, full vs delta, on real model pairs |

Order puts the statistics (0.1) first because it is the deep-dive and needs nothing else.

## What is laptop-only

0.1 to 0.4 and 0.6. Real device types (0.5) need the Jetson and the Pico 2; until then every device is emulated and the results must say "emulated".

## Risks and unknowns

- **The sequential test choice.** Candidates include a sequential probability ratio test on paired per-frame differences and an always-valid confidence sequence. Per-frame outcomes are correlated (consecutive frames), which breaks the independence the textbook tests assume; the first deliverable is a test that stays valid under that correlation, and showing the naive one does not.
- **Emulated fleets prove the control logic, not the devices.** Latency, flaky radio and thermal throttling are real-device properties (this laptop's GPU already shows thermal collapse in long runs).
- **Signing tools.** Sigstore and cosign need network services and an identity; an offline-capable path (a local key, then keyless later) is likely needed for an air-gapped story. Not yet checked what each requires.
- **Overlap.** IronBoot already covers signed A/B images for firmware; FleetForge must add something (model-level metrics, canary logic, shadow evaluation), not repeat it.
- **Scope.** Terraform, k3s and a cloud control plane in the source plan are deployment detail; a single-node control plane is enough to answer the question.

## Deep-dive part

The sequential test, with its validity under correlated frames shown by simulation, and its frame count against a fixed-sample test at equal error rates.
