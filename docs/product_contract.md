# Unclaw Product Contract

## Purpose

Unclaw is a local-first AI runtime for autonomous assistance on a user's own machine, built around local models, bounded tool use, and inspectable execution.

## Product Identity

- Unclaw is a local-first autonomous agent runtime.
- It is not a cloud assistant clone.
- It is not a deterministic automation workflow disguised as an agent.
- It is not a conversation-quality-first product.
- It is an execution-first local AI product.

## End-State Vision

The intended destination is a truly capable mainstream local AI agent product for personal computing: usable by ordinary people, trustworthy on personal data, effective on real tasks, and powered by local execution rather than cloud dependence.

## Current Truth Boundary

All project-facing claims must match the repository as it exists today. Current descriptions may reference the shipped local runtime, local model profiles, terminal and Telegram interfaces, grounded web search, local persistence, local traces, and the bounded agent loop that is available only on compatible profiles.

Unclaw must not be described as fully autonomous universal AI. It must not be described as general-public ready unless the repository actually supports that standard in interfaces, reliability, safety, and usability. Product truth outranks marketing ambition.

## Core Product Promise

- Local execution is the default and the center of the product.
- Private data stays under local control by default.
- The runtime can perform real tool use, real file operations, and real automation steps within explicit safety boundaries.
- The core runtime does not depend on cloud inference or a cloud control plane.

## Target Users

Current target users:

- technically comfortable users
- developers
- privacy-conscious power users
- local AI enthusiasts working on consumer hardware

Later target users:

- broader personal-computing users who want a practical local AI agent without cloud dependence
- mainstream users once the product reaches the required level of usability, reliability, and safety

## Non-Goals

Unclaw must not become:

- a cloud-dependent product
- a prompt-router full of language-specific regex rules
- a fake agent implemented by deterministic flowcharts
- a chat-quality-first benchmark-chasing product
- an opaque black-box system with hidden behavior
- a framework-heavy orchestration stack that hides simple runtime behavior
- a product that overclaims autonomy beyond what the repository actually ships

## Architectural Constraints

- The core runtime must remain local-only.
- Project artifacts, docs, configs, and prompts must be written in English only.
- Runtime behavior must remain multilingual and universal rather than FR-only or EN-only in its core assumptions.
- The design center is small and medium local models on mainstream consumer hardware.
- The agentic core must stay model-driven rather than being replaced by deterministic orchestration.
- Deterministic safety guardrails remain mandatory.
- The architecture must stay modular, auditable, and scalable without requiring heavyweight framework adoption.

## Model Reality

Unclaw must be designed primarily for local models in the 4B-14B range. Smaller models may require compensation through prompt design, better tool contracts, tighter context shaping, or clearer runtime guidance, but that compensation must not harden into permanent deterministic core architecture. Large-model-only assumptions are forbidden. Larger local models may be optional headroom, not the basis of the product contract.

## Interface Reality

Current shipped interfaces are the terminal runtime and the Telegram channel. A future chat-style GUI or simpler mainstream interface is a product direction, not a justification for distorting the runtime architecture into a chat-first system now. Interface evolution must serve the local-first agent runtime rather than redefine it.

## Public Claims Rules

Claims the team may make only when true:

- that a capability is shipped and works in the repository
- that a workflow is autonomous by default
- that an interface is ready for broader users
- that a memory, automation, or tool capability is part of the normal product path

Claims the team must not make before they are true:

- that Unclaw is fully autonomous universal AI
- that Unclaw is already a general-public-ready product
- that the agent core is free of command-heavy fallbacks if that is not true
- that a capability is model-driven by default when it still depends mainly on manual commands or narrow deterministic scaffolding

## Feature Prioritization Filter

Every new feature must strengthen at least one of these outcomes without breaking the local-first agent core:

- stronger autonomous local execution
- stronger user trust through safety, transparency, or privacy
- stronger usability on the path from power-user tooling toward mainstream adoption

Features that mainly add surface area, cloud dependence, hidden orchestration, or benchmark-friendly conversation polish without improving execution should be deprioritized.

## Final Product Sentence

Unclaw is a local-first, execution-first agent runtime that must grow toward mainstream usability without surrendering local control, architectural honesty, or model-driven agency.
