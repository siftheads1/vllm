# MPR Implementation Plan

This directory tracks the Mixed-Precision Recovery implementation plan,
milestone progress, reference analyses, and integration notes.

## Collaboration Rules

The project-wide collaboration and execution rules live in:

- `project_plan.md`

Before implementation changes, especially in scoring, recall policy,
request/block ownership, serving behavior, or ArkVale kernel integration,
review those rules and confirm the concrete plan with the user.

## Directory Layout

- `project_plan.md`: overall architecture, scope, milestones, and rules
- `backlog.md`: follow-up tasks and cleanup items
- `integration/`: vLLM integration notes shared across milestones
- `references/`: external/reference-system analysis reports
- `milestones/`: milestone-specific plans, logs, results, and reports

## Milestones

- `milestones/milestone_0/`: Code Cartography
- `milestones/milestone_1/`: Score-Only Prototype
- `milestones/milestone_2/`: CPU fp16 Backup
- `milestones/milestone_3/`: Full-Precision Recovery
- `milestones/milestone_4/`: Mixed-Precision Recovery
- `milestones/milestone_4_5/`: INT4 Packed Recovery Tier Integration
- `milestones/milestone_5/`: Recovery Cleanup and Optimization
