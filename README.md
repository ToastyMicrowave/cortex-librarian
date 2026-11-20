# Cortex Librarian

A local-LLM note librarian for an Obsidian vault, plus the evaluation harness used to decide whether it's good enough to trust.

It reads new notes in the inbox and proposes a one-line summary, tags, links to existing wiki pages and a destination folder. It runs entirely on-device (Ollama, `qwen3.5:9b`, on a 16GB M4 MacBook Pro) and **changes nothing**. It writes a proposals file for me to review.

This is one component of **Cortex**, my personal second-brain system. The vault itself is private and never pushed anywhere. This repo holds the code and a **fictional** demo vault so the evals can be run and read.

## Status (honest)

Cortex is designed as three agents: this Librarian (local model), a Writer (Claude Code, headless) and an Auditor (Codex CLI). They coordinate only through files in the vault and are run by a launchd shell script.

| Phase | State |
|---|---|
| Vault, git, local model | done |
| **Librarian, proposal mode (this repo)** | **built and evaluated, v1.2.1** |
| Librarian, apply mode | not started: blocked on the eval gate below |
| Writer + Auditor | designed, not built |
| Automation (launchd) | not built |

## Design: the model is a function, not an agent

The model has no tools and no loop. The script reads each note, sends one request, gets one JSON object back, checks it, and writes a proposal. Why:

- **Small models are unreliable at tool calling.** One request with one structured answer leaves nothing to get wrong except the answer itself.
- **Safety.** A note saying "delete everything" can't delete anything, because there is nothing to call. The worst case is a bad proposal, and I review every one.
- **Debuggability.** One note means one request and one response, with no multi-step trace to reconstruct.

## What enforces the rules

The main lesson from the evals: **rules I only asked for in the prompt broke. Rules enforced in code held.** So the important rules live in code:

- **Constrained decoding.** Ollama structured outputs turn a JSON schema into a grammar, and tokens that would break it are masked before sampling. Destinations and link targets can only be real folders and pages. That guarantees shape, not sense, so everything is validated again in code.
- **Name links are found by code, not the model.** If a page's title or alias is written in the note, code links it. The model only suggests "topic" links, and each must include a quote that code checks is really in the note.
- **Tags are a true/false checklist, not pick-from-a-list.** With an array the model started a tag the list didn't have, then got forced into the nearest one that did ("travel" on a handstand note).
- **Lookalike guard (v1.2.1).** A model link is dropped when the note has a word one letter off the page name and never says the real name (SGTC vs SSTC in my own notes; "rower" vs Rover in the demo).
- **Prompt-injection tripwire.** Notes with text aimed at an AI are quarantined and never sent to the model. There is no free-text output field that becomes an action.
- **Private-note handling.** Notes in `90-private/` get their output only inside `90-private/`. Logs and commits carry counts only. Inbox notes that look private (health, logins, account numbers) are flagged and their summary withheld.
- **Ungrounded-summary flag.** A summary that mentions a number, version or acronym the note doesn't contain gets flagged for me to read.
- **Context-window handling.** Ollama's default context on a <24GB Mac is 4096 tokens, and it silently cuts overlong prompts *from the start*, where the instructions are. The script requests 8192, estimates size pessimistically (3 chars/token) and truncates the note itself, saying so.
- **Write allowlist and symlink checks.** The script refuses to write anywhere except its own files, and won't follow symlinks out of the vault.
- **Code lives outside the vault.** The Writer agent can edit vault files, so if this code (which runs with my user permissions) lived there, an agent could change it. That would be privilege escalation.

## Results (private set, 18 real inbox notes, qwen3.5:9b)

These numbers come from my real notes, which aren't published. Only the counts are shown.

| | v1.1 | v1.2 | Codex-written candidate |
|---|---|---|---|
| destination good / also fine | 17/18 | 17/18 | 16/18 |
| link precision | 82% | 88% (15/17) | 100% (5/5) |
| link recall | 64% | 100% (14/14) | 38% (5/13) |
| tag precision | 84% | 96% | 100% |
| tag recall | 100% | 100% | 65% |
| injection test | FAIL | PASS | PASS |
| private-ish note flagged | 0/1 | 1/1 | 0/1 |
| median time per note | ~3.5 s | 6.9 s | n/a |
| gate for apply mode | fails 5 of 7 | fails 1 (link precision) | fails 3 |

- **The Codex candidate** was an alternative Librarian I had Codex write as a comparison. It scored high precision because it barely linked anything, so it isn't useful as a librarian.
- **Caveats, which matter more than the numbers:**
  - 18 notes is tiny. With ~17 links, one link moves precision by ~6 points.
  - These 18 notes have been tuned against, so they're now effectively training data. A held-out set is the next step.
  - My own approval rate was far below the answer key's: reviewing v1.1 by hand, I'd approve only 3/18 proposals unchanged. The automated score measures the answer key, not whether the output is good enough.
  - v1.2.1 is mock-tested only. The live rerun is pending.

## The demo vault

`demo-vault/` is **fictional**: invented notes, people and places. Each of its 18 notes mirrors one private test note and keeps that note's purpose:

- several notes about one project (the Rover)
- a lookalike word ("rower")
- a note named after an existing page (`sam.md`)
- a private-ish note (`visit home.md`, mentions a GP appointment)
- an adversarial prompt-injection note (`zz-injection-test.md`)
- a list of unrelated items that should stay in the inbox

Scores on it aren't comparable to the private-set numbers above.

Known issue the demo set surfaced: the lookalike guard treats ordinary 4-letter words one letter off a page name as lookalikes ("over" vs Rover). Here that only drops links that were wrong anyway, but it could drop a correct one.

## Running it

Needs macOS or Linux, Python 3.9+ (standard library only) and [Ollama](https://ollama.com) with the model pulled.

```bash
ollama pull qwen3.5:9b
CORTEX_VAULT=./demo-vault python3 evals/eval.py           # run and score; writes nothing in the vault
CORTEX_VAULT=./demo-vault python3 evals/eval.py --save    # also save the report to evals/results/
CORTEX_VAULT=./demo-vault python3 librarian.py --dry-run  # print proposals, write nothing
python3 evals/probe_tags.py "00-inbox/fingerboard.md"     # compare free vs forced tagging (set CORTEX_VAULT too)
```

Without `CORTEX_VAULT`, the vault defaults to `~/Cortex`.

## Who did what

I designed the system with Claude (Anthropic) as a pair-programmer and teacher. Claude wrote most of the code. I made the design decisions, ran and tested everything on my machine, labelled the answer key, reviewed every proposal, and pushed the changes between versions based on the eval results.
