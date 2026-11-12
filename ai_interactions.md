# AI Interactions Log

This log covers two kinds of AI interaction:

1. **Inside the system:** the VibeFinder agent's own intermediate reasoning traces, which are required for the Agentic Workflow Enhancement stretch feature.
2. **Building the system:** how I used an AI coding agent (Claude Code) to develop the project.

The Module 3 version of this log is in git history at commit `5243ef5`.

---

## Part 1: Agent reasoning traces

Every run records each step the agent takes. A step is one of: **guardrail**, **plan** (parse, search knowledge, make a plan), **act** (retrieve knowledge, search the catalog, recommend), **check** (critique, stop decision), **revise** (change strategy), and **final** (select the best attempt). The traces below are verbatim output of `python -m src.app "<request>" --trace`.

The raw machine-readable traces, one JSON object per step plus a final result record, are committed here:

- [20261112-101306_chill-lofi-for-studying.jsonl](assets/sample_runs/traces/20261112-101306_chill-lofi-for-studying.jsonl)
- [20261112-101307_high-energy-pop-for-the-gym.jsonl](assets/sample_runs/traces/20261112-101307_high-energy-pop-for-the-gym.jsonl)
- [20261112-101307_k-pop-dance-hits.jsonl](assets/sample_runs/traces/20261112-101307_k-pop-dance-hits.jsonl)
- [20261112-101307_something-like-storm-runner.jsonl](assets/sample_runs/traces/20261112-101307_something-like-storm-runner.jsonl)
- [20261112-101308_sad-but-high-energy-edm.jsonl](assets/sample_runs/traces/20261112-101308_sad-but-high-energy-edm.jsonl)

Regenerate any of them with:

```bash
python -m src.app "high energy pop for the gym" --trace --save-trace   # writes logs/traces/<time>_<slug>.jsonl
```

### Trace 1: diversity revision ("chill lofi for studying")

The plan chooses three retrieval steps. The first attempt scores high, but the critic flags repeated artists. The agent revises with the diversity penalty and keeps the varied list, because it costs only 0.04 confidence.

```
$ python -m src.app chill\ lofi\ for\ studying --trace --save-trace 

Agent trace:
   1. [guardrail] check_input: request passed input checks
   2. [plan] parse_request: genre=lofi (keyword 'lofi'), mood=chill (keyword)
   3. [plan] search_knowledge: genres.md#lofi (0.93); activities.md#studying (0.72); moods.md#chill (0.66)
   4. [plan] make_plan: retrieve_knowledge: apply 'studying' activity guide | retrieve_knowledge: add related genres from the genre guide | retrieve_knowledge: add related moods from the mood guide | recommend: score and rank the catalog | critique: check fit and confidence | revise: up to 2 alternative strategies if confidence is not high
   5. [act] retrieve_knowledge: activities.md#studying -> energy target 0.35 from range 0.25-0.45
   6. [act] retrieve_knowledge: activities.md#studying -> instrumental tracks preferred
   7. [act] retrieve_knowledge: genres.md#lofi -> related genres: jazz 0.5, ambient 0.6, hip hop 0.4, classical 0.3
   8. [act] retrieve_knowledge: moods.md#chill -> related moods: relaxed 0.9, focused 0.6, peaceful 0.7
   9. [act] recommend: attempt 1 (mode=balanced, diversity=off): Library Rain, Midnight Coding, Quiet Hours, Focus Flow, Spacewalk Thoughts
  10. [check] critique: confidence 0.88 (high); genre_fit=0.92, mood_fit=0.84, energy_fit=0.91, diversity=0.60, strength=0.99; issues: low_diversity
  11. [revise] change_strategy: artists repeat too often, so turn on the diversity penalty: {'diversity': True}
  12. [act] recommend: attempt 2 (mode=balanced, diversity=on): Library Rain, Midnight Coding, Spacewalk Thoughts, Coffee Shop Stories, Moonlit Sonata Redux
  13. [check] critique: confidence 0.83 (high); genre_fit=0.68, mood_fit=0.78, energy_fit=0.88, diversity=1.00, strength=0.99; no issues
  14. [check] stop: confidence is high and no issues remain
  15. [final] select_best: kept attempt 2 of 2 (confidence 0.83, mode=balanced, diversity=on); rule: highest confidence, minus 0.05 if artist diversity is below 0.8
```

### Trace 2: catalog search and an inferred-vs-explicit decision ("high energy pop for the gym")

The workout guide supplies an *inferred* mood (intense). The critic's issue order puts artist diversity ahead of that inferred mood gap, so the agent fixes the repeats and keeps "pop", the genre the user actually typed, at the center.

```
$ python -m src.app high\ energy\ pop\ for\ the\ gym --trace --save-trace 

Agent trace:
   1. [guardrail] check_input: request passed input checks
   2. [plan] parse_request: genre=pop (keyword 'pop'), energy=0.85 (word 'high energy')
   3. [plan] search_knowledge: activities.md#workout (0.87); genres.md#pop (0.70); genres.md#k-pop (0.25); genres.md#edm (0.21); genres.md#indie-pop (0.21)
   4. [plan] make_plan: retrieve_knowledge: apply 'workout' activity guide | retrieve_knowledge: add related genres from the genre guide | retrieve_knowledge: add related moods from the mood guide | recommend: score and rank the catalog | critique: check fit and confidence | revise: up to 2 alternative strategies if confidence is not high
   5. [act] retrieve_knowledge: activities.md#workout -> mood intense (+ related confident, euphoric)
   6. [act] retrieve_knowledge: genres.md#pop -> related genres: indie pop 0.8, k-pop 0.8, city pop 0.6, edm 0.5, house 0.4, r&b 0.4, latin 0.4
   7. [act] retrieve_knowledge: moods.md#intense -> related moods: angry 0.7
   8. [act] recommend: attempt 1 (mode=balanced, diversity=off): Gym Hero, Sunrise City, Neon Heartbeat, Golden Hour Drive, Laser Bloom
   9. [check] critique: confidence 0.79 (medium); genre_fit=0.90, mood_fit=0.34, energy_fit=0.84, diversity=0.60, strength=0.97; issues: low_diversity, mood_gap
  10. [revise] change_strategy: artists repeat too often, so turn on the diversity penalty: {'diversity': True}
  11. [act] recommend: attempt 2 (mode=balanced, diversity=on): Gym Hero, Laser Bloom, Afterglow Avenue, Rooftop Lights, Sunrise City
  12. [check] critique: confidence 0.82 (high); genre_fit=0.74, mood_fit=0.48, energy_fit=0.88, diversity=1.00, strength=0.97; no issues
  13. [check] stop: confidence is high and no issues remain
  14. [final] select_best: kept attempt 2 of 2 (confidence 0.82, mode=balanced, diversity=on); rule: highest confidence, minus 0.05 if artist diversity is below 0.8
```

### Trace 3: a three-attempt decision chain on a conflicting request ("sad but high energy edm")

Each attempt fixes one gap and opens another. Mood-first finds sad songs but loses all the energy. Genre-first recovers the electronic dance and energy fit. The agent hits its attempt limit, keeps the best attempt, and reports medium confidence.

```
$ python -m src.app sad\ but\ high\ energy\ edm --trace --save-trace 

Agent trace:
   1. [guardrail] check_input: request passed input checks
   2. [plan] parse_request: genre=edm (keyword 'edm'), mood=sad (keyword), energy=0.85 (word 'high energy')
   3. [plan] search_knowledge: genres.md#edm (0.84); moods.md#sad (0.71); activities.md#workout (0.20)
   4. [plan] make_plan: retrieve_knowledge: add related genres from the genre guide | retrieve_knowledge: add related moods from the mood guide | recommend: score and rank the catalog | critique: check fit and confidence | revise: up to 2 alternative strategies if confidence is not high
   5. [act] retrieve_knowledge: genres.md#edm -> related genres: house 0.8, k-pop 0.5, pop 0.5, trap 0.4, synthwave 0.4
   6. [act] retrieve_knowledge: moods.md#sad -> related moods: melancholic 0.9, nostalgic 0.4, moody 0.4
   7. [act] recommend: attempt 1 (mode=balanced, diversity=off): Laser Bloom, Afterglow Avenue, Night Drive Loop, Sunrise City, Neon Heartbeat
   8. [check] critique: confidence 0.59 (low); genre_fit=0.64, mood_fit=0.08, energy_fit=0.89, diversity=0.80, strength=0.74; issues: mood_gap
   9. [revise] change_strategy: mood fit is low, so try weighting mood more: {'mode': 'mood_first'}
  10. [act] recommend: attempt 2 (mode=mood_first, diversity=off): Heartbreak Highway, Delta Midnight, Slow Motion Rain, Paper Boats, Moonlit Sonata Redux
  11. [check] critique: confidence 0.50 (low); genre_fit=0.00, mood_fit=0.98, energy_fit=0.01, diversity=1.00, strength=0.76; issues: genre_gap, energy_gap
  12. [revise] change_strategy: genre fit is low, so try weighting genre more: {'mode': 'genre_first'}
  13. [act] recommend: attempt 3 (mode=genre_first, diversity=off): Laser Bloom, Afterglow Avenue, Sunrise City, Neon Heartbeat, Gym Hero
  14. [check] critique: confidence 0.60 (medium); genre_fit=0.66, mood_fit=0.00, energy_fit=0.90, diversity=0.80, strength=0.89; issues: mood_gap
  15. [check] stop: reached the attempt limit
  16. [final] select_best: kept attempt 3 of 3 (confidence 0.60, mode=genre_first, diversity=off); rule: highest confidence, minus 0.05 if artist diversity is below 0.8
```

### Trace 4: a song-seeded profile ("something like Storm Runner")

The catalog-search tool seeds genre, mood, energy, and tags from the named song and excludes it from the results. Storm Runner is the catalog's only rock song, so the critic flags a genre gap. The agent tries genre-first, sees confidence fall from 0.66 to 0.60, and keeps the first attempt. A revision can make things worse, and the agent rejects it when it does.

```
$ python -m src.app something\ like\ Storm\ Runner --trace --save-trace 

Agent trace:
   1. [guardrail] check_input: request passed input checks
   2. [plan] parse_request: nothing explicit found
   3. [plan] make_plan: search_catalog: seed profile from 'Storm Runner' | retrieve_knowledge: add related genres from the genre guide | retrieve_knowledge: add related moods from the mood guide | recommend: score and rank the catalog | critique: check fit and confidence | revise: up to 2 alternative strategies if confidence is not high
   4. [act] search_catalog: seeded from 'Storm Runner' by Voltline; excluding mentioned songs
   5. [act] retrieve_knowledge: genres.md#rock -> related genres: punk 0.7, metal 0.6, blues 0.4, indie pop 0.3
   6. [act] retrieve_knowledge: moods.md#intense -> related moods: angry 0.7, confident 0.5, euphoric 0.3
   7. [act] recommend: attempt 1 (mode=balanced, diversity=off): Broken Sirens, Iron Tempest, Midnight Run Club, Gym Hero, Concrete Crown
   8. [check] critique: confidence 0.66 (medium); genre_fit=0.26, mood_fit=0.78, energy_fit=0.88, diversity=0.80, strength=0.74; issues: genre_gap
   9. [revise] change_strategy: genre fit is low, so try weighting genre more: {'mode': 'genre_first'}
  10. [act] recommend: attempt 2 (mode=genre_first, diversity=off): Broken Sirens, Iron Tempest, Midnight Run Club, Delta Midnight, Rooftop Lights
  11. [check] critique: confidence 0.60 (medium); genre_fit=0.40, mood_fit=0.48, energy_fit=0.70, diversity=1.00, strength=0.71; no issues
  12. [check] stop: no untried revision addresses the remaining issues
  13. [final] select_best: kept attempt 1 of 2 (confidence 0.66, mode=balanced, diversity=off); rule: highest confidence, minus 0.05 if artist diversity is below 0.8
```

---

## Part 2: Building VibeFinder AI with an AI coding agent

**Tool:** Claude Code, an AI coding agent running in VS Code.

**Main prompt:** I attached the Project 4 assignment PDF and asked the agent to complete it, extending my Module 3 music recommender into an applied AI system.

### What the agent did, in order

| Step | What the agent generated or ran | What was verified, and how |
|---|---|---|
| 1. Environment | Cloned the Module 3 repo into `applied-ai-system-project/` and pointed `origin` at the new repo URL | `git log` showed the full Module 3 history. The agent was not permitted to create the public GitHub repo, so that step and the first push are done by hand. |
| 2. Design | Chose an agentic loop as the required feature, with RAG, a specialized classifier, and an eval harness as stretch features. Made Claude parsing optional because no API key was configured and the project must be reproducible. | Design reviewed against the rubric list in the PDF |
| 3. Knowledge base and data | Wrote `knowledge/genres.md`, `moods.md`, and `activities.md`, and added 8 songs | Removed aliases that caused false matches, such as "love" tagging "I love rock" as romantic, and "dance" forcing the electronic dance genre |
| 4. Retriever and scoring | `src/knowledge.py`, plus related-label partial credit in `score_song` | Checked by hand on studying, k-pop, and off-topic queries |
| 5. Classifier | `scripts/generate_synthetic.py` and `src/intent.py` | Spot checks showed it refuses to guess on off-topic text (p = 0.14) |
| 6. Agent, critic, guardrails, CLI | `src/agent.py`, `critic.py`, `guardrails.py`, `parser.py`, `llm.py`, `app.py` | Ran `--demo --trace` and read every trace |
| 7. Evaluation | `eval/cases.json`, `eval/mood_holdout.csv`, `eval/run_eval.py` | The first run passed 13 of 16. Each failure was diagnosed before changing anything (below). |
| 8. Tests and docs | 30 new unit tests (39 total), README, model card, this log | Every number in the docs was checked against `eval/results.md` and the sample runs |

### Issues found by reading traces and eval output

1. **The gym request dropped the user's genre.** This was a real design flaw. The critic treated inferred and explicit preferences equally. Fixed with provenance-weighted confidence and explicit-first revision order.
2. **High confidence stopped revision even with repeated artists.** Fixed by revising while issues remain, and choosing the best attempt with a 0.05 diversity preference.
3. **The wellbeing case still recommended "sad" songs.** This was a responsible-AI gap. Fixed with a heavy-mood filter and a new eval check.
4. **The "too long" eval case was only about 470 characters.** The bug was in the test. Fixed by generating a 600-character request.
5. **A unit test expected three pop songs for k-pop.** The test was stale after fix 2. It now checks related-genre coverage instead.
6. **Implicit sadness ("crying since the funeral") fails.** This is a real limitation, and it was left unfixed on purpose: tuning the training data on holdout misses would leak test answers. It is documented in the model card.

### Design-pattern notes

- **Strategy pattern (from Module 3) reused as the agent's action space.** The critic's revisions are just swaps between existing `ScoringMode` strategies and the diversity flag. So the agent can only take well-understood, tested actions.
- **Tool registry by convention.** Retrieval, catalog search, recommend, and critique are plain functions the agent calls in a planned order. The plan is written to the trace before any tool runs, so the plan and the actual steps can be compared.
