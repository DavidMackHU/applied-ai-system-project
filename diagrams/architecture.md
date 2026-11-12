# VibeFinder AI: System Architecture

Source of truth: [architecture.mmd](architecture.mmd). This file has the same diagram in a Mermaid code block, so GitHub renders it.

```mermaid
%% VibeFinder AI - system architecture
flowchart TD
    U([Listener request<br/>plain English]) --> G1{Input guardrails<br/>empty / too long /<br/>prompt injection}
    G1 -- blocked --> B([Blocked message])
    G1 -- allowed --> W[Wellbeing check<br/>988 support note +<br/>filters heavy moods]
    W --> P

    subgraph PLAN [1. Plan]
        P[Local parser<br/>genre, mood, energy,<br/>decade, mentions] --> C[Mood classifier<br/>Naive Bayes on<br/>synthetic data]
        P -. optional --llm .-> L[Claude parser<br/>JSON structured output]
        L --> V[LLM output validator<br/>allow-list + clamping]
        C --> PL[Make plan<br/>choose tools]
        V --> PL
    end

    PL -- no music signal --> OT([Off-topic help message])
    PL --> R

    subgraph ACT [2. Act]
        R[Retriever<br/>TF-IDF + phrase match] <--> KB[(Knowledge base<br/>genres.md / moods.md /<br/>activities.md)]
        S[Catalog search<br/>songs / artists named] <--> CSV[(data/songs.csv<br/>30 songs)]
        R --> PR[Profile builder<br/>related genres + moods,<br/>energy targets, citations]
        S --> PR
        PR --> REC[Recommender<br/>score_song + rank_songs]
    end

    REC --> CR

    subgraph CHECK [3. Check and revise]
        CR[Critic<br/>genre / mood / energy fit,<br/>diversity, strength<br/>= confidence 0-1] --> D{High confidence<br/>and no issues?}
        D -- no, attempts left --> RV[Revise strategy<br/>scoring mode or<br/>diversity penalty]
        RV --> REC
    end

    D -- yes / limit reached --> SEL[Select best attempt]
    SEL --> G2{Output guardrails<br/>songs exist in catalog,<br/>scores finite}
    G2 --> OUT([Ranked songs + reasons +<br/>confidence + sources])

    SEL -. every step .-> T[(Trace + logs<br/>logs/vibefinder.log,<br/>logs/traces/*.jsonl)]

    subgraph TESTING [Humans and testing]
        PT[pytest unit tests<br/>39 tests]
        EV[Eval harness<br/>16 cases x 3 configs +<br/>classifier holdout]
        HR[Human review<br/>of traces and outputs]
    end
    EV -. runs .-> U
    HR -. reads .-> T
    HR -. reads .-> OUT
```

## How to read it

- **Input guardrails** run before any AI step. Empty, oversized, and prompt-injection requests are blocked. Self-harm language adds a 988 support note and filters out sad, melancholic, and angry songs.
- **Plan:** the local parser extracts preferences. The Naive Bayes mood classifier fills in a mood when none is named. The optional Claude parser's output passes an allow-list validator. The agent then chooses its tools.
- **Act:** the retriever searches the three knowledge files. Catalog search seeds the profile from any named song. The profile builder merges both, with citations, and the recommender ranks the 30-song catalog.
- **Check and revise:** the critic turns fit metrics into a 0-1 confidence score. Until confidence is high and no issues remain, the agent switches scoring mode or turns on the diversity penalty, for up to 3 attempts.
- **Output guardrails** keep only songs that exist in the catalog and have finite scores.
- **Humans and testing:** every step is written to the log and JSONL traces for human review. pytest and the evaluation harness check the system from outside.
