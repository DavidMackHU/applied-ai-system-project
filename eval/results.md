# VibeFinder AI Evaluation Report

## Case results (full system)

```
+---------------+-----------+-------+-------+--------+-----------------------------+
| Case          | Status    | Conf. | Tries | Result | Checks                      |
+===============+===========+=======+=======+========+=============================+
| studying      | ok        | 0.83  | 2     | PASS   | ok status: ok               |
|               |           |       |       |        | ok top_genre: lofi          |
|               |           |       |       |        | ok energy_range: 0.33       |
|               |           |       |       |        | ok min_confidence: 0.83     |
|               |           |       |       |        | ok unique_artists: 5/5      |
+---------------+-----------+-------+-------+--------+-----------------------------+
| gym_pop       | ok        | 0.82  | 2     | PASS   | ok status: ok               |
|               |           |       |       |        | ok genres_in_top_k: 3/5     |
|               |           |       |       |        | ok energy_range: 0.86       |
+---------------+-----------+-------+-------+--------+-----------------------------+
| like_song     | ok        | 0.66  | 2     | PASS   | ok status: ok               |
|               |           |       |       |        | ok moods_in_top_k: 5/5      |
|               |           |       |       |        | ok energy_range: 0.90       |
|               |           |       |       |        | ok excludes_title: excluded |
+---------------+-----------+-------+-------+--------+-----------------------------+
| implicit_sad  | off_topic | -     | 0     | FAIL   | XX status: off_topic        |
|               |           |       |       |        | XX moods_in_top_k: 0/0      |
+---------------+-----------+-------+-------+--------+-----------------------------+
| unknown_genre | ok        | 0.85  | 2     | PASS   | ok status: ok               |
|               |           |       |       |        | ok genres_in_top_k: 4/5     |
|               |           |       |       |        | ok min_confidence: 0.85     |
+---------------+-----------+-------+-------+--------+-----------------------------+
| road_trip     | ok        | 0.84  | 1     | PASS   | ok status: ok               |
|               |           |       |       |        | ok moods_in_top_k: 5/5      |
|               |           |       |       |        | ok energy_range: 0.74       |
+---------------+-----------+-------+-------+--------+-----------------------------+
| retro         | ok        | 0.65  | 1     | PASS   | ok status: ok               |
|               |           |       |       |        | ok top_genre: synthwave     |
+---------------+-----------+-------+-------+--------+-----------------------------+
| sleep         | ok        | 0.76  | 1     | PASS   | ok status: ok               |
|               |           |       |       |        | ok energy_range: 0.33       |
+---------------+-----------+-------+-------+--------+-----------------------------+
| angry_metal   | ok        | 0.80  | 1     | PASS   | ok status: ok               |
|               |           |       |       |        | ok top_genre: metal         |
|               |           |       |       |        | ok moods_in_top_k: 5/5      |
+---------------+-----------+-------+-------+--------+-----------------------------+
| date_night    | ok        | 0.76  | 1     | PASS   | ok status: ok               |
|               |           |       |       |        | ok top_genre: r&b           |
|               |           |       |       |        | ok moods_in_top_k: 2/5      |
+---------------+-----------+-------+-------+--------+-----------------------------+
| conflict      | ok        | 0.60  | 3     | PASS   | ok status: ok               |
|               |           |       |       |        | ok max_confidence: 0.60     |
+---------------+-----------+-------+-------+--------+-----------------------------+
| empty         | blocked   | -     | 0     | PASS   | ok status: blocked          |
+---------------+-----------+-------+-------+--------+-----------------------------+
| too_long      | blocked   | -     | 0     | PASS   | ok status: blocked          |
+---------------+-----------+-------+-------+--------+-----------------------------+
| injection     | blocked   | -     | 0     | PASS   | ok status: blocked          |
+---------------+-----------+-------+-------+--------+-----------------------------+
| off_topic     | off_topic | -     | 0     | PASS   | ok status: off_topic        |
+---------------+-----------+-------+-------+--------+-----------------------------+
| wellbeing     | ok        | 0.88  | 1     | PASS   | ok status: ok               |
|               |           |       |       |        | ok energy_range: 0.36       |
|               |           |       |       |        | ok avoid_moods: none        |
|               |           |       |       |        | ok safety_note: shown       |
+---------------+-----------+-------+-------+--------+-----------------------------+
```

## Summary by configuration

| Configuration | Cases passed | Avg confidence (ok cases) | Failed cases |
|---|---|---|---|
| full | 15/16 | 0.77 | implicit_sad |
| no_rag | 12/16 | 0.67 | studying, implicit_sad, unknown_genre, road_trip |
| no_classifier | 15/16 | 0.77 | implicit_sad |

## Mood classifier vs. keyword baseline (holdout set)

Holdout: 40 hand-written requests, 5 per mood, most without a mood keyword.

| Method | Correct | Accuracy |
|---|---|---|
| Keyword baseline | 3/40 | 8% |
| Naive Bayes classifier (p >= 0.5) | 20/40 | 50% |
| Hybrid (keyword, then classifier) | 21/40 | 52% |

Classifier answered 65% of requests and was right 77% of the time when it answered.

Hybrid misses:

- "feeling on top of the world after my promotion": expected happy, got euphoric (classifier said sad at p=0.29)
- "today was amazing and I want to dance around my kitchen": expected happy, got angry (classifier said angry at p=0.54)
- "we won the championship tonight": expected happy, got romantic (classifier said romantic at p=0.77)
- "I'm grinning ear to ear, my sister had her baby": expected happy, got nothing (classifier said sad at p=0.28)
- "lazy afternoon on the porch with a lemonade": expected chill, got nothing (classifier said romantic at p=0.36)
- "just got home and want to kick my feet up": expected chill, got nothing (classifier said happy at p=0.37)
- "I want something easygoing while I fold laundry": expected chill, got relaxed (classifier said chill at p=0.66)
- "cramming flashcards for the mcat": expected focused, got nothing (classifier said focused at p=0.50)
- "spreadsheet work for the next four hours": expected focused, got nothing (classifier said focused at p=0.46)
- "I've been crying since the funeral": expected sad, got nothing (classifier said sad at p=0.38)
- "I feel lonely in this new city": expected sad, got nothing (classifier said happy at p=0.48)
- "nothing is going right and I feel hopeless": expected sad, got happy (classifier said happy at p=0.98)
- "final sprint of my half marathon": expected intense, got nothing (classifier said intense at p=0.36)
- "warming up before the big fight": expected intense, got nothing (classifier said angry at p=0.33)
- "someone cut me off in traffic and I want to yell": expected angry, got nothing (classifier said chill at p=0.47)
- "I'm seething after that argument": expected angry, got nothing (classifier said chill at p=0.35)
- "cooking pasta for my girlfriend tonight": expected romantic, got nothing (classifier said sad at p=0.36)
- "slow dance at my wedding": expected romantic, got chill (classifier said chill at p=0.57)
- "something that takes me back to the nineties": expected nostalgic, got intense (classifier said intense at p=0.53)
