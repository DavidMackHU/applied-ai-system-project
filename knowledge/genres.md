# Genre Guide

Each section describes one genre. Structured lines (`Key: value`) are read by the retriever; the prose is searchable context.
`Related genres` weights (0 to 1) say how close another genre usually sounds. They let the recommender give partial credit instead of all-or-nothing genre matching.

## pop
Aliases: pop music, top 40, chart pop
Related genres: indie pop 0.8, k-pop 0.8, city pop 0.6, edm 0.5, house 0.4, r&b 0.4, latin 0.4
Energy: 0.6-0.9
Moods: happy, euphoric, uplifting
Mainstream, catchy, hook-driven songs with polished production. Good for singing along, parties, and upbeat moods.

## indie pop
Aliases: indie, bedroom pop
Related genres: pop 0.8, indie folk 0.5, synthwave 0.4, city pop 0.5
Energy: 0.5-0.8
Moods: happy, uplifting, nostalgic
Pop melodies with a looser, more handmade feel. Sunny guitars and dreamy synths.

## k-pop
Aliases: kpop, korean pop
Related genres: pop 0.9, edm 0.6, house 0.5, hip hop 0.4, r&b 0.4
Energy: 0.7-0.95
Moods: happy, euphoric, confident
Korean pop: high-gloss production, dance-heavy choruses, and genre mixing of pop, electronic dance, and hip hop.

## city pop
Aliases: japanese city pop
Related genres: synthwave 0.7, pop 0.6, indie pop 0.5, soul 0.4
Energy: 0.55-0.75
Moods: nostalgic, happy, moody
Glossy late-1970s and 1980s Japanese pop with funk bass and jazzy chords. Sounds like driving through a neon city at night.

## lofi
Aliases: lo-fi, lofi hip hop, chillhop, study beats
Related genres: jazz 0.5, ambient 0.6, hip hop 0.4, classical 0.3
Energy: 0.25-0.5
Moods: chill, focused, relaxed
Dusty, mellow beats with soft piano and vinyl crackle. The default soundtrack for studying, homework, and rainy afternoons.

## ambient
Aliases: atmospheric, soundscape
Related genres: lofi 0.6, classical 0.6, synthwave 0.3
Energy: 0.1-0.35
Moods: chill, relaxed, peaceful
Slow, textured music without strong beats. Good for sleep, meditation, and deep focus.

## classical
Aliases: orchestral, piano, symphony
Related genres: ambient 0.6, jazz 0.4, lofi 0.3
Energy: 0.1-0.5
Moods: melancholic, peaceful, focused
Orchestral and solo piano works. Calm pieces suit study and sleep; dramatic pieces can be intense.

## jazz
Aliases: smooth jazz, bebop
Related genres: blues 0.6, soul 0.6, lofi 0.5, r&b 0.4, classical 0.4
Energy: 0.3-0.6
Moods: relaxed, chill, romantic
Swing, improvisation, and warm horns. Coffee shops, dinners, and slow evenings.

## blues
Aliases: delta blues
Related genres: jazz 0.6, soul 0.6, rock 0.5, country 0.4
Energy: 0.3-0.6
Moods: sad, melancholic, soulful
Guitar-driven songs about hard times. Emotional, raw, and slow-burning.

## soul
Aliases: motown, neo soul
Related genres: r&b 0.8, jazz 0.6, blues 0.6, hip hop 0.3
Energy: 0.4-0.7
Moods: uplifting, romantic, happy
Gospel-rooted vocals, warm grooves, and big feelings.

## r&b
Aliases: rnb, rhythm and blues
Related genres: soul 0.8, hip hop 0.6, pop 0.4, latin 0.4
Energy: 0.4-0.7
Moods: romantic, chill, confident
Smooth vocals over laid-back grooves. Date nights and late evenings.

## hip hop
Aliases: hip-hop, rap
Related genres: trap 0.8, r&b 0.6, lofi 0.4, soul 0.3
Energy: 0.6-0.9
Moods: confident, intense, happy
Rhythmic vocals over beats. Swagger, storytelling, and workout energy.

## trap
Aliases: trap music
Related genres: hip hop 0.8, edm 0.4
Energy: 0.75-0.95
Moods: intense, confident, angry
Heavy 808 bass and rapid hi-hats. Aggressive and driving.

## rock
Aliases: rock music, alt rock, alternative
Related genres: punk 0.7, metal 0.6, blues 0.4, indie pop 0.3
Energy: 0.7-0.95
Moods: intense, angry, confident
Loud guitars and live drums. Workouts, road trips, and letting off steam.

## punk
Aliases: punk rock, pop punk
Related genres: rock 0.7, metal 0.5
Energy: 0.85-1.0
Moods: angry, intense, rebellious
Fast, short, loud, rebellious songs.

## metal
Aliases: heavy metal, metalcore
Related genres: rock 0.6, punk 0.5
Energy: 0.85-1.0
Moods: angry, intense, dark
Distorted guitars, pounding drums, and aggressive vocals.

## edm
Aliases: electronic, electronic dance music, rave
Related genres: house 0.8, k-pop 0.5, pop 0.5, trap 0.4, synthwave 0.4
Energy: 0.8-1.0
Moods: euphoric, happy, intense
Big drops and festival energy. Parties, clubs, and high-intensity workouts.

## house
Aliases: deep house
Related genres: edm 0.8, pop 0.4, soul 0.3
Energy: 0.7-0.9
Moods: euphoric, happy
Four-on-the-floor dance grooves for long nights out.

## synthwave
Aliases: retrowave, outrun, synth
Related genres: city pop 0.7, edm 0.4, indie pop 0.4, ambient 0.3
Energy: 0.5-0.8
Moods: moody, nostalgic, dreamy
1980s-inspired synthesizers and drum machines. Night drives and neon nostalgia.

## folk
Aliases: acoustic folk, singer-songwriter
Related genres: indie folk 0.9, country 0.6, blues 0.3
Energy: 0.2-0.5
Moods: sad, melancholic, tender
Acoustic guitar and honest storytelling. Quiet, intimate, and reflective.

## indie folk
Aliases: folk pop
Related genres: folk 0.9, indie pop 0.5, country 0.4
Energy: 0.25-0.55
Moods: sad, melancholic, nostalgic
Folk songwriting with indie textures. Rainy windows and long walks.

## country
Aliases: americana, country music
Related genres: folk 0.6, indie folk 0.4, blues 0.4, rock 0.3
Energy: 0.4-0.7
Moods: nostalgic, sad, happy
Storytelling songs about home, heartbreak, and the open road.

## reggae
Aliases: dancehall, roots reggae
Related genres: latin 0.4, soul 0.3, hip hop 0.3
Energy: 0.45-0.7
Moods: relaxed, happy, uplifting
Off-beat guitar skanks and deep bass. Sunny, easygoing, and warm.

## latin
Aliases: reggaeton, salsa, latin pop
Related genres: pop 0.5, reggae 0.4, r&b 0.4, house 0.3
Energy: 0.6-0.9
Moods: romantic, happy, festive
Latin rhythms from reggaeton to salsa. Dancing, celebrations, and summer nights.
