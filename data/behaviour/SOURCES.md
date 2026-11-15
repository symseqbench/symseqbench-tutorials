# Sources of the behaviour tables

This folder holds four open datasets as sequences of behaviour labels, one table per dataset. Each table was derived from the public files of its source by `scripts/build_behaviour_sequences.py`. No raw file is in this repository. If you use a table, cite its source as given below.

Every table has one row per bout and the columns `individual`, `recording`, `string` (number of the string), `bout` (number of the bout inside the string), `symbol`, `behaviour` (name in the source), `raw_label` and `duration_s` (seconds). For the turtles and the seals a bout of one class can hold several labels of the source; it then has one row per label, all with the same bout number.

## Zebrafish (`zebrafish.csv.gz`)

- What it is: larval zebrafish (Danio rerio), swimming bouts. In the words of the source: 463 freely swimming larval zebrafish were recorded from 30 min to 3 h (from the article of the dataset).
- Dataset: Sridhar G, Costa A (2024). Dataset for Uncovering multiscale structure in the variability of larval zebrafish navigation V2. Zenodo. https://doi.org/10.5281/zenodo.13605471
- Article: Sridhar G, Vergassola M, Marques JC, Orger MB, Costa AC, Wyart C (2024). Uncovering multiscale structure in the variability of larval zebrafish navigation. Proceedings of the National Academy of Sciences 121(47), e2410254121. https://doi.org/10.1073/pnas.2410254121
- Article: Marques JC, Lackner S, Félix R, Orger MB (2018). Structure of the zebrafish locomotor repertoire revealed with unsupervised behavioral clustering. Current Biology 28(2), 181-195. https://doi.org/10.1016/j.cub.2017.12.002 (the thirteen bout types)
- Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- In the table: 20 strings (all bouts of one fish) from 20 animals, 47901 bouts, 13 behaviours; strings of 590 to 6022 bouts, median 2039.
- Changes made: Only the bout type and the start and end frame of each bout were read. The thirteen bout types carry the names of the source file (for example RT, Slow1, J_turn); they are described in Marques et al. (2018). Bout durations are end frame minus start frame, divided by the frame rate of 700 per second. Every bout is one symbol with its own duration; consecutive bouts of the same type were not merged. A seeded random subset of the fish was kept.

## Green turtle (`turtle.csv.gz`)

- What it is: free-ranging immature green turtles (Chelonia mydas), Martinique. In the words of the source: The collected dataset contains the acceleration, angular velocity and depth profile of thirteen free-ranging immature green turtles. The video footage and the visualisation of the multi-sensors data allowed us to identify 46 behaviours with a specific signal (from the README of the dataset).
- Dataset: Jeantet L, Planas-Bielsa V, Benhamou S, Geiger S, Martin J, Siegwalt F, Lelong P, Gresser J, Etienne D, Hiélard G, Arque A, Regis S, Lecerf N, Frouin C, Benhalilou A, Murgale C, Maillet T, Andreani L, Campistron G, Delvaux H, Guyon C, Richard S, Lefebvre F, Aubert N, Habold C, Le Maho Y, Chevallier D (2020). Raw acceleration, gyroscope and depth profiles associated with the observed behaviours of free-ranging immature green turtles in Martinique. Dryad. https://doi.org/10.5061/dryad.hhmgqnkd9
- Article: Jeantet L et al. (2020). Behavioural inference from signal processing using animal-borne multi-sensor loggers: a novel solution to extend the knowledge of sea turtle ecology. Royal Society Open Science 7(5), 200139. https://doi.org/10.1098/rsos.200139
- Article: Hoffman B et al. (2024). A benchmark for computational analysis of animal behavior, using animal-borne tags. Movement Ecology 12, 78. https://doi.org/10.1186/s40462-024-00511-8 (the behaviour classes; the files were read from the copy in this benchmark, https://doi.org/10.5281/zenodo.7947104)
- Licence: CC0 1.0, https://creativecommons.org/publicdomain/zero/1.0/
- In the table: 148 strings (the bouts between two rests in one recording) from 14 animals, 3455 bouts, 7 behaviours; strings of 10 to 136 bouts, median 15.
- Changes made: Only the tables of observed behaviours were read, in the order of their rows. Each observed label was mapped to its class in the benchmark (breathing, feeding, gliding, resting, scratching, staying at surface, swimming, unknown); the label itself is kept in the column raw_label. Consecutive observations of the same class were merged into one bout. Resting was removed and used to cut each recording into strings. Strings of fewer than ten bouts were removed. A duration is the stop time minus the start time of the source; where that is not positive it is left empty. The animal of each recording is the number given in the benchmark, which counts 14 animals in the 16 recordings where the source describes thirteen turtles.
- The class unknown: kept as a symbol, 404 of 3455 bouts (11.7 %). Strings are not cut there.

## Bengalese finch (`finch.csv.gz`)

- What it is: adult male Bengalese finches (Lonchura striata domestica), song syllables. In the words of the source: Bengalese Finch song repository (title of the dataset). Syllable labels are specific to each bird: the same letter in two birds does not name the same syllable.
- Dataset: Nicholson D, Queen JE, Sober SJ (2017). Bengalese Finch song repository. figshare. https://doi.org/10.6084/m9.figshare.4805749
- Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- In the table: 1893 strings (the syllables of one song file) from 4 animals, 168959 bouts, a bout being one syllable (bl26lb16 445 songs with 20 syllable labels, gr41rd51 500 songs with 24 syllable labels, gy6or6 500 songs with 17 syllable labels, or60yw70 448 songs with 16 syllable labels); strings of 10 to 398 bouts, median 74.
- Changes made: Only the annotation files were read: the syllable labels and their onset and offset times. A duration is offset minus onset. Songs of fewer than ten syllables were removed. At most 500 songs per bird were kept, chosen at random with a fixed seed. The birds are kept apart. The notebook uses the bird with the most annotated songs.

## Fur seals and sea lions (`seal.csv.gz`)

- What it is: captive fur seals and sea lions (four otariid species). In the words of the source: Seals dataset (title in the licence file of the dataset), from the study Seeing It All: Evaluating Supervised Machine Learning Methods for the Classification of Diverse Otariid Behaviours.
- Dataset: Hoffman B, Cusimano M, Baglione V, Canestrari D, Chevallier D, DeSantis DL, Jeantet L, Ladds MA, Maekawa T, Mata-Silva V, Moreno-González V, Trapote E, Vainio O, Vehkaoja A, Yoda K, Zacarian K, Friedlaender A, Rutz C (2023). Bio-logger Ethogram Benchmark: A benchmark for computational analysis of animal behavior, using animal-borne tags (1.0.0). Zenodo. Seals dataset by Ladds MA, Thompson AP, Slip DJ, Hocking DP, Harcourt RG. https://doi.org/10.5281/zenodo.7947104
- Article: Ladds MA, Thompson AP, Slip DJ, Hocking DP, Harcourt RG (2016). Seeing it all: evaluating supervised machine learning methods for the classification of diverse otariid behaviours. PLOS ONE 11(12), e0166898. https://doi.org/10.1371/journal.pone.0166898
- Article: Hoffman B et al. (2024). A benchmark for computational analysis of animal behavior, using animal-borne tags. Movement Ecology 12, 78. https://doi.org/10.1186/s40462-024-00511-8
- Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- In the table: 100 strings (the bouts between two rests in one recording) from 12 animals, 2386 bouts, 4 behaviours; strings of 10 to 159 bouts, median 16.5.
- Changes made: Only the behaviour columns of each recording were read, sample by sample. Each sample was given its class in the benchmark (feeding, grooming, resting, travelling, unknown); the finer label of the source is kept in the column raw_label. Consecutive samples of the same class were merged into one bout. Resting was removed and used to cut each recording into strings. Strings of fewer than ten bouts were removed. A duration is the number of samples divided by the sampling rate of 25 per second.
- The class unknown: kept as a symbol, 442 of 2386 bouts (18.5 %). Strings are not cut there.

## Choices that hold for every table

- A string never spans two recordings, and the rows of a table are in the order of the source file.
- A zebrafish swim bout and a finch syllable are events: each is one symbol, and two in a row of the same type are two symbols. For the turtles and the seals consecutive equal classes are one bout.
- The class unknown of the turtles and the seals (for the seals the label Other of the source) stays a symbol, and strings are not cut there. Its share of the bouts is given with each table above.
- The four finches are four separate sets of strings and are never pooled. The notebook uses the bird with the most annotated songs, gr41rd51.
- The shuffled surrogate of the notebook permutes all bouts of a dataset (for the finch, of one bird) and cuts them into strings of the original lengths. It does not shuffle inside each string.

## Differences from the analysis of the SymSeqBench preprint

The preprocessing follows the script of that analysis: strings cut at rest, at least ten bouts per string, at most 500 strings, seed 42, and for the turtles and the seals bouts by merging consecutive equal classes. It differs in these points.

| Step | In that analysis | Here |
|---|---|---|
| Symbols | Letters are assigned string by string, from the behaviours present in that string. | One alphabet per dataset, built once over all its strings. The name of each behaviour in the source is kept beside its symbol. |
| Finch birds | The songs of the four birds form one set, from which 500 songs are drawn. | Each bird is its own set of strings, with at most 500 songs, because syllable labels are specific to each bird. The notebook uses the bird with the most annotated songs. |
| Zebrafish | All 463 fish. | A random subset of 20 fish (seed 42). The number was set by the file-size and run-time limits of the tutorial before any measure was computed. |
| Consecutive zebrafish bouts of one type | Merged into one bout, with their durations added. | Kept as they are: a swim bout is a discrete event, so two bouts of one type in a row are two symbols, each with its own duration. In the table 14756 of 47901 bouts (30.8 %) have the type of the bout before them. Consecutive equal classes are merged only for the turtles and the seals, where the source is annotated observation by observation or sample by sample and merging is what makes a bout. |
| Behaviour classes of the turtles and the seals | The labels are grouped by the script's own table, with a finer numbered level for the transition graph. | The classes are those published with the benchmark. The label of the source is kept in the column raw_label. The measures use the classes. |
| Mouse open-field dataset | Included. | Not used. |
| Recordings of one animal | The recordings of one animal are joined and their bouts are ordered by the time inside each recording. | Each recording is kept apart, with its bouts in the order of the source file. A string never spans two recordings. |
| Durations | Seals: last time stamp minus first time stamp of the bout. Turtles: stop minus start, also where that is not positive. | Seals: number of samples divided by 25 per second. Turtles: stop minus start, left empty where that is not positive. |
| Random draws | One random generator, seeded with 42, serves all datasets in turn. | Each draw uses its own generator seeded with 42. |

The values computed from these tables are therefore not expected to equal those of the preprint, which also used an earlier version of the library.
