# Experiment Results

| Agent Version                                                                                                                                                    | Music | Restaurants | Games | Books |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|-------------|-------|-------|
| [RL Based matcher (Adaptation Pipeline Only)](#agents/AdaptationPipeline.ipynb)                                                                                  |       |             |       |       |
| [RL Based Matcher ( Matcher + Blocker)](#agents/AdaptationPipeline_blocking_matching_extension_Final_Blocker_Matcher.ipynb)                                      |       |             |       |       |
| [RL Matcher ( Matcher + Blocker + Evaluation Reasoning Node)](#agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)                      |       |             |       |       |
| [ML Matcher ( Matcher + Blocker + Evaluation Reasoning Node)](#agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)                      |       |             |       |       |
| [RL Based Matcher ( Matcher + Blocker + Evaluation Reasoning Node +  CN + DB Tool)](#agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)|       |             |       |       |
| [ML Based Matcher ( Matcher + Blocker + Evaluation Reasoning Node +  CN + DB Tool)](#TODO)                                                                       |       |             |       |       |

## USECASE 1: Music Dataset

### Used Blocking Configuration:

| Datasets                                                | Strategy                                                     | Parameters                              | PC     |
|---------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------|--------|
| discogs <-> lastfm                                      | Embedding Blocker (`name`, `artist`)                         | top_k: 20                               | 0.96   |
| discogs <-> musicbrainz                                 | Embedding Blocker (`name`, `artist`, `release-date`)         | top_k: 20                               | 0.95   |
| musicbrainz <-> lastfm                                  | Embedding Blocker (`name`, `artist`)                         | top_k: 40                               | 0.949  |

```json
blocking_strategies": {
    "discogs_lastfm": {
      "strategy": "semantic_similarity",
      "columns": [
        "name",
        "artist"
      ],
      "params": {
        "top_k": 20
      },
      "pair_completeness": 0.9642857142857143,
      "num_candidates": 449753,
      "is_acceptable": true
    },
    "discogs_musicbrainz": {
      "strategy": "semantic_similarity",
      "columns": [
        "name",
        "artist",
        "release-date"
      ],
      "params": {
        "top_k": 20
      },
      "pair_completeness": 0.95,
      "num_candidates": 450205,
      "is_acceptable": true
    },
    "musicbrainz_lastfm": {
      "strategy": "semantic_similarity",
      "columns": [
        "name",
        "artist"
      ],
      "params": {
        "top_k": 40
      },
      "pair_completeness": 0.9493670886075949,
      "num_candidates": 186225,
      "is_acceptable": true
    }
  },
  "id_columns": {
    "discogs": "id",
    "lastfm": "id",
    "musicbrainz": "id"
  }
}
```

### Matcher Configurations

#### Rule Based

| Datasets                                                | F1     |
|---------------------------------------------------------|--------|
| discogs <-> lastfm                                      | 0.8247 |
| discogs <-> musicbrainz                                 | 0.7597 |
| musicbrainz <-> lastfm                                  | 0.7559 |

```json
{
  "id_columns": {
    "discogs": "id",
    "lastfm": "id",
    "musicbrainz": "id"
  },
  "matching_strategies": {
    "discogs_lastfm": {
      "comparators": [
        {
          "type": "string",
          "column": "artist",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "name",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "duration",
          "max_difference": 15.0
        },
        {
          "type": "string",
          "column": "tracks_track_name",
          "similarity_function": "jaccard",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        }
      ],
      "weights": [
        0.3,
        0.35,
        0.15,
        0.2
      ],
      "threshold": 0.6,
      "f1": 0.8247422680412371
    },
    "discogs_musicbrainz": {
      "comparators": [
        {
          "type": "string",
          "column": "artist",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "date",
          "column": "release-date",
          "max_days_difference": 730
        },
        {
          "type": "numeric",
          "column": "duration",
          "max_difference": 10.0
        },
        {
          "type": "string",
          "column": "tracks_track_name",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        },
        {
          "type": "string",
          "column": "release-country",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        }
      ],
      "weights": [
        0.22,
        0.28,
        0.12,
        0.08,
        0.18,
        0.12
      ],
      "threshold": 0.68,
      "f1": 0.7596899224806202
    },
    "musicbrainz_lastfm": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "artist",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "tracks_track_name",
          "similarity_function": "jaccard",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        },
        {
          "type": "string",
          "column": "duration",
          "similarity_function": "levenshtein",
          "preprocess": "strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "tracks_track_duration",
          "similarity_function": "jaccard",
          "preprocess": "strip",
          "list_strategy": "set_jaccard"
        }
      ],
      "weights": [
        0.35,
        0.25,
        0.2,
        0.1,
        0.1
      ],
      "threshold": 0.6,
      "f1": 0.7559055118110236
    }
  }
}
```

#### Machine Learning

```json

```

## Old Results

| Use Case      | Approach                                    | Overall Accuracy (RB) | Overall Accuracy (ML) |
|---------------|---------------------------------------------|-----------------------------|-----------------------------|
| Music         | [AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb](#agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)  | TBD                  | TBD                  |
| Restaurant (Test sets currently do not fit datasets)    | [AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb](#agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)  | 26.408%                  | 28.661%                  |