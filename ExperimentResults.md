# Setup

- Model: gpt-5.1

# Experiment Results - New Evaluation Template (Fusion Accuracy)

| Agent Version                                                                                                                              | Music | Restaurants | Games | Books |
|--------------------------------------------------------------------------------------------------------------------------------------------|-------|-------------|-------|-------|
| [RL Based matcher (Adaptation Pipeline Only)](/agents/AdaptationPipeline.ipynb)                                                            |       |             |       |       |
| [RL Based Matcher ( Matcher + Blocker)](/agents/AdaptationPipeline_blocking_matching_extension_Final_Blocker_Matcher.ipynb)                |       |             |       |       |
| [RL Matcher ( Matcher + Blocker + Evaluation Reasoning Node)](/agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)|70.558%|78.612%      |       |       |
| [ML Matcher ( Matcher + Blocker + Evaluation Reasoning Node)](/agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)|51.78% |75.924%      |       |       |
| [RL Based Matcher ( Matcher + Blocker + Evaluation Reasoning Node +  CN + DB Tool)](agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning_ClusterDocTool.ipynb) |54.822%  |63.830% |       |46.429%  |
| [ML Based Matcher ( Matcher + Blocker + Evaluation Reasoning Node +  CN + DB Tool)](agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning_ClusterDocTool.ipynb) |53.807% |81.227% |       |60.000%  |

# Experiment Results - Old Evaluation Template (Fusion Accuracy)

| Agent Version                                                                                                                              | Music | Restaurants | Games | Books |
|--------------------------------------------------------------------------------------------------------------------------------------------|-------|-------------|-------|-------|
| [RL Based matcher (Adaptation Pipeline Only)](/agents/AdaptationPipeline.ipynb)                                                            |17.966%|  51.815%    |**32.773%**|61.429%|
| [RL Based Matcher ( Matcher + Blocker)](/agents/AdaptationPipeline_blocking_matching_extension_Final_Blocker_Matcher.ipynb)                |22.335%|  44.180%    |5.882% |66.429%|
| [RL Matcher ( Matcher + Blocker + Evaluation Reasoning Node)](/agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)|**31.980%**|  43.930%    |10.924%|75.000%|
| [ML Matcher ( Matcher + Blocker + Evaluation Reasoning Node)](/agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning.ipynb)|16.244%|  50.688%    |16.807%|42.143%|
| [RL Based Matcher ( Matcher + Blocker + Evaluation Reasoning Node +  CN + DB Tool)](agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning_ClusterDocTool.ipynb) |31.985%      |49.312%          |20.812% |**77.192%**     |
| [ML Based Matcher ( Matcher + Blocker + Evaluation Reasoning Node +  CN + DB Tool)](agents/AdaptationPipeline_blocking_matching_extension_Final_Reasoning_ClusterDocTool.ipynb) |15.168%       |**55.069%**         |13.706%  |41.295%      |

## Token Usage

| Agent Version                             | Total Tokens Used  |
| ----------------------------------------- | :-------: |
| Agent I: Basic Pipeline Creation (RB)     | 21,452   |
| Agent II: Blocking & Matching Tester (RB) | 44,492  |
| Agent III: Reasoning (RB)                 | 63,272  |
| Agent IV: Reasoning (ML)                  | 75,896  |
| Agent V: Cluster & Documentation (RB)     | 108,138 |
| Agent VI: Cluster & Documentation (ML)    | 142,429 |

## USECASE 1: Music Dataset

| Agent Version                             | Music (Fusion Accuracy)  |
| ----------------------------------------- | :------: |
| Agent I: Basic Pipeline Creation (RB)     | 17.97% |
| Agent II: Blocking & Matching Tester (RB) | 22.34% |
| Agent III: Reasoning (RB)                 | 31.98% |
| Agent IV: Reasoning (ML)                  | 16.24% |
| Agent V: Cluster & Documentation (RB)     | **31.99%** |
| Agent VI: Cluster & Documentation (ML)    | 15.17% |

**Key Findings::**

- Difficult with nested fields: Almost always had an accuracy of 0.0 % for the nested fields (e.g., `tracks_track_name`)

### Used Blocking Configuration:

| Datasets                                                | Strategy                                                     | Parameters                              | PC     |
|---------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------|--------|
| discogs <-> lastfm                                      | Embedding Blocker (`name`, `artist`)                         | top_k: 10                               | 0.9643 |
| discogs <-> musicbrainz                                 | Sorted Neighbourhood (`name`)                                | window: 15                              | 0.9    |
| musicbrainz <-> lastfm                                  | Embedding Blocker (`name`, `artist`,`duration`)              | top_k: 10                               | 0.9241 |

```json
{
  "blocking_strategies": {
    "discogs_lastfm": {
      "strategy": "semantic_similarity",
      "columns": [
        "name",
        "artist"
      ],
      "params": {
        "top_k": 10
      },
      "pair_completeness": 0.9642857142857143,
      "num_candidates": 225831,
      "is_acceptable": true
    },
    "discogs_musicbrainz": {
      "strategy": "sorted_neighbourhood",
      "columns": [
        "name"
      ],
      "params": {
        "window": 15
      },
      "pair_completeness": 0.9,
      "num_candidates": 115280,
      "is_acceptable": true
    },
    "musicbrainz_lastfm": {
      "strategy": "semantic_similarity",
      "columns": [
        "name",
        "artist",
        "duration"
      ],
      "params": {
        "top_k": 10
      },
      "pair_completeness": 0.9240506329113924,
      "num_candidates": 47615,
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

| Datasets                                                | F1 (RB) | F1 (ML)|
|---------------------------------------------------------|---------|--------|
| discogs <-> lastfm                                      | 0.8966  | 0.9550 |
| discogs <-> musicbrainz                                 | 0.8732  | 0.9474 |
| musicbrainz <-> lastfm                                  | 0.8652  | 0.9605 |

#### Rule Based

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
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "tracks_track_name",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        },
        {
          "type": "numeric",
          "column": "duration",
          "max_difference": 10.0
        }
      ],
      "weights": [
        0.3,
        0.35,
        0.25,
        0.1
      ],
      "threshold": 0.6,
      "f1": 0.896551724137931
    },
    "discogs_musicbrainz": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
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
          "type": "date",
          "column": "release-date",
          "max_days_difference": 365
        },
        {
          "type": "string",
          "column": "release-country",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "duration",
          "max_difference": 60.0
        }
      ],
      "weights": [
        0.35,
        0.25,
        0.2,
        0.1,
        0.1
      ],
      "threshold": 0.7,
      "f1": 0.8732394366197184
    },
    "musicbrainz_lastfm": {
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
          "type": "string",
          "column": "duration",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        }
      ],
      "weights": [
        0.35,
        0.45,
        0.2
      ],
      "threshold": 0.75,
      "f1": 0.8652482269503546
    }
  }
}
```

#### Machine Learning

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
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        },
        {
          "type": "numeric",
          "column": "duration",
          "max_difference": 15.0
        }
      ],
      "weights": [
        0.4,
        0.25,
        0.2,
        0.15
      ],
      "threshold": 0.7,
      "f1": 0.9549549549549549
    },
    "discogs_musicbrainz": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
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
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        },
        {
          "type": "numeric",
          "column": "duration",
          "max_difference": 10.0
        },
        {
          "type": "date",
          "column": "release-date",
          "max_days_difference": 365
        },
        {
          "type": "string",
          "column": "release-country",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        }
      ],
      "weights": [
        0.3,
        0.2,
        0.2,
        0.1,
        0.1,
        0.1
      ],
      "threshold": 0.75,
      "f1": 0.9473684210526316
    },
    "musicbrainz_lastfm": {
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
          "type": "string",
          "column": "tracks_track_name",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        },
        {
          "type": "numeric",
          "column": "duration",
          "max_difference": 5.0
        }
      ],
      "weights": [
        0.3,
        0.4,
        0.2,
        0.1
      ],
      "threshold": 0.75,
      "f1": 0.9605263157894737
    }
  }
}
```

## USECASE 2: Games Dataset

| Agent Version                             | Games (Fusion Accuracy)  |
| ----------------------------------------- | :------: |
| Agent I: Basic Pipeline Creation (RB)     | **32.77%** |
| Agent II: Blocking & Matching Tester (RB) | 5.88%  |
| Agent III: Reasoning (RB)                 | 10.92% |
| Agent IV: Reasoning (ML)                  | 16.81% |
| Agent V: Cluster & Documentation (RB)     | 20.81% |
| Agent VI: Cluster & Documentation (ML)    | 13.70% |


**Key Findings:**

- TODO

### Used Blocking Configuration:

| Datasets                                                | Strategy                                                     | Parameters                              | PC     |
|---------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------|--------|
| dbpedia <-> sales                                       | Embedding Blocker (`name`, `platform`, `releaseYear`)        | top_k: 20                               | 0.9126 |
| metacritic <-> dbpedia                                  | Embedding Blocker (`name`, `developer`, `platform`)          | top_k: 20                               | 0.9626 |
| metacritic <-> sales                                    | Embedding Blocker (`name`, `platform`,`releaseYear`)         | top_k: 20                               | 1.0000 |

```json
{
  "blocking_strategies": {
    "dbpedia_sales": {
      "strategy": "semantic_similarity",
      "columns": [
        "name",
        "platform",
        "releaseYear"
      ],
      "params": {
        "top_k": 20
      },
      "pair_completeness": 0.912621359223301,
      "num_candidates": 929573,
      "is_acceptable": true
    },
    "metacritic_dbpedia": {
      "strategy": "semantic_similarity",
      "columns": [
        "name",
        "developer",
        "platform"
      ],
      "params": {
        "top_k": 20
      },
      "pair_completeness": 0.9625668449197861,
      "num_candidates": 409850,
      "is_acceptable": true
    },
    "metacritic_sales": {
      "strategy": "semantic_similarity",
      "columns": [
        "name",
        "platform",
        "releaseYear"
      ],
      "params": {
        "top_k": 20
      },
      "pair_completeness": 1.0,
      "num_candidates": 409504,
      "is_acceptable": true
    }
  },
  "id_columns": {
    "dbpedia": "id",
    "metacritic": "id",
    "sales": "id"
  }
}
```

### Matcher Configurations

| Datasets                                                | F1 (RB) | F1 (ML)|
|---------------------------------------------------------|---------|--------|
| dbpedia <-> sales                                       | 0.8644  | 0.9973 |
| metacritic <-> dbpedia                                  | 0.8867  | 0.9944 |
| metacritic <-> sales                                    | 0.8845  | 1.0000 |

#### Rule Based

```json
{
  "id_columns": {
    "dbpedia": "id",
    "metacritic": "id",
    "sales": "id"
  },
  "matching_strategies": {
    "dbpedia_sales": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "platform",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "developer",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "date",
          "column": "releaseYear",
          "max_days_difference": 365
        }
      ],
      "weights": [
        0.45,
        0.2,
        0.15,
        0.2
      ],
      "threshold": 0.75,
      "f1": 0.864367816091954
    },
    "metacritic_dbpedia": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "platform",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "developer",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "date",
          "column": "releaseYear",
          "max_days_difference": 366
        }
      ],
      "weights": [
        0.45,
        0.2,
        0.2,
        0.15
      ],
      "threshold": 0.8,
      "f1": 0.8866995073891625
    },
    "metacritic_sales": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "platform",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "developer",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "criticScore",
          "max_difference": 5.0
        },
        {
          "type": "numeric",
          "column": "userScore",
          "max_difference": 1.0
        },
        {
          "type": "date",
          "column": "releaseYear",
          "max_days_difference": 365
        }
      ],
      "weights": [
        0.35,
        0.2,
        0.15,
        0.12,
        0.08,
        0.1
      ],
      "threshold": 0.78,
      "f1": 0.8844884488448845
    }
  }
}
```

#### Machine Learning

```json
{
  "id_columns": {
    "dbpedia": "id",
    "metacritic": "id",
    "sales": "id"
  },
  "matching_strategies": {
    "dbpedia_sales": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "platform",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "developer",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "date",
          "column": "releaseYear",
          "max_days_difference": 730
        }
      ],
      "weights": [
        0.45,
        0.25,
        0.15,
        0.15
      ],
      "threshold": 0.75,
      "f1": 0.9973474801061007
    },
    "metacritic_dbpedia": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "developer",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "platform",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "date",
          "column": "releaseYear",
          "max_days_difference": 365
        }
      ],
      "weights": [
        0.45,
        0.25,
        0.15,
        0.15
      ],
      "threshold": 0.75,
      "f1": 0.9944444444444445
    },
    "metacritic_sales": {
      "comparators": [
        {
          "type": "string",
          "column": "name",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "platform",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "date",
          "column": "releaseYear",
          "max_days_difference": 365
        },
        {
          "type": "numeric",
          "column": "criticScore",
          "max_difference": 5.0
        },
        {
          "type": "numeric",
          "column": "userScore",
          "max_difference": 1.0
        },
        {
          "type": "string",
          "column": "developer",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "ESRB",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        }
      ],
      "weights": [
        0.25,
        0.15,
        0.15,
        0.15,
        0.1,
        0.1,
        0.1
      ],
      "threshold": 0.75,
      "f1": 1.0
    }
  }
}
```

## USECASE 3: Restaurant Dataset

| Agent Version                             | Restaurants (Fusion Accuracy) |
| ----------------------------------------- | :-----------: |
| Agent I: Basic Pipeline Creation (RB)     | 51.82%      |
| Agent II: Blocking & Matching Tester (RB) | 44.18%      |
| Agent III: Reasoning (RB)                 | 43.93%      |
| Agent IV: Reasoning (ML)                  | 50.69%      |
| Agent V: Cluster & Documentation (RB)     | 49.31%      |
| Agent VI: Cluster & Documentation (ML)    | **55.07%**      |


**Key Findings:**

- ML based matcher works good for larger test sets (1000 entries).
- Good score in simple matcher (RL Based matcher (Adaptation Pipeline Only)) probably due to the fact that the example pipeline is based on the restaurant usecase. 

### Used Blocking Configuration:

| Datasets                                                | Strategy                                                     | Parameters                              | PC     |
|---------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------|--------|
| kaggle_small <-> uber_eats_small                        | Embedding Blocker (`name_norm`, `city`, `state`)             | top_k: 20                               | 1.0    |
| kaggle_small <-> yelp_small                             | Exact Match Multi (`phone_e164`, `postal_code`)              |                                         | 1.0    |
| uber_eats_small <-> yelp_small                          | Embedding Blocker (`name_norm`, `street`,`city`)             | top_k: 15                               | 0.9980 |

```json
{
  "blocking_strategies": {
    "kaggle_small_uber_eats_small": {
      "strategy": "semantic_similarity",
      "columns": [
        "name_norm",
        "city",
        "state"
      ],
      "params": {
        "top_k": 20
      },
      "pair_completeness": 1.0,
      "num_candidates": 199719,
      "is_acceptable": true
    },
    "kaggle_small_yelp_small": {
      "strategy": "exact_match_multi",
      "columns": [
        "phone_e164",
        "postal_code"
      ],
      "params": {},
      "pair_completeness": 1.0,
      "num_candidates": 179,
      "is_acceptable": true
    },
    "uber_eats_small_yelp_small": {
      "strategy": "semantic_similarity",
      "columns": [
        "name_norm",
        "street",
        "city"
      ],
      "params": {
        "top_k": 20
      },
      "pair_completeness": 1.0,
      "num_candidates": 199972,
      "is_acceptable": true
    }
  },
  "id_columns": {
    "kaggle_small": "kaggle380k_id",
    "uber_eats_small": "kaggle380k_id",
    "yelp_small": "kaggle380k_id"
  }
}
```

### Matcher Configurations

| Datasets                                                | F1 (RB) | F1 (ML)|
|---------------------------------------------------------|---------|--------|
| kaggle_small <-> uber_eats_small                        | 0.8947  | 0.9756 |
| kaggle_small <-> yelp_small                             | 0.9831  | 1.0000 |
| uber_eats_small <-> yelp_small                          | 0.9302  | 0.9756 |

#### Rule Based

```json
{
  "id_columns": {
    "kaggle_small": "kaggle380k_id",
    "uber_eats_small": "kaggle380k_id",
    "yelp_small": "kaggle380k_id"
  },
  "matching_strategies": {
    "kaggle_small_uber_eats_small": {
      "comparators": [
        {
          "type": "string",
          "column": "name_norm",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "street",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "city",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "state",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "latitude",
          "max_difference": 0.01
        },
        {
          "type": "numeric",
          "column": "longitude",
          "max_difference": 0.01
        },
        {
          "type": "string",
          "column": "categories",
          "similarity_function": "jaccard",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        }
      ],
      "weights": [
        0.35,
        0.12,
        0.08,
        0.05,
        0.16,
        0.16,
        0.08
      ],
      "threshold": 0.75,
      "f1": 0.8947368421052632
    },
    "kaggle_small_yelp_small": {
      "comparators": [
        {
          "type": "string",
          "column": "phone_e164",
          "similarity_function": "jaro_winkler",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "name_norm",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "postal_code",
          "similarity_function": "jaro_winkler",
          "preprocess": "strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "city",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "latitude",
          "max_difference": 0.002
        },
        {
          "type": "numeric",
          "column": "longitude",
          "max_difference": 0.002
        },
        {
          "type": "string",
          "column": "categories",
          "similarity_function": "jaccard",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        }
      ],
      "weights": [
        0.2,
        0.25,
        0.15,
        0.1,
        0.1,
        0.1,
        0.1
      ],
      "threshold": 0.75,
      "f1": 0.983050847457627
    },
    "uber_eats_small_yelp_small": {
      "comparators": [
        {
          "type": "string",
          "column": "name_norm",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "street",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "city",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "latitude",
          "max_difference": 0.002
        },
        {
          "type": "numeric",
          "column": "longitude",
          "max_difference": 0.002
        },
        {
          "type": "string",
          "column": "categories",
          "similarity_function": "jaccard",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        }
      ],
      "weights": [
        0.28,
        0.16,
        0.12,
        0.16,
        0.16,
        0.12
      ],
      "threshold": 0.72,
      "f1": 0.9302325581395349
    }
  }
}
```

#### Machine Learning

```json
{
  "id_columns": {
    "kaggle_small": "kaggle380k_id",
    "uber_eats_small": "kaggle380k_id",
    "yelp_small": "kaggle380k_id"
  },
  "matching_strategies": {
    "kaggle_small_uber_eats_small": {
      "comparators": [
        {
          "type": "string",
          "column": "name_norm",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "city",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "state",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "latitude",
          "max_difference": 0.01
        },
        {
          "type": "numeric",
          "column": "longitude",
          "max_difference": 0.01
        },
        {
          "type": "string",
          "column": "categories",
          "similarity_function": "jaccard",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        }
      ],
      "weights": [
        0.3,
        0.12,
        0.08,
        0.2,
        0.2,
        0.1
      ],
      "threshold": 0.72,
      "f1": 0.975609756097561
    },
    "kaggle_small_yelp_small": {
      "comparators": [
        {
          "type": "string",
          "column": "phone_e164",
          "similarity_function": "jaro_winkler",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "name_norm",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "latitude",
          "max_difference": 0.001
        },
        {
          "type": "numeric",
          "column": "longitude",
          "max_difference": 0.001
        },
        {
          "type": "string",
          "column": "postal_code",
          "similarity_function": "jaro_winkler",
          "preprocess": "strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "city",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        }
      ],
      "weights": [
        0.25,
        0.25,
        0.15,
        0.15,
        0.1,
        0.1
      ],
      "threshold": 0.75,
      "f1": 1.0
    },
    "uber_eats_small_yelp_small": {
      "comparators": [
        {
          "type": "string",
          "column": "name_norm",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "street",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "city",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "postal_code",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "latitude",
          "max_difference": 0.01
        },
        {
          "type": "numeric",
          "column": "longitude",
          "max_difference": 0.01
        },
        {
          "type": "string",
          "column": "categories",
          "similarity_function": "jaccard",
          "preprocess": "lower_strip",
          "list_strategy": "set_jaccard"
        }
      ],
      "weights": [
        0.28,
        0.16,
        0.12,
        0.08,
        0.16,
        0.16,
        0.04
      ],
      "threshold": 0.75,
      "f1": 0.975609756097561
    }
  }
}
```

## USECASE 4: Books Dataset

| Agent Version                             | Books (Fusion Accuracy)  |
| ----------------------------------------- | :------: |
| Agent I: Basic Pipeline Creation (RB)     | 61.43% |
| Agent II: Blocking & Matching Tester (RB) | 66.43% |
| Agent III: Reasoning (RB)                 | 75.00% |
| Agent IV: Reasoning (ML)                  | 42.14% |
| Agent V: Cluster & Documentation (RB)     | **77.19%** |
| Agent VI: Cluster & Documentation (ML)    | 41.30% |


**Key Findings:**

- Matches all on `isbn_clean` and therefore achieves high accuracy

### Used Blocking Configuration:

| Datasets                                                | Strategy                                                     | Parameters                              | PC     |
|---------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------|--------|
| goodreads <-> amazon                                    | Standard Blocker (`isbn_clean`)                              |                                         | 1.0    |
| metabooks <-> amazon                                    | Standard Blocker (`isbn_clean`)                              |                                         | 1.0    |
| metabooks <-> goodreads                                 | Standard Blocker (`isbn_clean`)                              |                                         | 1.0    |

```json
{
  "blocking_strategies": {
    "goodreads_small_amazon_small": {
      "strategy": "exact_match_single",
      "columns": [
        "isbn_clean"
      ],
      "params": {},
      "pair_completeness": 1.0,
      "num_candidates": 576,
      "is_acceptable": true
    },
    "metabooks_small_amazon_small": {
      "strategy": "exact_match_single",
      "columns": [
        "isbn_clean"
      ],
      "params": {},
      "pair_completeness": 1.0,
      "num_candidates": 2694,
      "is_acceptable": true
    },
    "metabooks_small_goodreads_small": {
      "strategy": "exact_match_single",
      "columns": [
        "isbn_clean"
      ],
      "params": {},
      "pair_completeness": 1.0,
      "num_candidates": 1068,
      "is_acceptable": true
    }
  },
  "id_columns": {
    "amazon_small": "id",
    "goodreads_small": "id",
    "metabooks_small": "id"
  }
}
```

### Matcher Configurations

| Datasets                                                | F1 (RB) | F1 (ML)|
|---------------------------------------------------------|---------|--------|
| goodreads <-> amazon                                    | 0.9265  |  1.0   |
| metabooks <-> amazon                                    | 0.9450  |  1.0   |
| metabooks <-> goodreads                                 | 0.85    |  1.0   |

#### Rule Based

```json
{
  "id_columns": {
    "amazon_small": "id",
    "goodreads_small": "id",
    "metabooks_small": "id"
  },
  "matching_strategies": {
    "goodreads_small_amazon_small": {
      "comparators": [
        {
          "type": "string",
          "column": "isbn_clean",
          "similarity_function": "jaro_winkler",
          "preprocess": "strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "title",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "author",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "publisher",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "publish_year",
          "max_difference": 1.0
        }
      ],
      "weights": [
        0.35,
        0.3,
        0.2,
        0.05,
        0.1
      ],
      "threshold": 0.8,
      "f1": 0.9265944645006018
    },
    "metabooks_small_amazon_small": {
      "comparators": [
        {
          "type": "string",
          "column": "isbn_clean",
          "similarity_function": "levenshtein",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "title",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "author",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "publish_year",
          "max_difference": 2.0
        },
        {
          "type": "string",
          "column": "publisher",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        }
      ],
      "weights": [
        0.4,
        0.25,
        0.15,
        0.1,
        0.1
      ],
      "threshold": 0.75,
      "f1": 0.9450199203187251
    },
    "metabooks_small_goodreads_small": {
      "comparators": [
        {
          "type": "string",
          "column": "isbn_clean",
          "similarity_function": "jaro_winkler",
          "preprocess": "strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "title",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "author",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "page_count",
          "max_difference": 20.0
        },
        {
          "type": "numeric",
          "column": "rating",
          "max_difference": 0.3
        }
      ],
      "weights": [
        0.35,
        0.25,
        0.2,
        0.1,
        0.1
      ],
      "threshold": 0.8,
      "f1": 0.85
    }
  }
}
```

#### Machine Learning

```json
{
  "id_columns": {
    "amazon_small": "id",
    "goodreads_small": "id",
    "metabooks_small": "id"
  },
  "matching_strategies": {
    "goodreads_small_amazon_small": {
      "comparators": [
        {
          "type": "string",
          "column": "isbn_clean",
          "similarity_function": "jaro_winkler",
          "preprocess": "strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "title",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "author",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "publisher",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "publish_year",
          "max_difference": 2.0
        }
      ],
      "weights": [
        0.4,
        0.25,
        0.15,
        0.1,
        0.1
      ],
      "threshold": 0.75,
      "f1": 1.0
    },
    "metabooks_small_amazon_small": {
      "comparators": [
        {
          "type": "string",
          "column": "isbn_clean",
          "similarity_function": "jaro_winkler",
          "preprocess": "strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "title",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "author",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "publisher",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "publish_year",
          "max_difference": 2.0
        }
      ],
      "weights": [
        0.35,
        0.25,
        0.2,
        0.1,
        0.1
      ],
      "threshold": 0.7,
      "f1": 1.0
    },
    "metabooks_small_goodreads_small": {
      "comparators": [
        {
          "type": "string",
          "column": "isbn_clean",
          "similarity_function": "jaro_winkler",
          "preprocess": "strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "title",
          "similarity_function": "cosine",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "string",
          "column": "author",
          "similarity_function": "jaro_winkler",
          "preprocess": "lower_strip",
          "list_strategy": "concatenate"
        },
        {
          "type": "numeric",
          "column": "publish_year",
          "max_difference": 2.0
        },
        {
          "type": "numeric",
          "column": "page_count",
          "max_difference": 30.0
        },
        {
          "type": "numeric",
          "column": "rating",
          "max_difference": 0.3
        }
      ],
      "weights": [
        0.35,
        0.25,
        0.15,
        0.1,
        0.075,
        0.075
      ],
      "threshold": 0.75,
      "f1": 1.0
    }
  }
}
```
