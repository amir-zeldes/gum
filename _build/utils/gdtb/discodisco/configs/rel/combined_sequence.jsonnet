local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");

local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS"));  # uniquely determined by transformer_model

local features = {
    "nuc_children": {"source_key": "nuc_children"},
    "genre": {"source_key": "genre", "label_namespace": "genre"},
    "u1_discontinuous": {"source_key": "u1_discontinuous", "label_namespace": "discontinuous"},
    "u2_discontinuous": {"source_key": "u2_discontinuous", "label_namespace": "discontinuous"},
    "u1_issent": {"source_key": "u1_issent", "label_namespace": "issent"},
    "u2_issent": {"source_key": "u2_issent", "label_namespace": "issent"},
    "length_ratio": {"source_key": "length_ratio"},
    "same_speaker": {"source_key": "same_speaker", "label_namespace": "same_speaker"},
    "doclen": {"source_key": "doclen"},
    "u1_position": {"source_key": "u1_position"},
    "u2_position": {"source_key": "u2_position"},
    "distance": {"source_key": "distance"},
};

{
    "dataset_reader" : {
        "type": "disrpt_2021_rel_e2e",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model_name
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model_name
        },
        "features": features
    },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
  "model": {
    "type": "disrpt_2021_combined_sequence",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": transformer_model_name,
            "train_parameters": true,
            "last_layer_only": true
        }
      }
    },
    "encoder_args": {
      "input_dim": embedding_dim,
      "num_layers": 1,
      "feedforward_hidden_dim": embedding_dim * 2,
      "positional_encoding": "sinusoidal",
      "positional_embedding_size": 512,
      "num_attention_heads": 8,
      "activation": "gelu"
    },
    "features": features,
    "dropout": 0.5,
    "combine_sequences_using_submask": true
  },
  "data_loader": {
    "batch_size": 4,
    "shuffle": true
  },
  "trainer": {
    "num_epochs": 60,
    "patience" : 15,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.05,
      "betas": [0.9, 0.9],
      "parameter_groups": [
        [[".*embedder.*transformer.*"], {"lr": 1e-5}]
      ]
    },
    "validation_metric": "+relation_accuracy"
  }
}
