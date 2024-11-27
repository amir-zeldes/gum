local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local max_length = 512;

local transformer_dim = std.parseInt(std.extVar("EMBEDDING_DIMS"));  # uniquely determined by transformer_model

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
        "max_length": 512,
        "features": features
    },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
  "model": {
    "type": "disrpt_2021_e2e",
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
    "context_layer": {
        "type": "pass_through",
        "input_dim": transformer_dim
    },
    "initializer": {
      "regexes": [
        [".*_span_updating_gated_sum.*weight", {"type": "xavier_normal"}],
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer.*weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
      ]
    },
    "features": features,
  },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
  "trainer": {
    "num_epochs": 40,
    "patience" : 10,
    //"learning_rate_scheduler": {
    //  "type": "slanted_triangular",
    //  "cut_frac": 0.06
    //},
    "validation_metric": "+relation_accuracy",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.05,
      "betas": [0.9, 0.9]
      //"parameter_groups": [
      //  [[".*transformer.*"], {"lr": 1e-5}]
      //],
    }
  }
}
