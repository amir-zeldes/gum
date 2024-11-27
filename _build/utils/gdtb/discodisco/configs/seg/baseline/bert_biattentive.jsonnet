local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS")) + 64 * 2 + 300;
local context_hidden_size = 400;
local encoder_hidden_dim = 256;

local context_encoder = {
    "type": "lstm",
    "input_size": embedding_dim,
    "hidden_size": context_hidden_size / 4, // 4 <= 2 bilstms applied to 2 sentences
    "num_layers": 1,
    "bidirectional": true,
    "dropout": 0.2
};

local token_features = {
    "pos_tags": {"source_key": "pos", "label_namespace": "upos"},
    "cpos_tags": {"source_key": "cpos", "label_namespace": "xpos"},
    "head_distances": {"source_key": "head_dist", "xform_fn": "abs_natural_log"},
    "document_depth": {"source_key": "sent_doc_percentile"},
    "sentence_length": {"source_key": "s_len", "xform_fn": "natural_log"},
    "token_lengths": {"source_key": "tok_len", "xform_fn": "natural_log"}
};

// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        "type": "disrpt_2021_seg",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": transformer_model_name
            },
            "fasttext": {
                "type": "single_id",
                "namespace": "fasttext",
            },
            "token_characters": import "../../components/char_indexer.libsonnet"
        },
        "tokenizer": {
            "type": "whitespace"
        },
        "token_features": token_features
    },
    "model": {
        "type": "disrpt_2021_seg_biattentive",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": transformer_model_name,
                    "train_parameters": false,
                    "last_layer_only": false
                },
                "fasttext": {
                    "type": "embedding",
                    "vocab_namespace": "fasttext",
                    "embedding_dim": 300,
                    "trainable": false,
                    "pretrained_file": std.extVar("FASTTEXT_EMBEDDING_FILE")
                },
                "token_characters": import "../../components/char_embedder.libsonnet"
            }
        },
        // seq2vec encoders for neighbor sentences
        "prev_sentence_encoder": context_encoder,
        "next_sentence_encoder": context_encoder,
        "encoder": {
            "type": "lstm",
            "input_size": encoder_hidden_dim,
            "hidden_size": encoder_hidden_dim,
            "num_layers": 1,
            "bidirectional": true
        },
        "integrator": {
            "type": "qanet_encoder",
            "input_dim": encoder_hidden_dim * 3,
            "hidden_dim": encoder_hidden_dim * 2,
            "attention_projection_dim": encoder_hidden_dim,
            "feedforward_hidden_dim": encoder_hidden_dim,
            "num_blocks": 1,
            "num_convs_per_block": 3,
            "conv_kernel_size": 3,
            "num_attention_heads": 8,
            "dropout_prob": 0.2,
            "layer_dropout_undecayed_prob": 0.2,
            "attention_dropout_prob": 0.2,
        },
        "token_features": token_features,
        "embedding_dropout": 0.2,
        "encoder_dropout": 0.5,
        "feature_dropout": 0.3,
        "use_crf": false
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-4,
            "parameter_groups": [
                [[".*transformer.*"], {"lr": 1e-5}]
            ]
        },
        "run_confidence_checks": false,
        "patience": 7,
        "num_epochs": 60,
        // probably best to just use loss
        "validation_metric": "+span_f1"
    }
}
