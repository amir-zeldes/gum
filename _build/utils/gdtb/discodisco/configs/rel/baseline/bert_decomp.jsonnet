local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS")) + 64 * 2 + 300;
local feedforward_hidden_dim = 256;

local features = {
    //"nuc_children": {"source_key": "nuc_children"},
    "genre": {"source_key": "genre", "label_namespace": "genre"},
    "u1_discontinuous": {"source_key": "u1_discontinuous", "label_namespace": "discontinuous"},
    "u2_discontinuous": {"source_key": "u2_discontinuous", "label_namespace": "discontinuous"},
    "u1_issent": {"source_key": "u1_issent", "label_namespace": "issent"},
    "u2_issent": {"source_key": "u2_issent", "label_namespace": "issent"},
    "u1_length": {"source_key": "u1_length"},
    "u2_length": {"source_key": "u2_length"},
    "length_ratio": {"source_key": "length_ratio"},
    "u1_speaker": {"source_key": "u1_speaker", "label_namespace": "speaker"},
    "u2_speaker": {"source_key": "u2_speaker", "label_namespace": "speaker"},
    "same_speaker": {"source_key": "same_speaker", "label_namespace": "same_speaker"},
    "u1_func": {"source_key": "u1_func", "label_namespace": "func"},
    "u1_pos": {"source_key": "u1_pos", "label_namespace": "pos"},
    "u1_depdir": {"source_key": "u1_depdir", "label_namespace": "depdir"},
    "u2_func": {"source_key": "u2_func", "label_namespace": "func"},
    "u2_pos": {"source_key": "u2_pos", "label_namespace": "pos"},
    "u2_depdir": {"source_key": "u2_depdir", "label_namespace": "depdir"},
    "doclen": {"source_key": "doclen"},
    "u1_position": {"source_key": "u1_position"},
    "u2_position": {"source_key": "u2_position"},
    "distance": {"source_key": "distance"},
    //"lex_overlap_words": {"source_key": "lex_overlap_words", "label_namespace": "lex_overlap_words"},
    "lex_overlap_length": {"source_key": "lex_overlap_length"}
};


local seq2seq_encoder = {
    "type": "pytorch_transformer",
    "input_dim": embedding_dim,
    "num_layers": 1,
    "num_attention_heads": 4,
    "positional_encoding": "sinusoidal",
    "dropout_prob": 0.1,
    "activation": "gelu"
};

// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader": {
        "type": "disrpt_2021_rel",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model_name
            },
            "fasttext": {
                "type": "single_id",
                "namespace": "fasttext",
            },
            "token_characters": import "../../components/char_indexer.libsonnet"
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
        "type": "disrpt_2021_rel_decomp",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    // https://docs.allennlp.org/v2.1.0/api/modules/token_embedders/pretrained_transformer_embedder/
                    "type": "pretrained_transformer",
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
        //"unit1_encoder": seq2seq_encoder,
        //"unit2_encoder": seq2seq_encoder,
        "attend_feedforward": {
            "input_dim": embedding_dim,
            "num_layers": 2,
            "hidden_dims": feedforward_hidden_dim,
            "activations": "relu",
            "dropout": 0.2
        },
        "matrix_attention": {"type": "dot_product"},
        "compare_feedforward": {
            "input_dim": embedding_dim * 2,
            "num_layers": 2,
            "hidden_dims": feedforward_hidden_dim,
            "activations": "relu",
            "dropout": 0.2
        },
        "initializer": {
            "regexes": [
                [".*linear_layers.*weight", {"type": "xavier_normal"}],
                [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
            ]
        },
        "features": features
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-4
        },
        "num_epochs": 40,
        "patience": 10,
        "cuda_device": 0,
        "validation_metric": "+accuracy"
    }
}
