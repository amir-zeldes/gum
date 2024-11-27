local transformer_model_name = 'distilbert-base-cased';
local embedding_dim = 768 + (2 * 80);

// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        "type": "disrpt_2021_rel",
        // See this and sister modules:
        // https://docs.allennlp.org/main/api/data/token_indexers/pretrained_transformer_indexer/
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model_name
            },
            "token_characters": import "../components/char_indexer.libsonnet"
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model_name
        }
    },
    "model": {
        "type": "disrpt_2021_rel_baseline",
        "embedder": {
            // See this and sister modules:
            // https://docs.allennlp.org/v2.1.0/api/modules/token_embedders/pretrained_transformer_embedder/
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": transformer_model_name,
                    "train_parameters": false,
                    "last_layer_only": false
                },
                "token_characters": import "../components/char_embedder.libsonnet"
            }
        },
        // encoders: see https://docs.allennlp.org/v2.1.0/api/modules/seq2vec_encoders/boe_encoder/
        "encoder1": {
            "type": "boe",
            "embedding_dim": embedding_dim,
        },
        "encoder2": {
            "type": "boe",
            "embedding_dim": embedding_dim,
        }
    },
    "train_data_path": "sharedtask2021/data/eng.rst.gum/eng.rst.gum_dev.rels",
    "validation_data_path": "sharedtask2021/data/eng.rst.gum/eng.rst.gum_dev.rels",
    "data_loader": {
        "batch_size": 1,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 20
    }
}
