local transformer_model_name = 'distilbert-base-cased';
local embedding_dim = 768;

// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        "type": "disrpt_2021_rel",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model_name
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model_name
        }
    },
    "model": {
        "type": "disrpt_2021_rel_baseline",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    // https://docs.allennlp.org/v2.1.0/api/modules/token_embedders/pretrained_transformer_embedder/
                    "type": "pretrained_transformer",
                    "model_name": transformer_model_name,
                    "train_parameters": false,
                    "last_layer_only": false
                }
            }
        },
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
