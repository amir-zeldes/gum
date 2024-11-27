{
    "type": "character_encoding",
    "embedding": {
        "embedding_dim": 32,
        "vocab_namespace": "token_characters"
    },
    "encoder": {
        "type": "lstm",
        "input_size": 32,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": true
    }
    //"encoder": {
    //    "type": "cnn",
    //    "embedding_dim": 64,
    //    "num_filters": 128,
    //    "ngram_filter_sizes": [2,3,4],
    //    "conv_layer_activation": "relu",
    //    "output_dim": 64
    //}
}
