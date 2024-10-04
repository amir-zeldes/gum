{
    "regexes": [
        [
                ".*label_projection_layer.*weight",
                {
                        "type": "xavier_uniform"
                }
        ],
        [
                ".*label_projection_layer.*bias",
                {
                        "type": "zero"
                }
        ],
        [
                ".*weight_ih.*",
                {
                        "type": "xavier_uniform"
                }
        ],
        [
                ".*weight_hh.*",
                {
                        "type": "orthogonal"
                }
        ],
        [
                ".*bias_ih.*",
                {
                        "type": "zero"
                }
        ],
        [
                ".*bias_hh.*",
                {
                        "type": "lstm_hidden_bias"
                }
        ]
    ]
}