local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local corpus_name = std.extVar("CORPUS");
local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS"));  # uniquely determined by transformer_model

local features = {
  "features": {
    "nuc_children": {"source_key": "nuc_children"},
    "sat_children": {"source_key": "sat_children"},
    "genre": {"source_key": "genre", "label_namespace": "genre"},
    "u1_discontinuous": {"source_key": "u1_discontinuous", "label_namespace": "discontinuous"},
    "u2_discontinuous": {"source_key": "u2_discontinuous", "label_namespace": "discontinuous"},
    "u1_issent": {"source_key": "u1_issent", "label_namespace": "issent"},
    "u2_issent": {"source_key": "u2_issent", "label_namespace": "issent"},
    "unit1_case": {"source_key": "unit1_case", "label_namespace": "case"},
    "unit2_case": {"source_key": "unit2_case", "label_namespace": "case"},
    "u1_depdir": {"source_key": "u1_depdir", "label_namespace": "depdir"},
    "u2_depdir": {"source_key": "u2_depdir", "label_namespace": "depdir"},
    "u1_func": {"source_key": "u1_func", "label_namespace": "func"},
    "u2_func": {"source_key": "u2_func", "label_namespace": "func"},
    "length_ratio": {"source_key": "length_ratio"},
    "same_speaker": {"source_key": "same_speaker", "label_namespace": "same_speaker"},
    "doclen": {"source_key": "doclen"},
    //"distance": {"source_key": "distance"},
    "distance": {
      "source_key": "distance",
      "xform_fn": {
        "type": "bins",
        "bins": [[-1e9, -8], [-8, -2], [-2, 0], [0, 2], [2, 8], [8, 1e9]]
      },
      "label_namespace": "distance"
    },
    "u1_position": {
    "source_key": "u1_position",
    "xform_fn": {
      "type": "bins",
      "bins": [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1e9]]
      //"bins": [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1e9]]
    },
    "label_namespace": "u1_position"
    }, 
    "u2_position": {
    "source_key": "u2_position",
    "xform_fn": {
      "type": "bins",
      "bins": [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1e9]]
      //"bins": "bins": [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1e9]]
    },
    "label_namespace": "u2_position"
    },
    "lex_overlap_length": {
      "source_key": "lex_overlap_length",
      "xform_fn": {
        "type": "bins",
        "bins": [[0, 2], [2, 7], [7, 1e9]]
      },
      "label_namespace": "lex_overlap"
    }
  },
  "corpus": corpus_name,
  // By default, we will use all features for a corpus, but they can be overridden below.
  // The values inside the array need to match a key under the "features" dict above.
  "corpus_configs": {
    "deu.rst.pcc": ["distance", "u1_depdir", "u2_depdir", "u2_func", "u1_position", "u2_position",
                    "sat_children", "nuc_children"],
    "eng.pdtb.pdtb": ['u2_depdir', 'u2_func', 'u2_issent', 'u2_position', 'length_ratio'],
    "eng.rst.gum": ["distance", "same_speaker", "u2_func", "u2_depdir", "unit1_case", "unit2_case", "nuc_children",
                    "sat_children", "genre", "lex_overlap_length", "u2_discontinuous", "u1_discontinuous",
                    "u1_position", "u2_position"],
    "eng.rst.rstdt": ['u2_discontinuous', 'u1_position', 'u2_position', 'u2_func', 'u2_issent'],
    "eng.sdrt.stac": ["same_speaker"],
    "eus.rst.ert": ["u2_position"],
    "fra.sdrt.annodis": ["u1_depdir", "u1_position"],
    "nld.rst.nldt": ['distance', 'u1_depdir', 'sat_children', 'genre', 'u1_position'],
    "por.rst.cstn": ['u2_discontinuous', 'u1_position', 'u2_position'],
    "rus.rst.rrt": ['distance', 'nuc_children', 'sat_children', 'u1_position', 'u2_position', 'u1_depdir',
                    'u2_depdir', 'u2_func', 'u1_issent', 'u2_issent'],
    "spa.rst.rststb": ["distance", "u1_depdir", "u1_discontinuous", "u2_depdir", "sat_children", "u2_func", "genre"],
    "spa.rst.sctb": ['distance', 'u1_position', 'sat_children'],
    "tur.pdtb.tdb": ["distance", "u1_depdir", "u2_depdir", "u2_func", "u1_issent", "u2_issent", "length_ratio",
                     "u1_position", "u2_position"],
    "zho.pdtb.cdtb": ["distance", "u1_depdir", "u2_depdir", "u2_func", "u1_issent", "u2_issent", "length_ratio"],
    "zho.rst.sctb": ['sat_children', 'nuc_children', 'genre', 'u2_discontinuous', 'u1_discontinuous', 'u1_depdir', 'u1_func'],
    "fas.rst.prstc": ['distance', 'nuc_children', 'sat_children', 'u2_discontinuous', 'genre'],
  }
};

// For small corpora, make this number reflect the size of train
// For larger corpora, use a smaller number, aiming for 1/3 of total size
local batches_per_epoch = {
  "deu.rst.pcc": 541,
  "eng.pdtb.pdtb": 3000, // real: 10980
  "eng.rst.gum": 1700, // real: 3475
  "eng.rst.rstdt": 2000, // real: 4001
  "eng.sdrt.stac": 1200, // real: 2395
  "eus.rst.ert": 634,
  "fas.rst.prstc": 1025,
  "fra.sdrt.annodis": 547,
  "nld.rst.nldt": 402,
  "por.rst.cstn": 1037,
  "rus.rst.rrt": 2500, // real: 7217
  "spa.rst.rststb": 560,
  "spa.rst.sctb": 110,
  "tur.pdtb.tdb": 613,
  "zho.pdtb.cdtb": 915,
  "zho.rst.sctb": 110
};

{
  "dataset_reader" : {
    "type": "disrpt_2021_rel_flair_clone",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model_name,
        "max_length": 511
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
    "type": "disrpt_2021_flair_clone",
    "embedder": {
      "type": "featureful_bert",
      "model_name": transformer_model_name,
      "max_length": 511,
      "train_parameters": true,
      "last_layer_only": true
    },
    "seq2vec_encoder": {
        "type": "bert_pooler",
        "pretrained_model": transformer_model_name
    },
    "feature_dropout": 0.0,
    "features": features,
  },
  "data_loader": {
    "batches_per_epoch": batches_per_epoch[corpus_name],
    // NOTE: if you need to change batch size, scale batches_per_epoch, which assumes
    // a batch size of 4, by an appropriate amount. E.g., if you need to make batch
    // size 2, then use `batches_per_epoch[corpus_name] * 2`
    "batch_size": 4,
    "shuffle": true
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 12,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      #"weight_decay": 0.05,
      #"betas": [0.9, 0.99],
      #"parameter_groups": [
      #  [[".*embedder.*transformer.*"], {"lr": 2e-5}]
      #],
    },
    #"learning_rate_scheduler": {
    #  "type": "slanted_triangular",
    #  "num_epochs": 50,
    #  "cut_frac": 0.1,
    #},
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.6,
      "mode": "max",
      "patience": 2,
      "verbose": true,
      "min_lr": 5e-7
    },
    //"learning_rate_scheduler": {
    //  "type": "cosine",
    //  "t_initial": 5,
    //},
    "validation_metric": "+relation_accuracy"
  }
}
