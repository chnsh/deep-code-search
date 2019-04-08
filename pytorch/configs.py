def get_config():
    conf = {
        'workdir': './data/vectors/',
        # data_params
        # training data
        'train_name': 'train.methname.npy',
        'train_api': 'train.apiseq.npy',
        'train_tokens': 'train.tokens.npy',
        'train_desc': 'train.desc.npy',
        # test data
        'valid_name': 'small.test.methname.npy',
        'valid_api': 'small.test.apiseq.npy',
        'valid_tokens': 'small.test.tokens.npy',
        'valid_desc': 'small.test.desc.npy',
        # use data (computing code vectors)
        'use_codebase': 'small.rawcode.txt',  # 'use.rawcode.h5'
        'use_names': 'small.test.methname.npy',
        'use_apis': 'small.test.apiseq.npy',
        'use_tokens': 'small.test.tokens.npy',
        # results data(code vectors)
        'use_codevecs': 'use.codevecs.normalized.h5',  # 'use.codevecs.h5',

        # parameters
        'name_len': 5,
        'api_len': 45,
        'tokens_len': 55,
        'desc_len': 15,
        'n_words': 10002,  # len(vocabulary) + 1
        # vocabulary info
        'vocab_name': 'vocab.methname.pkl',
        'vocab_api': 'vocab.apiseq.pkl',
        'vocab_tokens': 'vocab.tokens.pkl',
        'vocab_desc': 'vocab.desc.pkl',

        # training_params
        'batch_size': 64,
        'chunk_size': 100000,
        'nb_epoch': 2000,
        'validation_split': 0.2,
        # 'optimizer': 'adam',
        'lr': 0.001,
        'valid_every': 10,
        'n_eval': 100,
        'evaluate_all_threshold': {
            'mode': 'all',
            'top1': 0.4,
        },
        'log_every': 100,
        'save_every': 3,
        'reload': 54,
        # 970,#epoch that the model is reloaded from . If reload=0, then train from scratch

        # model_params
        'emb_size': 100,
        'n_hidden': 400,  # number of hidden dimension of code/desc representation
        # recurrent
        'lstm_dims': 200,  # * 2
        'init_embed_weights_methname': None,  # 'word2vec_100_methname.h5',
        'init_embed_weights_tokens': None,  # 'word2vec_100_tokens.h5',
        'init_embed_weights_desc': None,  # 'word2vec_100_desc.h5',
        'margin': 0.05,
        'sim_measure': 'cos',  # similarity measure: gesd, cosine, aesd

    }
    return conf
