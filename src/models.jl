## The weights come from: https://huggingface.co/prithivida/flashrank/tree/main

register(DataDep("ms-marco-TinyBERT-L-2-v2",
    """
    TinyBERT-L-2-v2 cross-encoder trained on the ms-marco dataset.
    """,
    "https://huggingface.co/prithivida/flashrank/resolve/main/ms-marco-TinyBERT-L-2-v2.zip";
    post_fetch_method = unpack
));
register(DataDep("ms-marco-MiniLM-L-12-v2",
    """
    MiniLM-L-12-v2 cross-encoder trained on the ms-marco dataset.
    """,
    "https://huggingface.co/prithivida/flashrank/resolve/main/ms-marco-MiniLM-L-12-v2.zip";
    post_fetch_method = unpack
));

const MODEL_PATHS = Dict(
    :tiny => "opt/ms-marco-TinyBERT-L-2-v2/flashrank-TinyBERT-L-2-v2.onnx",
    :mini => "opt/ms-marco-MiniLM-L-12-v2/flashrank-MiniLM-L-12-v2.onnx"
)
const TOKENIZER_CONFIG_PATH = "opt/ms-marco-TinyBERT-L-2-v2/tokenizer_config.json"
const TOKENIZER_PATH = "opt/ms-marco-TinyBERT-L-2-v2/tokenizer.json"

# Load tokenizer and model based on type
function load_model(model_type)
    # Tokenizer setup
    tok_config = JSON3.read(TOKENIZER_CONFIG_PATH)
    tok = JSON3.read(TOKENIZER_PATH)
    vocab = tok[:model][:vocab]
    vocab_list = reverse_keymap_to_list(vocab)
    wp = WordPiece(vocab_list, tok[:model][:unk_token];
        max_char = tok[:model][:max_input_chars_per_word],
        subword_prefix = tok[:model][:continuing_subword_prefix])
    enc = BertTextEncoder(wp, Dict(String(k) => v for (k, v) in vocab))

    # Model loading
    fn = MODEL_PATHS[model_type]
    model = ORT.load_inference(fn)
    return enc, model
end