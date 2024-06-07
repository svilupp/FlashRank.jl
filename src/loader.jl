# finds specified file in any subfolder
function find_file(path::String, needle::String)
    for (root, _, files) in walkdir(path)
        for file in files
            ## skip mac artifacts
            if occursin(needle, file) && !startswith(file, "._")
                return joinpath(root, file)
            end
        end
    end
    return nothing
end

# Load tokenizer and model based on type
function load_model(alias::Symbol)
    model_root = if alias == :tiny
        datadep"ms-marco-TinyBERT-L-2-v2"
    elseif alias == :mini
        datadep"ms-marco-MiniLM-L-12-v2"
    else
        throw(ArgumentError("Invalid model type"))
    end

    # Tokenizer setup
    tok_path = find_file(model_root, "tokenizer.json")
    @assert !isnothing(tok_path) "Could not find tokenizer.json in $model_root"
    tok_config = JSON3.read(tok_path)
    model_config = tok_config[:model]
    vocab_list = reverse_keymap_to_list(model_config[:vocab])
    extract_and_add_tokens!(tok_config[:added_tokens], vocab_list)
    ## 0-based indexing as we provide it to models trained in python
    vocab = Dict(k => i - 1 for (i, k) in enumerate(vocab_list))

    wp = WordPiece(vocab_list, model_config[:unk_token];
        max_char = model_config[:max_input_chars_per_word],
        subword_prefix = model_config[:continuing_subword_prefix])

    ## We always assume lowercasing with our current tokenizer implementation
    @assert get(tok_config[:normalizer], :lowercase, true) "Tokenizer must be lowercased. Model implementation is not compatible."
    ## We assume truncation of 512 if not provided
    trunc = get(tok_config, :truncation, nothing) |> x -> isnothing(x) ? 512 : x
    enc = BertTextEncoder(wp, vocab; trunc)

    ## Double-check that padding is ID 0, because we pad with 0s in encode() function
    @assert enc.vocab[enc.padsym]==0 "Padding token must be first token in vocabulary with token ID 0."

    # Model loading
    onnx_path = find_file(model_root, ".onnx")
    @assert !isnothing(onnx_path) "Could not find ONNX file in $model_root"
    session = ORT.load_inference(onnx_path)
    return enc, session
end
