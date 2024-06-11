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
    elseif alias == :mini12 || alias == :mini
        datadep"ms-marco-MiniLM-L-12-v2"
    elseif alias == :mini6
        datadep"ms-marco-MiniLM-L-6-v2"
    elseif alias == :mini4
        datadep"ms-marco-MiniLM-L-4-v2"
    else
        throw(ArgumentError("Invalid model type"))
    end

    # Tokenizer setup
    tok_path = find_file(model_root, "tokenizer.json")
    tok_config_path = find_file(model_root, "tokenizer_config.json")
    vocab_path = find_file(model_root, "vocab.txt")
    if !isnothing(tok_path)
        ## Load from tokenizer.json
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
    elseif !isnothing(tok_config_path) && !isnothing(vocab_path)
        ## Load from tokenizer_config.json
        tok_config = JSON3.read(tok_config_path)
        vocab_list = readlines(vocab_path)

        ## Double check that all tokens are in vocab
        @assert all(
            sym -> in(tok_config[sym], vocab_list), [
                :unk_token, :cls_token, :sep_token, :pad_token])

        vocab = Dict(k => i - 1 for (i, k) in enumerate(vocab_list))

        wp = WordPiece(vocab_list, tok_config[:unk_token];
            max_char = get(tok_config, :max_input_chars_per_word, 200),
            subword_prefix = get(tok_config, :continuing_subword_prefix, "##"))

        ## We always assume lowercasing with our current tokenizer implementation
        @assert get(tok_config, :do_lower_case, true) "Tokenizer must be lowercased. Model implementation is not compatible."
        ## We assume truncation of 512 if not provided
        trunc = get(tok_config, :model_max_length, nothing) |> x -> isnothing(x) ? 512 : x
        enc = BertTextEncoder(wp, vocab; trunc,
            startsym = tok_config[:cls_token],
            endsym = tok_config[:sep_token],
            padsym = tok_config[:pad_token])
    else
        throw(ArgumentError("Could not find tokenizer.json or tokenizer_config.json + vocab.txt in $model_root"))
    end

    ## Double-check that padding is ID 0, because we pad with 0s in encode() function
    @assert enc.vocab[enc.padsym]==0 "Padding token must be first token in vocabulary with token ID 0."

    # Model loading
    onnx_path = find_file(model_root, ".onnx")
    @assert !isnothing(onnx_path) "Could not find ONNX file in $model_root"
    session = ORT.load_inference(onnx_path)
    return enc, session
end
