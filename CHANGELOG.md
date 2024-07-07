# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1]

### Updated 
- Updated README and documentation post-registration

## [0.4.0]

### Added
- Simplify embedding with the tiny models and provide native support for splitting of long strings into several smaller ones (kwarg `split_instead_trunc`)

## [0.3.0]

### Added
- Added a base Bert Tiny model to support lightning-fast embeddings (alias `tiny_embed`). See `?embed` for more details.


## [0.2.0]

### Added
- Added Sentence Transformers MiniLM L-4 and MiniLM-L-6 models with full precision to provide more choice between TinyBert and MiniLM L-12

## [0.1.0]

### Added
- Initial release