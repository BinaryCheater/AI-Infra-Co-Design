"""Replay frozen RTL probes with the same native causal memory used by ELF."""

from .rust_frontend import predict_span

MODEL_VERSION = "ventus-source-frontend-v4-causal"


def main():
    from .source_ready import main as replay

    replay(
        candidate_predict=lambda *a, **kw: predict_span(*a, **kw, trace=True),
        model_version=MODEL_VERSION,
    )


if __name__ == "__main__":
    main()
