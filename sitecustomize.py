"""Compatibility shims for the bundled local Python environment."""

try:
    import typing_extensions

    if not hasattr(typing_extensions, "Doc"):

        class Doc:
            def __init__(self, documentation: str):
                self.documentation = documentation

            def __repr__(self) -> str:
                return f"Doc({self.documentation!r})"

        typing_extensions.Doc = Doc
except Exception:
    pass
