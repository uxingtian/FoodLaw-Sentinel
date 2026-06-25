"""Food safety legal QA application."""


def _patch_typing_extensions() -> None:
    try:
        import typing_extensions

        if hasattr(typing_extensions, "Doc"):
            return

        class Doc:
            def __init__(self, documentation: str):
                self.documentation = documentation

            def __repr__(self) -> str:
                return f"Doc({self.documentation!r})"

        typing_extensions.Doc = Doc
    except Exception:
        return


_patch_typing_extensions()
