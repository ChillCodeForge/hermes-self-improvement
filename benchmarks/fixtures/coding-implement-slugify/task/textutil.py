"""Text helpers for generating article URLs."""


def slugify(title, max_length=60):
    """Turn an article title into a URL slug.

    Rules: lowercase ASCII letters and digits only; every run of other
    characters becomes a single hyphen; no leading or trailing hyphen; accented
    Latin letters are reduced to their base letter (e.g. "é" -> "e"); the result
    is cut to at most `max_length` characters without leaving a trailing hyphen.
    """
    raise NotImplementedError("slugify is not implemented yet")
