"""Pagination helpers used by the report listing endpoint."""


def paginate(items, page, per_page):
    """Return the items shown on 1-based `page` when showing `per_page` items per page."""
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be positive")
    start = (page - 1) * per_page
    return items[start:start + per_page]


def page_count(total, per_page):
    """Return how many pages are needed to show `total` items."""
    if total < 0 or per_page < 1:
        raise ValueError("total must be non-negative and per_page positive")
    return -(-total // per_page)
