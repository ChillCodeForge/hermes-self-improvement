"""Runtime settings for the sync client."""

import os

DEFAULTS = {
    "sync.batch_size": 100,
    "sync.request_timeout": 45,
    "sync.retry_limit": 3,
}

ENVIRONMENT_OVERRIDES = {
    "SYNC_BATCH_SIZE": "sync.batch_size",
    "SYNC_REQUEST_TIMEOUT": "sync.request_timeout",
    "SYNC_RETRY_LIMIT": "sync.retry_limit",
}


def load():
    settings = dict(DEFAULTS)
    for variable, key in ENVIRONMENT_OVERRIDES.items():
        if variable in os.environ:
            settings[key] = int(os.environ[variable])
    return settings
