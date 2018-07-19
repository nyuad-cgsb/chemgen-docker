#!/usr/bin/env bash

gunicorn --workers=2 --bind=0.0.0.0:5000 --keep-alive=2000 --timeout=2000 --log-level=debug flask_get_counts:app --daemon
## Concurrency is the number of workers celery processes at once
## Since we are parallizing this with docker swarm we set it at 1
celery -A flask_get_counts.celery worker --concurrency 1
