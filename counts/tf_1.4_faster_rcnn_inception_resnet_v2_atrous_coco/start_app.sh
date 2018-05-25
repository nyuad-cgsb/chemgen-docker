#!/usr/bin/env bash

gunicorn --workers=2 --bind=0.0.0.0:5000 --keep-alive=2000 --timeout=2000 --log-level=debug flask_get_counts:app --daemon
celery -A flask_get_counts.celery worker --concurrency 6
