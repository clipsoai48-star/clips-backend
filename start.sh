#!/bin/bash
rq worker --worker-class rq.worker.SimpleWorker clipso_jobs_priority clipso_jobs &
uvicorn main:app --host 0.0.0.0 --port $PORT
