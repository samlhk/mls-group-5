#!/bin/bash

URL="http://129.215.18.52:8000/rag"
DATA='{"query": "Which animals can hover in the air?"}'
HEADER="Content-Type: application/json"

# Number of concurrent requests to test
CONCURRENCY_LEVELS=(1 5 10 20)
ROUNDS=5  # How many batches to run at each concurrency level

for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do
  echo "Testing $CONCURRENCY concurrent requests, $ROUNDS rounds"

  for ((round = 1; round <= ROUNDS; round++)); do
    echo "  Round $round"

    for ((i = 1; i <= CONCURRENCY; i++)); do
      curl -s -o /dev/null -w "%{http_code} " -X POST "$URL" \
        -H "$HEADER" -d "$DATA" &
    done

    wait  # Wait for all concurrent requests to finish
    echo  # New line after response codes
  done

  echo "-----------------------------------------"
done
