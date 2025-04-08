#!/bin/bash

URL="http://192.168.47.132:8000/rag"
DATA='{"query": "Which animals can hover in the air?"}'
HEADER="Content-Type: application/json"

RATES=(1 5 10 20)  # requests per second
DURATION=10        # seconds per rate

for RATE in "${RATES[@]}"; do
  echo "Testing at $RATE requests/sec for $DURATION seconds"
  TOTAL_REQUESTS=$((RATE * DURATION))
  
  START_TIME=$(date +%s)
  
  for ((i = 1; i <= TOTAL_REQUESTS; i++)); do
    # Background request
    curl -s -o /dev/null -w "%{http_code}\n" -X POST "$URL" \
      -H "$HEADER" -d "$DATA" &

    sleep_time=$(awk "BEGIN {print 1/$RATE}")
    sleep $sleep_time
  done

  wait  # Wait for all background curl processes to finish
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished $TOTAL_REQUESTS requests in $ELAPSED seconds"
  echo "-----------------------------"
done

