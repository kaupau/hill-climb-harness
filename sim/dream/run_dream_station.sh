#!/bin/sh
# Dream-1 driving the MuJoCo sim through Dream's own station agent.
#   1. sim physics + frames:  ../.venv/bin/python ../server.py --port 8191
#   2. this script (station API on :8291, Dream key from Doppler):
#        doppler run -p typesafe -c dev -- ./run_dream_station.sh
#   3. the harness / conductor / grasp.py against http://127.0.0.1:8291 (8091 on this Mac is the tunnel to the REAL station).
# Never on the station machine itself.
cd "$(dirname "$0")"
: "${DREAM_API_KEY:?DREAM_API_KEY not set (doppler run -p typesafe -c dev -- $0)}"
export SIM_URL="${SIM_URL:-http://127.0.0.1:8191}"
export PYTHONPATH="$PWD/addons${PYTHONPATH:+:$PYTHONPATH}"
exec ../.venv-dream/bin/python -m robots.station --config station_sim.yaml --api-port "${API_PORT:-8291}" \
  --arms-lock "$PWD/arms.lock" --inference-log "$PWD/../out/dream-inference.jsonl" "$@"
