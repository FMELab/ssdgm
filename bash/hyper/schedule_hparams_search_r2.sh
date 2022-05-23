#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

python scripts/kube_scheduler.py --experiment_name exp6 --round 8
python scripts/kube_scheduler.py --experiment_name exp6 --round 9
python scripts/kube_scheduler.py --experiment_name exp6 --round 10
