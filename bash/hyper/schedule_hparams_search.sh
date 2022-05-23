#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

python scripts/kube_scheduler.py --experiment_name exp5 --round 1
python scripts/kube_scheduler.py --experiment_name exp5 --round 2
python scripts/kube_scheduler.py --experiment_name exp5 --round 3
python scripts/kube_scheduler.py --experiment_name exp5 --round 4
python scripts/kube_scheduler.py --experiment_name exp5 --round 5
python scripts/kube_scheduler.py --experiment_name exp5 --round 6
python scripts/kube_scheduler.py --experiment_name exp5 --round 7