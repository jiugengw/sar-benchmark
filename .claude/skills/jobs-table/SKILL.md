---
name: jobs-table
description: Show the SMU cluster's current SAR training jobs as a timeline (rows = time periods, columns = the two GPU slots) plus per-job details (what it is, GPU, state, epoch progress, estimated finish) and the latest driver-log lines. Use when the user types /jobs-table or asks for a jobs table, job status, queue, or "what's running".
---

# Jobs table

1. Run, from the repository root:

   ```bash
   python3 scripts/jobs_table.py
   ```

   It makes one read-only SSH call to the cluster login node (`$CLUSTER_HOST` or `scripts/.cluster_host`) and prints, first, a timeline: rows are time periods
   (clock times, `Mon 09:15` once past midnight) and columns are the user's two running slots ("GPU slot 1/2"), each cell
   the job in that slot with its id and GPU type. Running jobs end when their epochs run out; waiting jobs take the next
   free slot in Slurm priority order with a typical duration for their kind (`TYPICAL_HOURS` in the script), marked `≈`.
   Jobs a driver will submit later (next chain stage, packaging) are not in the queue yet, so add them from the
   driver-log lines when you know them. Then it prints a per-job table of every queued or running Slurm job of this user (job id, Slurm name, run folder under `runs/`, GPU type, state, elapsed time, epoch progress
   from the run's `train/results.csv` and `train/args.yaml`, and a rough time left), the cluster's file-quota count
   (files + folders, limit 100,000), and the last lines of the newest driver logs in `~/sbatch-staging/*.log`.

2. If it reports the cluster as unreachable, tell the user the VPN looks down; the cluster jobs keep running without it.

3. Present the timeline first (rows = time periods, columns = GPU slots), replacing run names with a short
   plain-language "what it is" for each job, then the per-job table if useful. Work it
   out from the run name and the driver-log lines, for example:
   - `ft40-s1ad-*`: all-data stride-1 chain stages or soup ingredients (memorise variants)
   - `yolo26m-p2-dota-s1-140-*`: the 140-epoch straight test (stride 1), compared with 100 + plain 40
   - `sar-s1soup-heads`: the stride-1 two-head classifier for the soup
   - `sar-*-pkg`: packaging a submission ZIP

   Translate pending reasons into words: `QOSMaxJobsPerUserLimit` = waiting for one of the user's 2 running slots;
   `ReqNodeNotAvail` / `Priority` = waiting for a free GPU of that type; `QOSMaxSubmitJobPerUserLimit` = at the
   4-submitted-jobs limit.

4. Say what finishes next and what the user can upload next, if anything. Flag the file quota if it is above ~97,000.
