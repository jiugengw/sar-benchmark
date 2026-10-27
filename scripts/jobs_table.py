#!/usr/bin/env python3
"""Print the cluster's jobs as Markdown: a timeline (rows = time periods, columns = the user's two running slots)
projected from the queue, then the per-job details (run folder, GPU, epoch progress, time left) and the latest lines of
the chain/soup driver logs.

Running jobs end when their epochs run out (elapsed per epoch x epochs left); waiting jobs take the next free slot in
priority order, with a typical duration for their kind (see TYPICAL_HOURS). Jobs that drivers will submit later are
not in the queue yet, so they are not shown.

One SSH call collects everything (the cluster needs the VPN). Read-only.

    python3 scripts/jobs_table.py [--host jgwong.2023@10.193.104.102]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

REMOTE = r'''
cd ~/scratchDirectory/sar-benchmark 2>/dev/null || exit 3
python3 - <<'PY'
import csv, glob, json, os, subprocess, time
q = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i|%j|%T|%M|%l|%R|%b|%V|%Q'],
                   capture_output=True, text=True).stdout.strip().splitlines()
jobs = []
for line in q:
    jid, name, state, elapsed, limit, reason, gres, submit, prio = line.split('|')
    runs = [d for d in glob.glob(f'runs/*-{jid}') if os.path.isdir(d)]
    # A chain job trains several runs in turn; its current one is the run whose args.yaml was written last.
    started = lambda d: os.path.getmtime(os.path.join(d, 'train', 'args.yaml')) if os.path.exists(
        os.path.join(d, 'train', 'args.yaml')) else os.path.getmtime(d)
    run = max(runs, key=started) if runs else ''
    epoch = total = done_secs = since_secs = None
    if run:
        res = os.path.join(run, 'train', 'results.csv')
        if os.path.exists(res):
            with open(res) as f:
                rows = list(csv.reader(f))
            epoch = len(rows) - 1 if rows else 0
            # Time spent on the finished epochs of this run, and on the current epoch so far.
            done_secs = int(os.path.getmtime(res) - started(run))
            since_secs = int(time.time() - os.path.getmtime(res))
        args = os.path.join(run, 'train', 'args.yaml')
        if os.path.exists(args):
            for l in open(args):
                if l.startswith('epochs:'):
                    total = int(l.split(':')[1])
    jobs.append(dict(id=jid, name=name, state=state, elapsed=elapsed, limit=limit, reason=reason,
                     gpu=gres.replace('gres/gpu:', '').replace('rtx_pro_6000_blackwell_max', 'blackwell'),
                     run=run, done_secs=done_secs, since_secs=since_secs, epoch=epoch, total=total, prio=int(prio or 0)))
logs = {}
for path in sorted(glob.glob(os.path.expanduser('~/sbatch-staging/*.log')), key=os.path.getmtime)[-6:]:
    with open(path) as f:
        lines = [l.rstrip() for l in f if l.strip()]
    logs[os.path.basename(path)] = lines[-2:]
try:
    entries = int(subprocess.run(['getfattr', '--only-values', '-n', 'ceph.dir.rentries',
                                  '/common/scratch/users/j/jgwong.2023'], capture_output=True, text=True).stdout)
except ValueError:
    entries = None
print(json.dumps(dict(now=time.strftime('%Y-%m-%d %H:%M'), jobs=jobs, logs=logs, entries=entries)))
PY
'''


def to_seconds(text: str) -> int:
    days, _, rest = text.rpartition('-')
    parts = [int(p) for p in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def seconds_left(job: dict) -> float:
    """Seconds until the job's current run ends: its time per finished epoch (a chain job's earlier runs excluded)
    times the epochs left, less the time already spent on the current epoch."""
    if job.get('done_secs'):
        return max(0.0, job['done_secs'] / job['epoch'] * (job['total'] - job['epoch']) - job['since_secs'])
    return to_seconds(job['elapsed']) / job['epoch'] * (job['total'] - job['epoch'])


TYPICAL_HOURS = {'sar-s1140': 11.0, 'sar-ft40': 2.5, 'pkg': 1.0, 'heads': 4.1}  # waiting jobs, by job-name pattern
SLOTS = 2  # running jobs allowed per user (studentqos)


def label(job: dict) -> str:
    run = job['run'].replace('runs/', '')
    short = run.rsplit('-', 1)[0].replace('ft40-', '') if run else job['name'].replace('sar-', '')
    return f"{short} [{job['id']}, {job['gpu'].split(':')[0] or '?'}]"


def timeline(data: dict) -> list[tuple[float, float, int, str]]:
    """(start, end, slot, label) in hours from now, running jobs first, then waiting ones in priority order."""
    free = [0.0] * SLOTS
    out = []
    running = [j for j in data['jobs'] if j['state'] == 'RUNNING']
    waiting = sorted((j for j in data['jobs'] if j['state'] != 'RUNNING'), key=lambda j: -j['prio'])
    for k, j in enumerate(running[:SLOTS]):
        if j['epoch'] and j['total']:
            left = seconds_left(j) / 3600
            mark = ''
        else:
            left = max(0.25, min(to_seconds(j['limit']) - to_seconds(j['elapsed']),
                                 3600 * next((h for key, h in TYPICAL_HOURS.items() if key in j['name']), 1.0)
                                 - to_seconds(j['elapsed'])) / 3600)
            mark = '≈'
        free[k] = left
        out.append((0.0, left, k, mark + label(j)))
    for j in waiting:
        k = min(range(SLOTS), key=lambda i: free[i])
        hours = next((h for key, h in TYPICAL_HOURS.items() if key in j['name']), to_seconds(j['limit']) / 3600)
        out.append((free[k], free[k] + hours, k, '≈' + label(j) + ' (waiting)'))
        free[k] += hours
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='jgwong.2023@10.193.104.102')
    parser.add_argument('--direct', action='store_true', help='log in directly (needs a working SSH key) when no '
                                                              'shared connection is open')
    args = parser.parse_args()
    # Log in with the cluster's own key (~/.ssh/id_ed25519_smu, comment mac-to-smu); else reuse a shared connection the
    # user opened (ssh -o ControlMaster=yes -o ControlPath=~/.ssh/cm-%r@%h:%p ...); else a key loaded in the agent.
    from pathlib import Path
    key = Path.home() / '.ssh' / 'id_ed25519_smu'
    shared = Path.home() / '.ssh' / f"cm-{args.host}:22"
    has_key = subprocess.run(['ssh-add', '-l'], capture_output=True).returncode == 0
    if not key.exists() and not shared.exists() and not has_key and not args.direct:
        # A direct login without a key fails, and the cluster counts failed logins against this address until it
        # refuses connections for a while.
        sys.exit(f'no SSH key ({key}) and no shared connection ({shared}); open one with: ssh -N -o ControlMaster=yes '
                 f'-o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=24h {args.host}')
    via = (['-i', str(key), '-o', 'IdentitiesOnly=yes'] if key.exists()
           else ['-o', f'ControlPath={shared}'] if shared.exists() else [])
    out = subprocess.run(['ssh', *via, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host, REMOTE],
                         capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith('{')]
    if not lines:
        sys.exit(f'cluster unreachable or no output (VPN?): {out.stderr.strip()[-300:]}')
    data = json.loads(lines[-1])
    import datetime as dt
    now = dt.datetime.strptime(data['now'], '%Y-%m-%d %H:%M')
    print(f"Cluster jobs at {data['now']}  (files+folders {data['entries']:,} of 100,000)\n")
    events = timeline(data)
    def clock(hours: float) -> str:
        t = now + dt.timedelta(hours=hours)
        return t.strftime('%H:%M' if t.date() == now.date() else '%a %H:%M')

    marks = sorted({round(t, 2) for e in events for t in e[:2]} | {0.0})
    print('| Time | ' + ' | '.join(f'GPU slot {k + 1}' for k in range(SLOTS)) + ' |')
    print('|---|' + '---|' * SLOTS)
    for a, b in zip(marks, marks[1:]):
        cells = []
        for k in range(SLOTS):
            hit = [e[3] for e in events if e[2] == k and e[0] <= a + 0.01 and e[1] >= b - 0.01]
            cells.append(hit[0] if hit else 'free')
        print(f'| {clock(a)} → {clock(b)} | ' + ' | '.join(cells) + ' |')
    print('\n≈ = estimated from a typical duration (no epoch counter yet). Jobs that drivers will submit later are not shown.\n')
    print('| Job | Name | Run | GPU | State | Elapsed | Progress | Est. finish |')
    print('|---|---|---|---|---|---|---|---|')
    for j in data['jobs']:
        progress = eta = ''
        if j['epoch'] is not None and j['total']:
            progress = f"epoch {j['epoch']}/{j['total']}"
            if j['state'] == 'RUNNING' and j['epoch'] > 0:
                left = seconds_left(j)
                eta = f"~{left / 3600:.1f} h"
        state = j['state'] if j['state'] == 'RUNNING' else f"{j['state']} {j['reason']}"
        print(f"| {j['id']} | {j['name']} | {j['run'].replace('runs/', '') or '–'} | {j['gpu'] or '–'} | {state} | "
              f"{j['elapsed']} | {progress or '–'} | {eta or '–'} |")
    if not data['jobs']:
        print('| – | no jobs | | | | | | |')
    print('\nLatest driver-log lines:')
    for name, last in data['logs'].items():
        for l in last:
            print(f'- `{name}`: {l}')


if __name__ == '__main__':
    main()
