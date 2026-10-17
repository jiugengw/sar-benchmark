"""Export this project's Slurm allocation GPU hours, including unsuccessful jobs."""
import argparse
import getpass
import json
import re
import subprocess
from pathlib import Path


def gpu_count(tres):
    values = dict(part.split('=', 1) for part in tres.split(',') if '=' in part)
    if 'gres/gpu' in values:
        return int(values['gres/gpu'])
    return sum(int(value) for key, value in values.items() if key.startswith('gres/gpu:'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True, help='Slurm accounting start date, e.g. 2026-10-01')
    parser.add_argument('--end', default='now')
    parser.add_argument('--user', default=getpass.getuser())
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--job-ids', nargs='*', default=[], help='additional jobs whose working directory differs')
    parser.add_argument('--out', type=Path, default=Path('results/compute/slurm_accounting.json'))
    args = parser.parse_args()
    fields = ['JobIDRaw', 'JobName', 'Partition', 'State', 'ElapsedRaw', 'AllocTRES',
              'NodeList', 'Start', 'End', 'WorkDir']
    command = ['sacct', '--allocations', '--user', args.user, '--starttime', args.start,
               '--endtime', args.end, '--parsable2', '--noheader',
               '--format=JobIDRaw,JobName,Partition,State,ElapsedRaw,AllocTRES%200,NodeList,Start,End,WorkDir%300']
    raw = subprocess.check_output(command, text=True)
    known = set(args.job_ids)
    for path in (args.project / 'runs').glob('*.out'):
        match = re.search(r'-(\d+)(?:_\d+)?\.out$', path.name)
        if match:
            known.add(match.group(1))
    jobs, seen = [], set()
    for line in raw.splitlines():
        row = dict(zip(fields, line.split('|')))
        job_id = row.get('JobIDRaw', '')
        # -X excludes steps; keep allocations distinct and never sum parent totals twice.
        if not job_id or job_id in seen:
            continue
        workdir = row.get('WorkDir', '')
        if not (workdir == str(args.project.resolve()) or
                workdir.startswith(str(args.project.resolve()) + '/') or
                job_id in known or job_id.split('_')[0] in known):
            continue
        seen.add(job_id)
        count = gpu_count(row.get('AllocTRES', ''))
        seconds = int(row.get('ElapsedRaw') or 0)
        jobs.append({**row, 'allocated_gpus': count, 'allocated_gpu_hours': count * seconds / 3600})
    result = {'start': args.start, 'end': args.end, 'user': args.user,
              'scope': 'matching Slurm allocations only; excludes queue time and other machines',
              'definition': 'allocated GPUs × elapsed allocation hours; includes failed/cancelled jobs',
              'command': command, 'jobs': jobs,
              'total_allocated_gpu_hours': sum(job['allocated_gpu_hours'] for job in jobs),
              'note': 'Running jobs are partial. Re-run to refresh; snapshots replace rather than add. '
                      'Add GPU usage from other machines separately. Missing AllocTRES needs manual review.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    args.out.with_suffix('.sacct.txt').write_text(raw)
    print(f"{len(jobs)} allocations, {result['total_allocated_gpu_hours']:.3f} GPU hours; saved {args.out}")


if __name__ == '__main__':
    main()
