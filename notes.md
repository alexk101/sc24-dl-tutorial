
Ideally, you want both because:

1. Python buffer handles normal cases
2. SLURM signal is backup if Python fails
3. Different time scales provide layered protection

Job Timeline:
[....Training....][Python Buffer (5min)][SLURM Signal (90s)][KILL]
                  ^                     ^                    ^
                  |                     |                    |
        Graceful checkpoint     Emergency cleanup     Force terminate
