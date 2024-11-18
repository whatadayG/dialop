        
        
import time
from pathlib import Path
        
dir = "/Users/georgiazhou/research_machine/dialop/dialop/checkpoints/debug_states"
dir = Path(dir)
checkpoints = list(dir.glob("best_path_*.pkl"))
if not checkpoints:
    print("No checkpoints found")
    exit()
# Convert to Path objects if they're strings
checkpoints = [Path(p) if isinstance(p, str) else p for p in checkpoints]
latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[2]))
checkpoint_time = latest.stat().st_mtime
print(checkpoint_time)
#if checkpoint_time > self.last_checkpoint_time:
 
