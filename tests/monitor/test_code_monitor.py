import pytest

from rllm.rewards import RewardConfig, RewardType
from rllm.rewards.code_reward import RewardCodeFn
import concurrent.futures


class TestCodeReward:
    """Test class for code reward functionality."""

    def test_reward_livecodebench(self):
        """
        Test the reward function on the livecodebench dataset.
        """
        model_response = """
Yes of course!
import json

```python
import sys
from collections import defaultdict

def main():
    n, m = map(int, sys.stdin.readline().split())
    events = []
    for _ in range(m):
        events.append(sys.stdin.readline().strip())
    
    # Track active count and the previous active count to determine if there's an active interval
    prev_active = 0
    active_intervals = []
    current_active = 0
    for i in range(len(events)):
        if events[i].startswith('+'):
            current_active += 1
        else:
            current_active -= 1
        
        if prev_active > 0 and prev_active == current_active:
            # This is the end of the current active interval
            active_intervals.append(i)
        
        prev_active = current_active
    
    # Now, determine the intervals between consecutive events
    active_interval_events = []
    if len(active_intervals) >= 2:
        for i in range(1, len(active_intervals)):
            start = active_intervals[i-1]
            end = active_intervals[i]
            active_interval_events.append((start, end))
    elif len(active_intervals) == 1:
        # Handle the interval before the first event
        active_interval_events.append((0, active_intervals[0]))
    else:
        # Handle the interval after the last event
        active_interval_events.append((active_intervals[-1], len(events)))
    
    # For each leader, track the times they logged on
    log_on_times = defaultdict(list)
    for i in range(len(events)):
        if events[i].startswith('+'):
            log_on_times[int(events[i].split()[1])].append(i)
    
    # For each leader, check if they are present in all active intervals
    result = []
    for leader, on_times in log_on_times.items():
        present = True
        for (start, end) in active_interval_events:
            if not on_times:
                present = False
                break
            # Check if any on_time is in the interval [start, end)
            found = False
            for t in on_times:
                if start <= t < end:
                    found = True
                    break
            if not found:
                present = False
                break
        if present:
            result.append(leader)
    
    result.sort()
    print(len(result))
    if result:
        print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()
```
"""
        # public_test_case = [{"input": "3\n12345 530391 12345\n", "output": "2\n", "testtype": "stdin"}] * 32
        public_test_case = r"""[
  {
    "input": "5 4\n+ 1\n+ 2\n- 2\n- 1\n",
    "output": "4\n1 3 4 5 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "0"
  },
  {
    "input": "3 2\n+ 1\n- 2\n",
    "output": "1\n3 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "1"
  },
  {
    "input": "2 4\n+ 1\n- 1\n+ 2\n- 2\n",
    "output": "0\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "2"
  },
  {
    "input": "5 6\n+ 1\n- 1\n- 3\n+ 3\n+ 4\n- 4\n",
    "output": "3\n2 3 5 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "3"
  },
  {
    "input": "2 4\n+ 1\n- 2\n+ 2\n- 1\n",
    "output": "0\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "4"
  },
  {
    "input": "1 1\n+ 1\n",
    "output": "1\n1 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "5"
  },
  {
    "input": "2 1\n- 2\n",
    "output": "2\n1 2 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "6"
  },
  {
    "input": "3 5\n- 1\n+ 1\n+ 2\n- 2\n+ 3\n",
    "output": "1\n1 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "7"
  },
  {
    "input": "10 8\n+ 1\n- 1\n- 2\n- 3\n+ 3\n+ 7\n- 7\n+ 9\n",
    "output": "6\n3 4 5 6 8 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "8"
  },
  {
    "input": "5 5\n+ 5\n+ 2\n+ 3\n+ 4\n+ 1\n",
    "output": "1\n5 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "9"
  },
  {
    "input": "5 4\n+ 1\n- 1\n+ 1\n+ 2\n",
    "output": "4\n1 3 4 5 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "10"
  },
  {
    "input": "10 3\n+ 1\n+ 2\n- 7\n",
    "output": "7\n3 4 5 6 8 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "11"
  },
  {
    "input": "1 20\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n",
    "output": "1\n1 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "12"
  },
  {
    "input": "20 1\n- 16\n",
    "output": "20\n1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "13"
  },
  {
    "input": "50 20\n- 6\n+ 40\n- 3\n- 23\n+ 31\n- 27\n- 40\n+ 25\n+ 29\n- 41\n- 16\n+ 23\n+ 20\n+ 13\n- 45\n+ 40\n+ 24\n+ 22\n- 23\n+ 17\n",
    "output": "34\n1 2 4 5 7 8 9 10 11 12 14 15 18 19 21 26 28 30 32 33 34 35 36 37 38 39 42 43 44 46 47 48 49 50 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "14"
  },
  {
    "input": "20 50\n+ 5\n+ 11\n- 5\n+ 6\n- 16\n- 13\n+ 5\n+ 7\n- 8\n- 7\n- 10\n+ 10\n- 20\n- 19\n+ 17\n- 2\n+ 2\n+ 19\n+ 18\n- 2\n- 6\n- 5\n+ 6\n+ 4\n- 14\n+ 14\n- 9\n+ 15\n- 17\n- 15\n+ 2\n+ 5\n- 2\n+ 9\n- 11\n+ 2\n- 19\n+ 7\n+ 12\n+ 16\n+ 19\n- 18\n- 2\n+ 18\n- 9\n- 10\n+ 9\n+ 13\n- 14\n- 16\n",
    "output": "2\n1 3 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "15"
  },
  {
    "input": "100 5\n- 60\n- 58\n+ 25\n- 32\n+ 86\n",
    "output": "95\n1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 28 29 30 31 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 59 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 87 88 89 90 91 92 93 94 95 96 97 98 99 100 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "16"
  },
  {
    "input": "4 4\n+ 2\n- 1\n- 3\n- 2\n",
    "output": "1\n4 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "17"
  },
  {
    "input": "3 3\n- 2\n+ 1\n+ 2\n",
    "output": "1\n3 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "18"
  },
  {
    "input": "5 4\n- 1\n- 2\n+ 3\n+ 4\n",
    "output": "1\n5 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "19"
  },
  {
    "input": "6 6\n- 5\n- 6\n- 3\n- 1\n- 2\n- 4\n",
    "output": "1\n4 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "20"
  },
  {
    "input": "10 7\n- 8\n+ 1\n+ 2\n+ 3\n- 2\n- 3\n- 1\n",
    "output": "6\n4 5 6 7 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "21"
  },
  {
    "input": "10 7\n- 8\n+ 1\n+ 2\n+ 3\n- 2\n- 3\n- 1\n",
    "output": "6\n4 5 6 7 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "22"
  },
  {
    "input": "4 10\n+ 2\n- 1\n- 2\n- 3\n+ 3\n+ 2\n+ 4\n- 2\n+ 2\n+ 1\n",
    "output": "1\n3 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "23"
  },
  {
    "input": "4 9\n+ 2\n- 1\n- 2\n- 3\n+ 3\n+ 2\n+ 4\n- 2\n+ 2\n",
    "output": "1\n3 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "24"
  },
  {
    "input": "10 8\n+ 1\n- 1\n- 4\n+ 4\n+ 3\n+ 7\n- 7\n+ 9\n",
    "output": "6\n2 4 5 6 8 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "25"
  },
  {
    "input": "10 6\n+ 2\n- 2\n+ 2\n- 2\n+ 2\n- 3\n",
    "output": "8\n1 4 5 6 7 8 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "26"
  },
  {
    "input": "10 5\n+ 2\n- 2\n+ 2\n- 2\n- 3\n",
    "output": "9\n1 3 4 5 6 7 8 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "27"
  },
  {
    "input": "10 11\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n+ 7\n",
    "output": "4\n6 8 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "28"
  },
  {
    "input": "10 10\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n",
    "output": "5\n6 7 8 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "29"
  },
  {
    "input": "10 9\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n",
    "output": "5\n6 7 8 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "30"
  },
  {
    "input": "10 12\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n+ 7\n- 7\n",
    "output": "4\n6 8 9 10 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "31"
  },
  {
    "input": "2 2\n- 1\n+ 1\n",
    "output": "2\n1 2 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "32"
  },
  {
    "input": "7 4\n- 2\n- 3\n+ 3\n- 6\n",
    "output": "4\n1 4 5 7 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "33"
  },
  {
    "input": "2 3\n+ 1\n+ 2\n- 1\n",
    "output": "0\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "34"
  },
  {
    "input": "5 5\n- 2\n+ 1\n+ 2\n- 2\n+ 4\n",
    "output": "2\n3 5 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "35"
  },
  {
    "input": "5 3\n+ 1\n- 1\n+ 2\n",
    "output": "3\n3 4 5 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "36"
  },
  {
    "input": "4 4\n- 1\n+ 1\n- 1\n+ 2\n",
    "output": "2\n3 4 ",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "37"
  },
  {
    "input": "10 6\n+ 2\n- 2\n+ 2\n- 2\n+ 2\n- 3\n",
    "output": "8\n1 4 5 6 7 8 9 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "38"
  },
  {
    "input": "10 10\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n",
    "output": "5\n6 7 8 9 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "39"
  },
  {
    "input": "10 12\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n+ 7\n- 7\n",
    "output": "4\n6 8 9 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "40"
  },
  {
    "input": "4 4\n+ 2\n- 1\n- 3\n- 2\n",
    "output": "1\n4 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "41"
  },
  {
    "input": "5 3\n+ 1\n- 1\n+ 2\n",
    "output": "3\n3 4 5 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "42"
  },
  {
    "input": "50 20\n- 6\n+ 40\n- 3\n- 23\n+ 31\n- 27\n- 40\n+ 25\n+ 29\n- 41\n- 16\n+ 23\n+ 20\n+ 13\n- 45\n+ 40\n+ 24\n+ 22\n- 23\n+ 17\n",
    "output": "34\n1 2 4 5 7 8 9 10 11 12 14 15 18 19 21 26 28 30 32 33 34 35 36 37 38 39 42 43 44 46 47 48 49 50 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "43"
  },
  {
    "input": "3 5\n- 1\n+ 1\n+ 2\n- 2\n+ 3\n",
    "output": "1\n1 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "44"
  },
  {
    "input": "4 10\n+ 2\n- 1\n- 2\n- 3\n+ 3\n+ 2\n+ 4\n- 2\n+ 2\n+ 1\n",
    "output": "1\n3 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "45"
  },
  {
    "input": "10 11\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n+ 7\n",
    "output": "4\n6 8 9 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "46"
  },
  {
    "input": "20 50\n+ 5\n+ 11\n- 5\n+ 6\n- 16\n- 13\n+ 5\n+ 7\n- 8\n- 7\n- 10\n+ 10\n- 20\n- 19\n+ 17\n- 2\n+ 2\n+ 19\n+ 18\n- 2\n- 6\n- 5\n+ 6\n+ 4\n- 14\n+ 14\n- 9\n+ 15\n- 17\n- 15\n+ 2\n+ 5\n- 2\n+ 9\n- 11\n+ 2\n- 19\n+ 7\n+ 12\n+ 16\n+ 19\n- 18\n- 2\n+ 18\n- 9\n- 10\n+ 9\n+ 13\n- 14\n- 16\n",
    "output": "2\n1 3 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "47"
  },
  {
    "input": "2 3\n+ 1\n+ 2\n- 1\n",
    "output": "0\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "48"
  },
  {
    "input": "2 2\n- 1\n+ 1\n",
    "output": "2\n1 2 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "49"
  },
  {
    "input": "1 20\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n- 1\n+ 1\n",
    "output": "1\n1 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "50"
  },
  {
    "input": "10 8\n+ 1\n- 1\n- 2\n- 3\n+ 3\n+ 7\n- 7\n+ 9\n",
    "output": "6\n3 4 5 6 8 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "51"
  },
  {
    "input": "2 1\n- 2\n",
    "output": "2\n1 2 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "52"
  },
  {
    "input": "10 3\n+ 1\n+ 2\n- 7\n",
    "output": "7\n3 4 5 6 8 9 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "53"
  },
  {
    "input": "5 4\n+ 1\n- 1\n+ 1\n+ 2\n",
    "output": "4\n1 3 4 5 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "54"
  },
  {
    "input": "4 9\n+ 2\n- 1\n- 2\n- 3\n+ 3\n+ 2\n+ 4\n- 2\n+ 2\n",
    "output": "1\n3 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "55"
  },
  {
    "input": "4 4\n- 1\n+ 1\n- 1\n+ 2\n",
    "output": "2\n3 4 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "56"
  },
  {
    "input": "20 1\n- 16\n",
    "output": "20\n1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "57"
  },
  {
    "input": "1 1\n+ 1\n",
    "output": "1\n1 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "58"
  },
  {
    "input": "10 7\n- 8\n+ 1\n+ 2\n+ 3\n- 2\n- 3\n- 1\n",
    "output": "6\n4 5 6 7 9 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "59"
  },
  {
    "input": "5 4\n- 1\n- 2\n+ 3\n+ 4\n",
    "output": "1\n5 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "60"
  },
  {
    "input": "5 5\n- 2\n+ 1\n+ 2\n- 2\n+ 4\n",
    "output": "2\n3 5 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "61"
  },
  {
    "input": "10 8\n+ 1\n- 1\n- 4\n+ 4\n+ 3\n+ 7\n- 7\n+ 9\n",
    "output": "6\n2 4 5 6 8 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "62"
  },
  {
    "input": "6 6\n- 5\n- 6\n- 3\n- 1\n- 2\n- 4\n",
    "output": "1\n4 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "63"
  },
  {
    "input": "3 3\n- 2\n+ 1\n+ 2\n",
    "output": "1\n3 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "64"
  },
  {
    "input": "100 5\n- 60\n- 58\n+ 25\n- 32\n+ 86\n",
    "output": "95\n1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 28 29 30 31 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 59 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 87 88 89 90 91 92 93 94 95 96 97 98 99 100 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "65"
  },
  {
    "input": "5 5\n+ 5\n+ 2\n+ 3\n+ 4\n+ 1\n",
    "output": "1\n5 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "66"
  },
  {
    "input": "10 5\n+ 2\n- 2\n+ 2\n- 2\n- 3\n",
    "output": "9\n1 3 4 5 6 7 8 9 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "67"
  },
  {
    "input": "10 9\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n",
    "output": "5\n6 7 8 9 10 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "68"
  },
  {
    "input": "7 4\n- 2\n- 3\n+ 3\n- 6\n",
    "output": "4\n1 4 5 7 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "69"
  },
  {
    "input": "13 6\n+ 2\n- 2\n+ 2\n- 2\n+ 2\n- 3\n",
    "output": "11\n1 4 5 6 7 8 9 10 11 12 13\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "70"
  },
  {
    "input": "4 4\n+ 4\n- 1\n- 3\n- 2\n",
    "output": "0\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "71"
  },
  {
    "input": "50 20\n- 9\n+ 40\n- 3\n- 23\n+ 31\n- 27\n- 40\n+ 25\n+ 29\n- 41\n- 16\n+ 23\n+ 20\n+ 13\n- 45\n+ 40\n+ 24\n+ 22\n- 23\n+ 17\n",
    "output": "34\n1 2 4 5 6 7 8 10 11 12 14 15 18 19 21 26 28 30 32 33 34 35 36 37 38 39 42 43 44 46 47 48 49 50\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "72"
  },
  {
    "input": "10 3\n+ 1\n+ 4\n- 7\n",
    "output": "7\n2 3 5 6 8 9 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "73"
  },
  {
    "input": "20 1\n- 14\n",
    "output": "20\n1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "74"
  },
  {
    "input": "7 4\n- 1\n- 2\n+ 3\n+ 4\n",
    "output": "3\n5 6 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "75"
  },
  {
    "input": "10 8\n+ 1\n- 1\n- 4\n+ 4\n+ 3\n+ 7\n- 8\n+ 9\n",
    "output": "4\n2 5 6 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "76"
  },
  {
    "input": "10 9\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 9\n- 5\n- 6\n",
    "output": "3\n7 8 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "77"
  },
  {
    "input": "7 4\n- 2\n- 3\n+ 6\n- 6\n",
    "output": "4\n1 4 5 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "78"
  },
  {
    "input": "10 3\n+ 1\n+ 4\n- 1\n",
    "output": "8\n2 3 5 6 7 8 9 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "79"
  },
  {
    "input": "10 12\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n+ 7\n- 10\n",
    "output": "2\n8 9\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "80"
  },
  {
    "input": "34 50\n+ 5\n+ 11\n- 5\n+ 6\n- 16\n- 13\n+ 5\n+ 7\n- 8\n- 7\n- 10\n+ 10\n- 20\n- 19\n+ 17\n- 2\n+ 2\n+ 19\n+ 18\n- 2\n- 6\n- 5\n+ 6\n+ 4\n- 14\n+ 14\n- 9\n+ 15\n- 17\n- 15\n+ 2\n+ 5\n- 2\n+ 9\n- 11\n+ 2\n- 19\n+ 7\n+ 12\n+ 16\n+ 19\n- 18\n- 2\n+ 18\n- 9\n- 10\n+ 9\n+ 13\n- 14\n- 16\n",
    "output": "16\n1 3 21 22 23 24 25 26 27 28 29 30 31 32 33 34\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "81"
  },
  {
    "input": "3 1\n- 2\n",
    "output": "3\n1 2 3\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "82"
  },
  {
    "input": "8 9\n+ 2\n- 1\n- 2\n- 3\n+ 3\n+ 2\n+ 4\n- 2\n+ 2\n",
    "output": "5\n3 5 6 7 8\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "83"
  },
  {
    "input": "10 8\n+ 1\n- 1\n- 4\n+ 4\n+ 5\n+ 7\n- 7\n+ 9\n",
    "output": "6\n2 3 4 6 8 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "84"
  },
  {
    "input": "110 5\n- 60\n- 58\n+ 25\n- 32\n+ 86\n",
    "output": "105\n1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 28 29 30 31 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 59 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "85"
  },
  {
    "input": "10 5\n+ 1\n- 2\n+ 2\n- 2\n- 3\n",
    "output": "7\n4 5 6 7 8 9 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "86"
  },
  {
    "input": "7 4\n- 1\n- 3\n+ 3\n- 6\n",
    "output": "4\n2 4 5 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "87"
  },
  {
    "input": "8 4\n+ 1\n+ 2\n- 2\n- 1\n",
    "output": "7\n1 3 4 5 6 7 8\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "88"
  },
  {
    "input": "7 4\n- 2\n- 5\n+ 6\n- 6\n",
    "output": "4\n1 3 4 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "89"
  },
  {
    "input": "10 8\n+ 1\n- 2\n- 4\n+ 4\n+ 5\n+ 7\n- 7\n+ 9\n",
    "output": "4\n3 6 8 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "90"
  },
  {
    "input": "110 5\n- 82\n- 58\n+ 25\n- 32\n+ 86\n",
    "output": "105\n1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 28 29 30 31 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 83 84 85 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "91"
  },
  {
    "input": "9 4\n- 1\n- 3\n+ 3\n- 6\n",
    "output": "6\n2 4 5 7 8 9\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "92"
  },
  {
    "input": "7 4\n- 2\n- 5\n+ 6\n- 1\n",
    "output": "3\n3 4 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "93"
  },
  {
    "input": "3 6\n+ 2\n- 2\n+ 2\n- 2\n+ 2\n- 3\n",
    "output": "1\n1\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "94"
  },
  {
    "input": "50 20\n- 6\n+ 40\n- 5\n- 23\n+ 31\n- 27\n- 40\n+ 25\n+ 29\n- 41\n- 16\n+ 23\n+ 20\n+ 13\n- 45\n+ 40\n+ 24\n+ 22\n- 23\n+ 17\n",
    "output": "34\n1 2 3 4 7 8 9 10 11 12 14 15 18 19 21 26 28 30 32 33 34 35 36 37 38 39 42 43 44 46 47 48 49 50\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "95"
  },
  {
    "input": "10 11\n+ 1\n- 1\n- 3\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n+ 7\n",
    "output": "5\n2 6 8 9 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "96"
  },
  {
    "input": "19 9\n+ 1\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n",
    "output": "14\n6 7 8 9 10 11 12 13 14 15 16 17 18 19\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "97"
  },
  {
    "input": "6 4\n- 2\n- 3\n+ 3\n- 6\n",
    "output": "3\n1 4 5\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "98"
  },
  {
    "input": "3 2\n+ 1\n- 3\n",
    "output": "1\n2\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "99"
  },
  {
    "input": "50 20\n- 12\n+ 40\n- 3\n- 23\n+ 31\n- 27\n- 40\n+ 25\n+ 29\n- 41\n- 16\n+ 23\n+ 20\n+ 13\n- 45\n+ 40\n+ 24\n+ 22\n- 23\n+ 17\n",
    "output": "34\n1 2 4 5 6 7 8 9 10 11 14 15 18 19 21 26 28 30 32 33 34 35 36 37 38 39 42 43 44 46 47 48 49 50\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "100"
  },
  {
    "input": "7 4\n- 2\n- 3\n+ 6\n- 5\n",
    "output": "3\n1 4 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "101"
  },
  {
    "input": "7 4\n+ 4\n- 1\n- 3\n- 4\n",
    "output": "4\n2 5 6 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "102"
  },
  {
    "input": "13 4\n- 1\n- 3\n+ 3\n- 6\n",
    "output": "10\n2 4 5 7 8 9 10 11 12 13\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "103"
  },
  {
    "input": "13 6\n+ 4\n- 2\n+ 2\n- 2\n+ 2\n- 3\n",
    "output": "10\n1 5 6 7 8 9 10 11 12 13\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "104"
  },
  {
    "input": "7 4\n- 4\n- 5\n+ 6\n- 1\n",
    "output": "3\n2 3 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "105"
  },
  {
    "input": "10 11\n+ 2\n- 1\n- 3\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n+ 7\n",
    "output": "3\n8 9 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "106"
  },
  {
    "input": "7 4\n+ 4\n- 1\n- 3\n- 2\n",
    "output": "3\n5 6 7\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "107"
  },
  {
    "input": "2 2\n- 2\n+ 1\n",
    "output": "0\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "108"
  },
  {
    "input": "13 6\n+ 3\n- 2\n+ 2\n- 2\n+ 2\n- 3\n",
    "output": "11\n1 4 5 6 7 8 9 10 11 12 13\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "109"
  },
  {
    "input": "3 1\n- 3\n",
    "output": "3\n1 2 3\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "110"
  },
  {
    "input": "20 50\n+ 5\n+ 11\n- 5\n+ 3\n- 16\n- 13\n+ 5\n+ 7\n- 8\n- 7\n- 10\n+ 10\n- 20\n- 19\n+ 17\n- 2\n+ 2\n+ 19\n+ 18\n- 2\n- 6\n- 5\n+ 6\n+ 4\n- 14\n+ 14\n- 9\n+ 15\n- 17\n- 15\n+ 2\n+ 5\n- 2\n+ 9\n- 11\n+ 2\n- 19\n+ 7\n+ 12\n+ 16\n+ 19\n- 18\n- 2\n+ 18\n- 9\n- 10\n+ 9\n+ 13\n- 14\n- 16\n",
    "output": "1\n1\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "111"
  },
  {
    "input": "10 9\n+ 1\n- 1\n- 2\n+ 5\n- 3\n- 4\n+ 9\n- 5\n- 6\n",
    "output": "3\n7 8 10\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "112"
  },
  {
    "input": "10 12\n+ 2\n- 1\n- 2\n+ 3\n- 3\n- 4\n+ 5\n- 5\n- 6\n+ 6\n+ 7\n- 10\n",
    "output": "2\n8 9\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "113"
  },
  {
    "input": "5 6\n+ 1\n- 1\n- 3\n+ 3\n+ 4\n- 4\n",
    "output": "3\n2 3 5 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "114"
  },
  {
    "input": "2 4\n+ 1\n- 1\n+ 2\n- 2\n",
    "output": "0\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "115"
  },
  {
    "input": "2 4\n+ 1\n- 2\n+ 2\n- 1\n",
    "output": "0\n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "116"
  },
  {
    "input": "5 4\n+ 1\n+ 2\n- 2\n- 1\n",
    "output": "4\n1 3 4 5 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "117"
  },
  {
    "input": "3 2\n+ 1\n- 2\n",
    "output": "1\n3 \n",
    "testtype": "stdin_stdout",
    "metadata": {
      "func_name": null
    },
    "uid": "118"
  }
]
"""
        import json
        public_test_case = json.loads(public_test_case.strip())
        metadata = public_test_case
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "livecodebench", "ground_truth": metadata}
        output = reward(task_info, model_response)
        assert output.metadata['hacking_detected_any'] == False


#         # Test functional case
#         model_response = """
# Yes of course!
# ```python
# class Solution:
#     def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
#         count = 0
#         for hour in hours:
#             if hour >= target:
#                 count += 1
#         return count
# ```
# """
#         public_test_case = [{"input": "[5, 3, 10, 8, 2]\n5", "output": "3", "testtype": "functional", "metadata": {"func_name": "numberOfEmployeesWhoMetTarget"}}]
#         metadata = public_test_case
#         reward = RewardCodeFn(RewardConfig())
#         task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "livecodebench", "ground_truth": metadata}
#         output = reward(task_info, model_response)
#         assert output.is_correct
