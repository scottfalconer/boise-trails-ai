git log --all --date=short --pretty=format:'%h %ad %s' --since='2025-06-09' --until='2025-06-22' | rg -i 'test|fix|bug|error|unplanned|one-way|cluster|route|planner|memory|oom|import|dependency'

