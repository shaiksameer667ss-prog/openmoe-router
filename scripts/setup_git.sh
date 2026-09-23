#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .git ]; then
  git init
fi

git add .
git status --short

echo
echo "Initial repository prepared. Commit with:"
echo '  git commit -m "chore: initialize OpenMoE-Router research scaffold"'
