#!/bin/bash
# Push Wingspan to https://github.com/marlonbarrios/spread_your_wings
set -e
cd "$(dirname "$0")"

if git status --porcelain | grep -q .; then
  echo "Uncommitted changes — commit first, then push."
  exit 1
fi

AHEAD="$(git rev-list --count origin/main..main 2>/dev/null || echo 0)"
if [ "$AHEAD" = "0" ]; then
  echo "Already up to date with origin/main."
  git status -sb
  exit 0
fi

echo "Commits to push:"
git log --oneline origin/main..main

if [ -n "$1" ]; then
  git push "https://marlonbarrios:${1}@github.com/marlonbarrios/spread_your_wings.git" main
else
  git push origin main
fi

echo "Done: https://github.com/marlonbarrios/spread_your_wings"
