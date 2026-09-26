#!/bin/bash
set -e

# If a commit message was provided as an argument, use it.
# Otherwise, prompt for one or use a timestamp.
if [ -n "$1" ]; then
    MSG="$1"
else
    read -p "Enter commit message (press Enter for timestamp): " MSG
    if [ -z "$MSG" ]; then
        MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
fi

echo "📦 Staging changes..."
git add .

echo "💾 Committing: $MSG"
git commit -m "$MSG"

echo "🚀 Pushing to GitHub..."
git push

echo "✅ Successfully pushed to GitHub!"
