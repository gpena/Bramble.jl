#!/usr/bin/env bash
# Deletes the entries on `main` that a julia-actions/cache key supersedes, before that
# action's post step saves the new one.
#
#   delete_superseded_cache.sh <full cache key>     # the action's `cache-key` output
#
# The key ends in `;run_id=<id>;run_attempt=<n>`. Every entry on `main` with the same key
# up to `;run_id=` and another run id is deleted; this run's own entry stays, since the
# shards of one matrix job share a key and the first to finish has already saved it.
#
# julia-actions/cache never deletes on the default branch, so without this the old and the
# new entry coexist until nightly.yml's `prune-caches` job, and that transient doubling took
# the repository over its 10 GB quota: GitHub then evicts the least recently used entry,
# which was the docs one. `DRY_RUN=1` lists without deleting. Needs `GH_TOKEN` with
# `actions: write`, and `GITHUB_REPOSITORY`.
set -euo pipefail

key=$1
case "$key" in
  *";run_id="*) ;;
  *) echo "not a julia-actions/cache key: $key" >&2; exit 1 ;;
esac
prefix="${key%%;run_id=*};run_id="
run="${key#"$prefix"}"
run="${run%%;*}"

gh api -X GET --paginate "repos/$GITHUB_REPOSITORY/actions/caches" \
  -f ref=refs/heads/main -f key="$prefix" -F per_page=100 \
  --jq '.actions_caches[] | "\(.id) \(.key)"' |
  while read -r id entry; do
    case "$entry" in
      "$prefix$run;"*) continue ;;
      "$prefix"*) ;;
      *) continue ;;
    esac
    echo "Deleting $entry"
    [ -n "${DRY_RUN:-}" ] || gh api -X DELETE "repos/$GITHUB_REPOSITORY/actions/caches/$id" || true
  done
