#!/usr/bin/env julia
#
# v2 -> v3 rename of the benchmark baseline keys (gpena/Bramble.jl#211).
#
#     julia --project=. dev/rename_baselines.jl [--check]
#
# `benchmark/baselines/*.json` are settled measurements: a baseline is written once, at a
# release tag, and never re-run. A renamed benchmark therefore has to be renamed *inside* the
# old baselines too, or the trend line in `docs/src/benchmarks.md` splits into a pre-rename
# series and a post-rename one with nothing joining them.
#
# Only the three keys below carry a renamed operator, and they appear in all 15 baselines.
# The rename is a text substitution of the quoted key rather than a JSON round-trip on
# purpose: re-encoding would rewrite every float in the file, and these numbers are the
# record. (Checked first: the baselines store raw UTF-8, not `\uXXXX` escapes, so the key
# text is there to be matched. A file written with escapes would need the JSON path instead.)

const KEY_RENAMES = [
    # Longest first, so the bare `"∇₋ₕ"` rule cannot claim the composite key's prefix.
    "\"∇₋ₕ (3 components)\"" => "\"∇ₕ (3 components)\"",
    "\"∇₋ₕ\"" => "\"∇ₕ\"",
    "\"M₋ₓ\"" => "\"Mₓ\""
]

function main(args)
    check_only = "--check" in args
    files = sort(filter(f -> endswith(f, ".json"), readdir("benchmark/baselines"; join = true)))
    isempty(files) && error("no baselines found under benchmark/baselines")

    total = 0
    for path in files
        text = read(path, String)

        for (_, new) in KEY_RENAMES
            occursin(new, text) &&
                error("$path already holds $new -- refusing to rename twice")
        end

        hits = sum(count(old, text) for (old, _) in KEY_RENAMES)
        hits == 0 && continue

        new_text = replace(text, KEY_RENAMES...)
        for (old, _) in KEY_RENAMES
            count(old, new_text) == 0 || error("$path: $old survived the rename")
        end

        total += hits
        check_only || write(path, new_text)
        println(check_only ? "would rename" : "renamed", " $hits key(s) in $path")
    end

    println("\n$(check_only ? "would rename" : "renamed") $total keys across $(length(files)) baselines")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
