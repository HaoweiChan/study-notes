#!/usr/bin/env bash

# Common utility functions for note creation scripts

# Get the current date in YYYY-MM-DD format
get_current_date() {
    date '+%Y-%m-%d'
}

# Generate a slug from a title
generate_slug() {
    local title="$1"
    echo "$title" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g' | sed -E 's/^-|-$//g'
}

# Create target directory if it doesn't exist
ensure_target_directory() {
    local target_dir="$1"
    mkdir -p "$target_dir" templates
}

# Replace placeholders in template file
replace_placeholders() {
    local template_file="$1"
    local target_file="$2"
    local title="$3"
    local date="$4"
    local slug="$5"
    local category="$6"

    local esc_title=$(printf '%s' "$title" | sed 's|/|\\/|g' | sed 's|&|\\\&|g')

    sed -e "s/{{TITLE}}/${esc_title}/" \
        -e "s/{{DATE}}/${date}/" \
        -e "s/{{SLUG}}/${slug}/" \
        -e "s/{{CATEGORY}}/${category}/" \
        -e "s/{{LEETCODE_URL}}//" \
        -e "s/{{DIFFICULTY}}//" \
        -e "s/{{LEETCODE_TOPICS}}//" \
        -e "s/{{SUMMARY}}//" \
        -e "s/{{PROBLEM_DESCRIPTION}}//" \
        -e "s/{{APPROACH}}//" \
        -e "s/{{TIME_COMPLEXITY}}//" \
        -e "s/{{SPACE_COMPLEXITY}}//" \
        -e "s/{{INSIGHTS}}//" \
        -e "s/{{SOLUTION_CODE}}//" \
        -e "s/{{WALKTHROUGH}}//" \
        -e "s/{{EDGE_CASES}}//" \
        -e "s/{{RELATED_PROBLEMS}}//" \
        "$template_file" > "$target_file"
}

# Print success message
print_success_message() {
    local target_file="$1"
    echo "Created $target_file"
}
