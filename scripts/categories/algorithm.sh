#!/usr/bin/env bash

# Algorithm category handler for note creation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

handle_algorithm_category() {
    local title="$1"
    local slug="$2"
    local date="$3"
    local category="$4"

    local target_dir="notes/${category}"
    local template_file="templates/default-template.md"
    local target_file="${target_dir}/${slug}.md"

    # Ensure target directory exists
    ensure_target_directory "$target_dir"

    # Replace placeholders and create file
    replace_placeholders "$template_file" "$target_file" "$title" "$date" "$slug" "$category"

    # Print success message
    print_success_message "$target_file"
}
