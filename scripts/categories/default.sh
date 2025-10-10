#!/usr/bin/env bash

# Default category handler for note creation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

handle_default_category() {
    local title="$1"
    local slug="$2"
    local date="$3"
    local category="$4"  # This is the key addition - we need the category

    local target_dir="notes/${category}"

    # Map categories to their templates
    case "$category" in
        "leetcode")
            template_file="templates/leetcode-template.md"
            ;;
        "system-design")
            template_file="templates/system-design-template.md"
            ;;
        *)
            template_file="templates/default-template.md"
            ;;
    esac

    local target_file="${target_dir}/${slug}.md"

    # Ensure target directory exists
    ensure_target_directory "$target_dir"

    # Replace placeholders and create file
    replace_placeholders "$template_file" "$target_file" "$title" "$date" "$slug" "$category"

    # Print success message
    print_success_message "$target_file"
}
