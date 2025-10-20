#!/usr/bin/env bash

# LeetCode category handler for note creation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

handle_leetcode_category() {
    local title="$1"
    local slug_arg="$2"
    local date="$3"
    local category="$4"

    local target_dir="notes/leetcode"
    local template_file="templates/leetcode-template.md"
    local problem_number=""
    local final_slug=""

    # Ensure target directory exists
    ensure_target_directory "$target_dir"

    # Handle problem number logic
    if [ -n "$slug_arg" ]; then
        # Check if SLUG_ARG is a number (problem number) or a slug
        if echo "$slug_arg" | grep -qE '^[0-9]+$'; then
            problem_number="$slug_arg"
            # Generate slug from title for the filename part
            local base_slug=$(generate_slug "$title")
            final_slug="${problem_number}-${base_slug}"
        else
            # Use provided slug as-is, but still need problem number
            echo -n "Enter LeetCode problem number: "
            read problem_number
            if [ -z "$problem_number" ]; then
                echo "Problem number is required for LeetCode notes"
                exit 1
            fi
            final_slug="$slug_arg"
        fi
    else
        # Prompt for problem number if not provided
        echo -n "Enter LeetCode problem number: "
        read problem_number
        if [ -z "$problem_number" ]; then
            echo "Problem number is required for LeetCode notes"
            exit 1
        fi
        # Generate slug from title
        final_slug=$(generate_slug "$title")
    fi

    # Validate problem number
    if ! echo "$problem_number" | grep -qE '^[0-9]+$'; then
        echo "Problem number must be numeric"
        exit 1
    fi

    # Use problem number in the filename
    final_slug="${problem_number}-${final_slug}"
    local target_file="${target_dir}/${final_slug}.md"

    # Replace placeholders and create file
    # For LeetCode, we need to set the slug to the final_slug (without problem number for frontmatter)
    local frontmatter_slug=$(generate_slug "$title")
    replace_placeholders "$template_file" "$target_file" "$title" "$date" "$frontmatter_slug" "leetcode"

    # Print success message
    print_success_message "$target_file"
}
