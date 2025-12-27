#!/bin/bash
set -e

# Directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
WEB_DIR="$PROJECT_ROOT/web"
DOCS_DIR="$PROJECT_ROOT/docs"

echo "🚀 Starting deployment..."

# 1. Update artifacts in web/public
echo "📦 Updating artifacts..."
mkdir -p "$WEB_DIR/public/artifacts"
cp "$DOCS_DIR/artifacts/"*.json "$WEB_DIR/public/artifacts/"

# 1b. Copy Notes to web/public
echo "📝 Copying notes content..."
mkdir -p "$WEB_DIR/public/notes"
# Remove old notes to ensure clean state
rm -rf "$WEB_DIR/public/notes/"*
cp -r "$PROJECT_ROOT/notes/"* "$WEB_DIR/public/notes/"

# 2. Build React App
echo "🏗️  Building Web App..."
cd "$WEB_DIR"
npm install
npm run build

# 3. Deploy to docs/
echo "deg  Deploying to docs/..."
# Remove old files in docs, but keep artifacts folder for safety (though dist has it)
# Actually, safest to clear docs but ensure artifacts are preserved or restored if build fails
# We will just overwrite.
rm -rf "$DOCS_DIR"/*
cp -r "$WEB_DIR/dist/"* "$DOCS_DIR/"

# 4. Disable Jekyll (Crucial for serving raw .md files and _assets)
touch "$DOCS_DIR/.nojekyll"

echo "✅ Deployment complete! open docs/index.html to test."


