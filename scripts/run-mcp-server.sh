#!/usr/bin/env bash
set -euo pipefail

# Script to run PySearch MCP servers
# Usage: ./scripts/run-mcp-server.sh [server_name]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Available servers
declare -A SERVERS=(
    ["pysearch"]="mcp/servers/pysearch_mcp_server.py"
    ["fastmcp"]="mcp/servers/pysearch_mcp_server.py"
    ["default"]="mcp/servers/pysearch_mcp_server.py"
)

show_help() {
    echo "Usage: $0 [server_name]"
    echo ""
    echo "Available servers:"
    for server in "${!SERVERS[@]}"; do
        echo "  $server - ${SERVERS[$server]}"
    done
    echo ""
    echo "If no server is specified, the pysearch server will be used (default)."
}

# Parse arguments
SERVER_NAME="${1:-pysearch}"

if [[ "$SERVER_NAME" == "-h" || "$SERVER_NAME" == "--help" ]]; then
    show_help
    exit 0
fi

# Check if server exists
if [[ ! -v SERVERS[$SERVER_NAME] ]]; then
    echo "Error: Unknown server '$SERVER_NAME'"
    echo ""
    show_help
    exit 1
fi

SERVER_PATH="${SERVERS[$SERVER_NAME]}"
FULL_PATH="$PROJECT_ROOT/$SERVER_PATH"

# Check if server file exists
if [[ ! -f "$FULL_PATH" ]]; then
    echo "Error: Server file not found: $FULL_PATH"
    exit 1
fi

echo "🚀 Starting PySearch MCP Server: $SERVER_NAME"
echo "📁 Server path: $SERVER_PATH"
echo "🔌 Running on stdio transport..."
echo ""

# Change to project root and run server
cd "$PROJECT_ROOT"
python "$SERVER_PATH"
