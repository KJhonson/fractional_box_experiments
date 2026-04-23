#!/bin/bash

# Jupyter Startup Script for DOLFINx Environment
# This script helps you start Jupyter Notebook or JupyterLab easily

echo "🚀 Jupyter Startup Script for DOLFINx Environment"
echo "=================================================="
echo ""

# Function to start Jupyter Notebook
start_notebook() {
    echo "Starting Jupyter Notebook..."
    echo "📝 Access at: http://localhost:8888"
    echo "🛑 Press Ctrl+C to stop"
    echo ""
    jupyter notebook \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --notebook-dir=/home/dolfinx/shared
}

# Function to start JupyterLab
start_lab() {
    echo "Starting JupyterLab..."
    echo "📝 Access at: http://localhost:8888"
    echo "🛑 Press Ctrl+C to stop"
    echo ""
    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --notebook-dir=/home/dolfinx/shared
}

# Function to show help
show_help() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  notebook, nb    Start Jupyter Notebook (classic interface)"
    echo "  lab, jlab       Start JupyterLab (modern interface)"
    echo "  help, -h        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 notebook     # Start classic Jupyter Notebook"
    echo "  $0 lab          # Start JupyterLab"
    echo "  $0              # Interactive menu"
}

# Main logic
case "$1" in
    "notebook"|"nb")
        start_notebook
        ;;
    "lab"|"jlab")
        start_lab
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    "")
        # Interactive menu
        echo "Choose your Jupyter interface:"
        echo "1) Jupyter Notebook (classic)"
        echo "2) JupyterLab (modern)"
        echo "3) Help"
        echo "4) Exit"
        echo ""
        read -p "Enter your choice (1-4): " choice
        
        case $choice in
            1)
                start_notebook
                ;;
            2)
                start_lab
                ;;
            3)
                show_help
                ;;
            4)
                echo "Goodbye! 👋"
                exit 0
                ;;
            *)
                echo "❌ Invalid choice. Please run '$0 help' for usage information."
                exit 1
                ;;
        esac
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac
