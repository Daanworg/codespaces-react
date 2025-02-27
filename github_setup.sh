#!/bin/bash
# Configure GitHub repository for automatic backups

# Check if git is installed
if ! command -v git &>/dev/null; then
    echo "Git is not installed. Please install git and try again."
    exit 1
fi

# Set default values
DEFAULT_REPO_NAME="cloud-rag-webhook"
DEFAULT_GITHUB_USER=""

# Prompt for GitHub username if not provided
read -p "Enter your GitHub username [$DEFAULT_GITHUB_USER]: " GITHUB_USER
GITHUB_USER=${GITHUB_USER:-$DEFAULT_GITHUB_USER}

# Prompt for repository name if not provided
read -p "Enter GitHub repository name [$DEFAULT_REPO_NAME]: " REPO_NAME
REPO_NAME=${REPO_NAME:-$DEFAULT_REPO_NAME}

# Check if personal access token is already set
if [[ -z $(git config --global --get github.token) ]]; then
    echo "You need a GitHub Personal Access Token for authentication."
    echo "Generate one at: https://github.com/settings/tokens"
    echo "Ensure it has 'repo' permissions."
    read -p "Enter your GitHub Personal Access Token: " GITHUB_TOKEN
    
    # Store the token (optional)
    read -p "Would you like to store this token for future use? (y/n): " STORE_TOKEN
    if [[ $STORE_TOKEN == "y" ]]; then
        git config --global github.token "$GITHUB_TOKEN"
    fi
else
    echo "GitHub token already configured."
fi

# Initialize git repository if needed
if [ ! -d ".git" ]; then
    git init
    echo "Initialized git repository"
fi

# Set up git user information if not already configured
if [[ -z $(git config --get user.email) ]]; then
    read -p "Enter your email for git commits: " GIT_EMAIL
    git config user.email "$GIT_EMAIL"
fi

if [[ -z $(git config --get user.name) ]]; then
    read -p "Enter your name for git commits: " GIT_NAME
    git config user.name "$GIT_NAME"
fi

# Check if remote origin exists, and create if it doesn't
if ! git remote | grep -q "^origin$"; then
    REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"
    git remote add origin "$REPO_URL"
    echo "Added remote: $REPO_URL"
else
    echo "Remote 'origin' already exists. Current remote URL:"
    git remote get-url origin
    
    read -p "Do you want to update the remote URL? (y/n): " UPDATE_REMOTE
    if [[ $UPDATE_REMOTE == "y" ]]; then
        REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"
        git remote set-url origin "$REPO_URL"
        echo "Updated remote to: $REPO_URL"
    fi
fi

# Initial commit and push
echo "Creating initial commit and pushing to GitHub..."
git add .
git commit -m "Initial commit: Setting up cloud-rag-webhook"

# Check if the repository exists on GitHub first
echo "Pushing to GitHub..."
if git push -u origin master; then
    echo "Successfully pushed to GitHub!"
else
    echo "Push failed. The repository might not exist on GitHub."
    echo "Please create the repository at: https://github.com/new"
    echo "Repository name: $REPO_NAME"
    read -p "Have you created the repository? (y/n): " REPO_CREATED
    
    if [[ $REPO_CREATED == "y" ]]; then
        echo "Trying to push again..."
        git push -u origin master
    else
        echo "Please create the repository and then run: git push -u origin master"
    fi
fi