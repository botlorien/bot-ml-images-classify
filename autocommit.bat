@echo off
setlocal enabledelayedexpansion

:: Navigate to your repository directory
::cd "path\to\your\git\repository"
cd /d %~dp0
:: Pull the latest changes from the origin. 
:: This will ensure your local repository is up-to-date with the remote before you make new commits.
git pull origin

:: Add all changes to staging area
git add .

:: Commit the changes. You can modify the commit message as needed
git commit -m "Automated commit message"

:: Push to the remote repository
git push origin

:: If using a different branch than 'master', replace 'master' with your branch name.

echo Commit and push completed!

