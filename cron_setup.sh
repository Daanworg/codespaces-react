#!/bin/bash
# Setup cron jobs for both automated processing and backup

# Create a temporary crontab file
TEMP_CRONTAB=$(mktemp)

# Export current crontab
crontab -l > "$TEMP_CRONTAB" 2>/dev/null || echo "# Creating new crontab" > "$TEMP_CRONTAB"

# Add job for processing bucket files (runs daily at 2:00 AM)
if ! grep -q "auto_process_bucket.py" "$TEMP_CRONTAB"; then
  echo "0 2 * * * cd $PWD && python auto_process_bucket.py >> processing_log.txt 2>&1" >> "$TEMP_CRONTAB"
  echo "Added scheduled processing job"
fi

# Add job for backing up code (runs every 6 hours)
if ! grep -q "auto_backup.sh" "$TEMP_CRONTAB"; then
  echo "0 */6 * * * cd $PWD && bash auto_backup.sh >> backup_log.txt 2>&1" >> "$TEMP_CRONTAB"
  echo "Added scheduled backup job"
fi

# Install new crontab
crontab "$TEMP_CRONTAB"
rm "$TEMP_CRONTAB"

echo "Cron jobs have been set up successfully!"
echo "- auto_process_bucket.py will run daily at 2:00 AM"
echo "- auto_backup.sh will run every 6 hours"

# Show current crontab
echo -e "\nCurrent crontab:"
crontab -l