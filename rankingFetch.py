import requests
import csv

# Your TBA API key
API_KEY = 'YOUR_KEY'

# The event key for which you want rankings
EVENT_KEY = '2024nhgrs'  # Replace this with your actual event key

# Define the headers for authorization
headers = {
    'X-TBA-Auth-Key': API_KEY
}

# Fetch the rankings data from The Blue Alliance API
url = f'https://www.thebluealliance.com/api/v3/event/{EVENT_KEY}/rankings'
response = requests.get(url, headers=headers)

if response.status_code == 200:
    rankings_data = response.json()

    # Open or create a CSV file
    with open('team_rankings.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(['teamNumber', 'rank'])

        # Loop through the rankings and write them to the CSV
        for team in rankings_data['rankings']:
            team_number = team['team_key'].replace('frc', '')  # Removes 'frc' prefix from team number
            rank = team['rank']
            writer.writerow([team_number, rank])

    print("CSV file 'team_rankings.csv' has been created.")
else:
    print(f"Failed to fetch data: {response.status_code}")
