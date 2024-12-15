import json
import csv
import os
from datetime import datetime

# Input folder containing JSON files and output directory for CSV files
json_folder_path = r'/Users/pateldhrit/Desktop/Hadoop-Project/all_json'  # Folder with JSON files
output_folder_path = r'/Users/pateldhrit/Desktop/Hadoop-Project/output'  # Folder to save CSV files
os.makedirs(output_folder_path, exist_ok=True)  # Create the output folder if it doesn't exist

# CSV columns
columns = [
    "match_id", "date", "team1", "team2", "inning", "over", "ball",
    "batsman", "bowler", "non_striker", "runs_batsman", "runs_extras",
    "runs_total", "wicket_type", "wicket_player", "extras_type",
    "match_type", "gender", "batting_team", "bowling_team"
]

# Configurable chunk size for rows per CSV file
chunk_size = 10000  # Number of rows per CSV file
file_index = 1  # Counter for CSV file naming
row_count = 0  # Counter for rows written

# Open the first CSV file
csv_file = open(os.path.join(output_folder_path, f'cricket_data_part_{file_index}.csv'), mode='w', newline='')
writer = csv.DictWriter(csv_file, fieldnames=columns)
writer.writeheader()

# Process all JSON files in the folder
for json_file_name in os.listdir(json_folder_path):
    if json_file_name.endswith('.json'):
        json_file_path = os.path.join(json_folder_path, json_file_name)
        
        with open(json_file_path, 'r') as f:
            match_data = json.load(f)
        
        # Extract metadata from JSON
        match_id = match_data.get("info", {}).get("match_type_number", "unknown")
        date_string = match_data.get("info", {}).get("dates", ["unknown"])[0]
        try:
            date = datetime.strptime(date_string, "%Y-%m-%d").date()
        except ValueError:
            date = "unknown"
        teams = match_data.get("info", {}).get("teams", ["Unknown", "Unknown"])
        team1, team2 = teams if len(teams) == 2 else ("Unknown", "Unknown")
        match_type = match_data.get("info", {}).get("match_type")
        gender = match_data.get("info", {}).get("gender")
        
        # Process innings data
        for inning_data in match_data.get("innings", []):
            batting_team = inning_data.get("team")
            overs = inning_data.get("overs", [])
            
            for over_data in overs:
                over_number = over_data.get("over")
                
                for delivery in over_data.get("deliveries", []):
                    batsman = delivery.get("batter")
                    bowler = delivery.get("bowler")
                    non_striker = delivery.get("non_striker")
                    runs_batsman = delivery["runs"].get("batter", 0)
                    runs_extras = delivery["runs"].get("extras", 0)
                    runs_total = delivery["runs"].get("total", 0)
                    wicket_info = delivery.get("wickets", [{}])[0] if "wickets" in delivery else {}
                    wicket_type = wicket_info.get("kind")
                    wicket_player = wicket_info.get("player_out")
                    extras_type = list(delivery.get("extras", {}).keys())[0] if delivery.get("extras") else None
                    bowling_team = team2 if batting_team == team1 else team1
                    
                    # Write the row to the current CSV
                    writer.writerow({
                        "match_id": match_id,
                        "date": date,
                        "team1": team1,
                        "team2": team2,
                        "inning": batting_team,
                        "over": over_number,
                        "ball": delivery.get("ball"),
                        "batsman": batsman,
                        "bowler": bowler,
                        "non_striker": non_striker,
                        "runs_batsman": runs_batsman,
                        "runs_extras": runs_extras,
                        "runs_total": runs_total,
                        "wicket_type": wicket_type,
                        "wicket_player": wicket_player,
                        "extras_type": extras_type,
                        "match_type": match_type,
                        "gender": gender,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team
                    })
                    
                    row_count += 1

                    # Check if we need a new CSV file
                    if row_count >= chunk_size:
                        csv_file.close()
                        file_index += 1
                        row_count = 0
                        csv_file = open(os.path.join(output_folder_path, f'cricket_data_part_{file_index}.csv'), mode='w', newline='')
                        writer = csv.DictWriter(csv_file, fieldnames=columns)
                        writer.writeheader()

# Close the final CSV file
if csv_file:
    csv_file.close()

print(f"Data successfully split into {file_index} CSV files in {output_folder_path}")
