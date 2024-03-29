import csv

# Function to update CSV file with start times for specific IDs
def update_csv_with_start_times(csv_file, id_list, start_time):
    # Read existing data from CSV file
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Update Out_Time for specified IDs
    for row in data:
        if int(row['IDs']) in id_list:
            row['Out_Time'] = str(start_time)

    # Write updated data back to CSV file
    with open(csv_file, 'w', newline='') as file:
        fieldnames = ['IDs', 'In_Time', 'Out_Time']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Test the function
csv_file = 'persons_data.csv'
id_list = [1, 3, 4]
start_time = 10
update_csv_with_start_times(csv_file, id_list, start_time)
print("CSV file updated successfully.")
