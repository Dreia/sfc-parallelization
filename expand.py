import pandas as pd

# Load the CSV file
filename = 'multiple_lane_changes.csv'
df = pd.read_csv(filename, delimiter=';')

# Function to recalculate timestamps and repeat the dataset
def update_timestamps(dataframe, repeat_count):
    final_df = pd.DataFrame()
    original_sec_interval = dataframe['sampleTimeStamp.seconds'].iloc[-1] - dataframe['sampleTimeStamp.seconds'].iloc[0]
    original_micro_interval = dataframe['sampleTimeStamp.microseconds'].iloc[-1] - dataframe['sampleTimeStamp.microseconds'].iloc[0]
    num_rows = len(dataframe)
    sec_interval_per_row = original_sec_interval / (num_rows - 1)
    micro_interval_per_row = original_micro_interval / (num_rows - 1)

    for i in range(repeat_count):
        temp_df = dataframe.copy()
        sec_increment = i * original_sec_interval
        micro_increment = i * original_micro_interval

        temp_df['sampleTimeStamp.seconds'] = dataframe['sampleTimeStamp.seconds'] + sec_increment + (temp_df.index * sec_interval_per_row)
        temp_df['sampleTimeStamp.microseconds'] = dataframe['sampleTimeStamp.microseconds'] + micro_increment + (temp_df.index * micro_interval_per_row)

        # Normalize microseconds if they exceed 1000000 (roll over to seconds)
        overflow = temp_df['sampleTimeStamp.microseconds'] // 1000000
        temp_df['sampleTimeStamp.seconds'] += overflow
        temp_df['sampleTimeStamp.microseconds'] %= 1000000

        final_df = pd.concat([final_df, temp_df], ignore_index=True)

    return final_df

def generate_filename(original_filename, repeat_count):
    # Remove the file extension and add the repeat count and file extension
    base_name = original_filename.split('.csv')[0]
    new_filename = f"{base_name}_x{repeat_count}.csv"
    return new_filename

# Repeat and update the data 1000 times
repeat_count = 1000
result_df = update_timestamps(df, repeat_count)

# Generate the new filename
new_filename = generate_filename(filename, repeat_count)

# Save the expanded and updated dataframe to a new CSV file
result_df.to_csv(new_filename, index=False, sep=';')

print("File saved successfully as:", new_filename)
