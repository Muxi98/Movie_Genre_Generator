import os
import pandas as pd

# Define paths
input_parquet_path = "metadata_one_hot_final.parquet"
output_parquet_path = "metadata_one_hot_filtered.parquet"
img_dir = "Posters"

# 1. Load the parquet metadata file
if not os.path.exists(input_parquet_path):
    raise FileNotFoundError(f"Input parquet file not found at: {input_parquet_path}")

df = pd.read_parquet(input_parquet_path)
initial_row_count = len(df)
print(f"Original Parquet rows: {initial_row_count}")

# 2. Check for image existence
def image_exists(row_id):
    img_path = os.path.join(img_dir, f"{int(row_id)}.jpg")
    return os.path.exists(img_path)

# Filter the DataFrame to keep only rows with existing images
exists_mask = df['id'].apply(image_exists)
filtered_df = df[exists_mask].reset_index(drop=True)

final_row_count = len(filtered_df)
matched_rows = final_row_count
missing_rows = initial_row_count - final_row_count

# 3. Save to the new parquet file
filtered_df.to_parquet(output_parquet_path)

print(f"\n--- Process Completed ---")
print(f"Rows with matching images: {matched_rows}")
print(f"Rows with missing images: {missing_rows}")
print(f"Saved the filtered dataset to: '{output_parquet_path}'")
print(f"The original file '{input_parquet_path}' remains untouched.")


# =============================================
# GENRE FILTERING & DATA CLEANUP
# =============================================

# 1. Define the 12 genres you want to keep
KEEP_GENRES = [
    'Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Horror',
    'Crime', 'Adventure', 'Science Fiction', 'Fantasy', 'Animation', 'Mystery'
]

# 2. Load the current filtered dataset
input_path = "metadata_one_hot_filtered.parquet"
output_path = "metadata_one_hot_12genres.parquet"
img_dir = "Posters"

df = pd.read_parquet(input_path)
print(f"Original dataset: {len(df)} rows, {len(df.columns) - 1} genres")
print(f"All genre columns: {[c for c in df.columns if c != 'id']}")

# 3. Check which of the 12 genres actually exist as columns
genre_cols_available = [g for g in KEEP_GENRES if g in df.columns]
missing_genres = [g for g in KEEP_GENRES if g not in df.columns]

if missing_genres:
    print(f"\n These genres were not found as columns: {missing_genres}")
    print("Proceeding with the ones that exist.")

print(f"\nKeeping {len(genre_cols_available)} genres: {genre_cols_available}")

# 4. Keep only 'id' + the selected genre columns
df_filtered = df[['id'] + genre_cols_available].copy()

# 5. Remove rows where ALL genre labels are 0 (no genre assigned)
all_zero_mask = (df_filtered[genre_cols_available].sum(axis=1) == 0)
rows_to_remove = df_filtered[all_zero_mask]
print(f"\nRows with all-zero labels (to be removed): {len(rows_to_remove)}")

df_cleaned = df_filtered[~all_zero_mask].reset_index(drop=True)
print(f"Rows remaining after cleanup: {len(df_cleaned)}")

# 6. Delete images that no longer have any genre label
removed_count = 0
for _, row in rows_to_remove.iterrows():
    img_path = os.path.join(img_dir, f"{int(row['id'])}.jpg")
    if os.path.exists(img_path):
        os.remove(img_path)
        removed_count += 1

print(f"Deleted {removed_count} orphaned images from '{img_dir}/'")

# 7. Save the cleaned dataset
df_cleaned.to_parquet(output_path, index=False)
print(f"\n Saved cleaned dataset to '{output_path}'")

# 8. Show genre distribution
print("\n Genre Distribution:")
genre_counts = df_cleaned[genre_cols_available].sum().sort_values(ascending=False)
for genre, count in genre_counts.items():
    pct = count / len(df_cleaned) * 100
    print(f"  {genre:20s} -> {int(count):>6,} samples ({pct:.1f}%)")







# 1. Load the parquet file
file_path = 'metadata_one_hot_12genres.parquet' 
df = pd.read_parquet(file_path)

# 2. Identify the genre columns. 
# Replace this list with the actual exact names of your 12 genre columns.
genre_cols = [
    'Action', 'Comedy', 'Drama', 'Horror', 'Thriller', 'Romance', 
    'Crime', 'Adventure', 'Science Fiction', 'Fantasy', 'Animation', 'Mystery'
]

# 3. Create a helper function that turns the 1s and 0s into a string list of genres
def get_genre_combination(row):
    # This looks at the row and keeps the names of the genres that have a '1'
    active_genres = [genre for genre in genre_cols if row[genre] == 1]
    
    # If a movie has no genres (which shouldn't happen, but just in case)
    if not active_genres:
        return "NO_GENRE"
        
    # Joins them with a comma (e.g., "Action, Thriller, Science Fiction")
    return ", ".join(active_genres)

# 4. Apply the function to create a new column representing the combination
print("Calculating combinations... this might take a few seconds.")
df['genre_combination'] = df.apply(get_genre_combination, axis=1)

# 5. Count the occurrences of each unique combination
combination_counts = df['genre_combination'].value_counts()

# 6. Display the results!
print(f"Total unique combinations found: {len(combination_counts)}\n")

print("Top 20 most frequent genre combinations:")
print("-" * 50)
print(combination_counts.head(20))

# Optional: If you want to see combinations that only appear exactly 1 time
# single_combinations = combination_counts[combination_counts == 1]
# print(f"\nThere are {len(single_combinations)} combinations that only appear exactly 1 time.")
