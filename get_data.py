import pandas as pd
import requests
from bs4 import BeautifulSoup
import concurrent.futures

# Global variable to track the successful fetch count
i = 0

# Function to clean text
def clean_text(text):
    return ' '.join(text.replace('\\n', ' ')
                     .replace('\\r', ' ')
                     .replace('\\t', ' ')
                     .split())

# Function to fetch the text of a book from its URL
def fetch_text(url):
    global i  # Declare i as global to modify it
    try:
        # Fetch the HTML content of the book's webpage
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for a bad HTTP status
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to find the link to the plain text version
        text_link = soup.find("a", string="Plain Text UTF-8")
        
        if text_link:
            # Construct the full URL for the plain text version
            text_url = 'http://www.gutenberg.org' + text_link['href']
            text_response = requests.get(text_url)
            text_response.raise_for_status()

            i += 1  # Increment the count for successful fetch
            print(f"Successful fetch count: {i}")
            
            # Extract and clean the raw text
            raw_text = text_response.text
            return clean_text(raw_text)
        else:
            return None  # No plain text link found
    except requests.exceptions.RequestException as e:
        print(f"Error fetching text from {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

# Function to process each row in the CSV and add text
def process_row(row):
    try:
        # Extract the link to the book from the row
        book_url = row['Link']
        # Fetch the book text
        text = fetch_text(book_url)
        # Return the row with the text added
        return {**row.to_dict(), 'Text': text}
    except Exception as e:
        print(f"Error processing row {row['Title']}: {e}")
        return None

# Main function to load data, scrape texts, and save the results
def main():
    # Load the CSV file
    df = pd.read_csv('gutenberg_metadata.csv')
    
    # Prepare a list to hold the results
    results = []
    
    # Use ThreadPoolExecutor to process rows concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    # Convert results to a DataFrame
    df_result = pd.DataFrame(results)
    
    # Save the result to a new CSV file
    df_result.to_csv('gutenberg_data_with_text.csv', index=False)
    print(f"Data saved to 'gutenberg_data_with_text.csv'")

# Run the main function
if __name__ == '__main__':
    main()
