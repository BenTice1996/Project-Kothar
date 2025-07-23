library(jsonlite)
library(dplyr)
library(readr)
library(purrr)

# Load the CSV file
file_path <- "#Add your file path here"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# Function to extract opinion text
extract_opinion_text <- function(json_string) {
  parsed_json <- fromJSON(json_string)
  
  # Extract the first non-empty text field
  fields <- c("plain_text", "html", "html_lawbox", "html_columbia", "html_anon_2020", "xml_harvard")
  for (field in fields) {
    if (!is.null(parsed_json[[field]]) && parsed_json[[field]] != "") {
      return(parsed_json[[field]])
    }
  }
  return(NA)  # Return NA if all fields are empty
}

# Apply function to extract opinion text
df <- df %>%
  mutate(opinion_text = map_chr(response_data, ~ extract_opinion_text(.)))

# Remove extra text
df <- df %>%
  mutate(opinion_text = gsub("<.*?>", "", opinion_text))

# Save the modified dataframe if needed
write_csv(df, "#Add your file path here") 
