# Install required packages if not already installed
if (!requireNamespace("httr2", quietly = TRUE)) {
  install.packages("httr2")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("stringdist", quietly = TRUE)) {
  install.packages("stringdist")
}

# Load required packages
library(httr2)
library(jsonlite)
library(dplyr)
library(stringdist)

# Set your API Token
api_key = "" #Insert your API key here

# Set base API URL
search_url <- "https://www.courtlistener.com/api/rest/v4/search/"

# Initialize variables
all_results <- list()  # Store all API responses
next_url <- search_url  # Start with base search URL

# Max retries for handling transient errors
max_retries <- 5
retry_count <- 0

# Loop to handle pagination
repeat {
  # Try-Catch block to handle errors
  tryCatch({
    # API request with delay
    response <- request(next_url) %>%
      req_method("GET") %>%
      req_headers("Authorization" = paste("Token", api_key)) %>%
      req_url_query(
        # Can alter the search in the below line to retreive a different set of cases
        q = "(copyright AND !infring) OR (\"copy right\"~2 AND !infring)",
        type = "o"
      ) %>%
      req_perform()
    
    # Convert response to JSON
    request_results <- response %>% resp_body_json()
    
    # Reset retry counter on success
    retry_count <- 0
    
    # Print response structure for debugging
    print(str(request_results))
    
    # Store the results
    if ("results" %in% names(request_results)) {
      all_results <- append(all_results, request_results$results)
    } else {
      print("Warning: 'results' not found in response.")
    }
    
    # Check if there's a next page
    if (!"next" %in% names(request_results) || is.null(request_results$"next")) {
      print("No more pages available. Exiting loop.")
      break  # Exit loop when no more pages
    }
    
    # Update next URL for the following request
    next_url <- request_results$"next"
    
    # Sleep to avoid overloading API
    Sys.sleep(2)
    
  }, error = function(e) {
    message("Error encountered: ", e$message)
    
    if (grepl("502 Bad Gateway", e$message) && retry_count < max_retries) {
      retry_count <- retry_count + 1
      message("Retrying... (", retry_count, "/", max_retries, ")")
      Sys.sleep(5)  # Wait before retrying
    } else {
      stop("API request failed after multiple retries.")
    }
  })
}

# Save as JSON
write_json(all_results, "", pretty = TRUE) #Insert your file path here



