# Load required packages
library(httr2)
library(jsonlite)
library(dplyr)
library(readr)

# === CONFIG ===
api_key <- "b58f4ded325462cccf1226ae55d7c228f1032978"
opinions_url <- "https://www.courtlistener.com/api/rest/v4/opinions/"
json_file_path <- "C:/Users/Ben Tice/Desktop/search_api_results.json"
csv_file_path <- "C:/Users/Ben Tice/Desktop/court_api_results.csv"

# === Load or create responses dataframe ===
if (file.exists(csv_file_path)) {
  cases_api_responses_df <- read_csv(csv_file_path, show_col_types = FALSE) %>%
    mutate(case_id = as.character(case_id))
  processed_ids <- cases_api_responses_df$case_id
  message("✅ Resuming from existing CSV.")
} else {
  cases_api_responses_df <- tibble(case_id = character(), response_data = character())
  processed_ids <- character()
  message("🆕 Starting fresh.")
}

# === Load JSON input ===
cases <- fromJSON(json_file_path)
cases$cluster_id <- as.character(cases$cluster_id)
cases$opinions <- sapply(cases$opinions, function(x) paste(x, collapse = ","))

# === Loop through cases ===
for (i in seq_along(cases$cluster_id)) {
  case_id <- cases$cluster_id[i]
  
  # Skip invalid or already processed
  if (is.na(case_id) || case_id == "" || case_id %in% processed_ids) {
    next
  }
  
  message(paste("🔍 Checking Case ID:", case_id))
  
  # HEAD request with error handling
  head_status <- tryCatch({
    head_request <- request(paste0(opinions_url, case_id)) |>
      req_method("HEAD") |>
      req_headers("Authorization" = paste("Token", api_key))
    head_response <- req_perform(head_request)
    resp_status(head_response)
  }, error = function(e) {
    if (grepl("404", conditionMessage(e))) {
      message(paste("❌ Opinion not found (404) for Case ID:", case_id))
      return(404)
    } else {
      message(paste("⚠️ HEAD request error for Case ID:", case_id, "|", conditionMessage(e)))
      return(NA)
    }
  })
  
  if (is.na(head_status) || head_status == 404) {
    next
  }
  
  Sys.sleep(1)
  
  # GET request with error handling
  response_data <- tryCatch({
    response <- request(paste0(opinions_url, case_id)) |>
      req_method("GET") |>
      req_headers("Authorization" = paste("Token", api_key)) |>
      req_perform()
    
    if (resp_status(response) == 200) {
      resp_body_string(response)
    } else {
      message(paste("❌ GET failed for Case ID:", case_id, "| Status:", resp_status(response)))
      return(NA)
    }
  }, error = function(e) {
    message(paste("⚠️ GET request error for Case ID:", case_id, "|", conditionMessage(e)))
    return(NA)
  })
  
  if (!is.na(response_data)) {
    new_row <- tibble(case_id = case_id, response_data = as.character(response_data))
    cases_api_responses_df <- bind_rows(cases_api_responses_df, new_row)
    write_csv(cases_api_responses_df, csv_file_path)  # Save after each successful call
    message(paste("✅ Stored and saved Case ID:", case_id))
  }
  
  Sys.sleep(1)
}

message("🎉 All case data processed and saved!")
