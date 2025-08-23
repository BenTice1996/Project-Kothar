# Load relevant libraries
library(dplyr)
library(jsonlite)
library(readr)
library(openxlsx)
library(purrr)
library(stringr)

# Load datasets
json_file <- fromJSON("C:/Users/Ben Tice/Desktop/search_api_results.json")
json_cases <- as.data.frame(json_file)

cases_file <- read_csv("C:/Users/Ben Tice/Desktop/kothar_dataset")
csv_cases <- as.data.frame(cases_file)

# Remove shortened opinions from json_cases
json_cases <- json_cases %>%
  rename(case_id = cluster_id) %>%
  select(-opinions)

# Find duplicates
table1 <- table(duplicated(csv_cases$case_id))  # Check for duplicates in csv_cases
table2 <- table(duplicated(json_cases$case_id)) # Check for duplicates in json_cases

# Define safe converters
safe_to_string <- function(x) {
  if (is.null(x) || length(x) == 0) return(NA_character_)
  if (is.atomic(x)) return(toString(x))
  if (is.list(x)) return(toString(unlist(x, recursive = TRUE, use.names = FALSE)))
  if (is.data.frame(x)) return(toString(unlist(x, recursive = TRUE, use.names = FALSE)))
  as.character(x)
}

to_chr_col <- function(col) {
  # data-frame column (df-column)
  if (inherits(col, "data.frame")) {
    if (ncol(col) == 0L) {
      return(rep(NA_character_, nrow(col)))  # empty df-column -> NAs
    }
    # Bundle the row's fields into one list before stringifying
    return(pmap_chr(col, function(...) safe_to_string(list(...))))
    # equivalently: pmap_chr(col, ~ safe_to_string(list(...)))
  }
  # plain list-column
  if (is.list(col)) {
    return(map_chr(col, safe_to_string))
  }
  # atomic but not character
  if (is.atomic(col) && !is.character(col)) {
    return(as.character(col))
  }
  # already character
  if (is.character(col)) {
    return(col)
  }
  # final fallback: coerce each cell
  map_chr(as.list(col), safe_to_string)
}

# Define column name lists
list_1 <- colnames(csv_cases)
list_2 <- colnames(json_cases)

# Apply conversion functions
csv_cases_test <- csv_cases %>%
  mutate(across(all_of(list_1), ~ map_chr(., safe_to_string)))

json_cases_test <- json_cases %>%
  mutate(across(everything(), to_chr_col))

# helper for collapsing per-group
collapse_unique <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(x)]
  if (length(x) == 0) return(NA_character_)
  paste(unique(x), collapse = "; ")
}

# Merge rows with identical cluster ids
csv_cases_test <- csv_cases_test %>%
  dplyr::group_by(case_id) %>%
  dplyr::summarise(
    dplyr::across(dplyr::everything(), collapse_unique),
    .groups = "drop"
  )

json_cases_test <- json_cases_test %>%
  dplyr::group_by(case_id) %>%
  dplyr::summarise(
    dplyr::across(dplyr::everything(), collapse_unique),
    .groups = "drop"
  )

# Merge dataframes
merged_df <- full_join(csv_cases_test, json_cases_test, by = "case_id")

# Log and Remove Empty cases
missing_case <- merged_df %>%
  filter(is.na(opinion_text))

updated_df <- merged_df %>%
  filter(!is.na(opinion_text))

# Remove short cases
updated_df <- updated_df %>%
  filter(9 < str_count(opinion_text, '\\w+'))

write_csv(updated_df, "C:/Users/Ben Tice/Desktop/kothar_dataset_metadata.csv")



