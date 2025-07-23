# Load relevant libraries
library(readr)
library(dplyr)

# Load merged
merged_df <- read_csv("#Add your file path here")

# Find missing cases
missing_case <- merged_df %>%
  filter(is.na(opinion_text))

merged_df_new <- merged_df %>%
  filter(!is.na(opinion_text))

write_csv(merged_df_new, "#Add your file path here")

colnames(merged_df)

comp <- read_csv("#Add your file path here")

colnames(comp)