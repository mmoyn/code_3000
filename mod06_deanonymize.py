import pandas as pd

def load_data(anonymized_path, auxiliary_path):
  """
  Load anonymized and auxiliary datasets.
  """
  anon = pd.read_csv(anonymized_path)
  aux = pd.read_csv(auxiliary_path)
  return anon, aux

def link_records(anon_df, aux_df):
  """
  Attempt to link anonymized records to auxiliary records
  using exact matching on quasi-identifiers.

  Returns a DataFrame with columns:
    anon_id, matched_name
  containing ONLY uniquely matched records.
  """
  anon_df_unique = anon_df.drop_duplicates(subset=["age","zip3","gender"], keep=False)
  aux_df_unique = aux_df.drop_duplicates(subset=["age","zip3","gender"], keep=False)

  res_list = []
  for row_anon in anon_df_unique.itertuples():
    for row_aux in aux_df_unique.itertuples():
      # item 0 is label, item 1 is anon_id/name, items 2+ are data
      if (row_anon[2:] == row_aux[2:]):
        res_list.append( (row_anon[1], row_aux[1]) )

  return pd.DataFrame(data=res_list, columns=["anon_id","matched_name"])

def deanonymization_rate(matches_df, anon_df):
  """
  Compute the fraction of anonymized records
  that were uniquely re-identified.
  """
  return matches_df.shape[0]/anon_df.shape[0]
