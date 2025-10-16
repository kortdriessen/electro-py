def merge_consecutive_labels(df, label_col='label'):
    """merges rows with consecutive state labels, so that you end up with a single row with a single start_s and end_s for each state

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe, which needs columns 'start_s' and 'end_s', plus the label_col
    label_col : str, optional
        name of the column that contains the labels, by default 'label'
    """
    if df.empty:
        return df
    
    # Sort by start_s to ensure proper order
    df_sorted = df.sort_values('start_s').reset_index(drop=True)
    
    # Create a group identifier for consecutive labels
    df_sorted['group'] = (df_sorted[label_col] != df_sorted[label_col].shift()).cumsum()
    
    # Group by the label and group identifier, then aggregate
    merged_df = df_sorted.groupby([label_col, 'group']).agg({
        'start_s': 'first',
        'end_s': 'last'
    }).reset_index()
    
    # Drop the temporary group column
    merged_df = merged_df.drop('group', axis=1)
    
    # Sort by start_s for final output
    merged_df = merged_df.sort_values('start_s').reset_index(drop=True)
    
    return merged_df