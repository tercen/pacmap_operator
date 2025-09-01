import polars as pl
import pandas as pd
import numpy as np
import pacmap

def reshape_data(main_data, col_data, row_data):
    """
    Reshape the long-format data into a matrix suitable for PaCMAP.
    
    Args:
        main_data: Polars DataFrame with .y, .ci, .ri columns
        col_data: Polars DataFrame with eventId, .ci columns
        row_data: Polars DataFrame with pixel_id, .ri columns
        
    Returns:
        numpy.ndarray: Matrix where each row is an event (image) and each column is a pixel
    """
    # Join main data with column and row data
    df = (main_data
          .join(col_data, on='.ci', how='left')
          .join(row_data, on='.ri', how='left'))
    
    # Convert to wide format where each row is an event and each column is a pixel
    pivot_df = df.pivot(
        index='logicle..event_id',
        columns=['logicle..channel_name', 'logicle..channel_description'],
        values='.y'
    )
    
    # Convert to numpy array for PaCMAP
    # issue 5: Must remove eventId
    matrix = pivot_df.drop('logicle..event_id').to_numpy()
    
    # Fill NaN values with 0 (if any)
    # matrix = np.nan_to_num(matrix, nan=0.0)
    
    return matrix

def apply_pacmap(data_matrix, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0):
    """
    Apply PaCMAP dimensionality reduction.
    
    Args:
        data_matrix: numpy.ndarray where each row is an event and each column is a pixel
        n_components: Number of dimensions in the embedding
        n_neighbors: Number of neighbors for the kNN graph
        MN_ratio: Ratio of mid-near pairs to be sampled
        FP_ratio: Ratio of further pairs to be sampled
        
    Returns:
        numpy.ndarray: Reduced dimensionality embedding
    """
    # Initialize PaCMAP
    embedding = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio
    )
    
    # Create wide format data matrix - transpose so each pixel_id is a column
   
    # Fit and transform the data
    # Issue #4 : Missing init pca
    
    result = embedding.fit_transform(data_matrix, init='pca')

    return result

def format_results(embedding, col_data):
    """
    Format the PaCMAP results for Tercen.
    
    Args:
        embedding: numpy.ndarray with the reduced dimensionality data
        col_data: Polars DataFrame with eventId, .ci columns
        
    Returns:
        pandas.DataFrame: DataFrame in Tercen format with .ci, .ri, and component values
    """
    # Create a DataFrame with the embedding results
    result_df = pd.DataFrame(embedding)
    
    # Rename columns to indicate PaCMAP components
    renamed_cols = {i: f"PaCMAP_{i+1}" for i in range(embedding.shape[1])}
    result_df = result_df.rename(columns=renamed_cols)
    
    # Add eventId from col_data
    unique_events_col = col_data.select(['.ci']).to_pandas()
    # unique_events_row = col_data.select(['.ri']).to_pandas()
    # result_df['eventId'] = unique_events['eventId'].values
    result_df['.ci'] = unique_events_col['.ci'].values
    
    # Add .ri column (set to 0 as we have one row per event)
    result_df['.ri'] =  0#unique_events_row['.ri'].values
    
    # Convert .ci and .ri to integers
    result_df = result_df.astype({".ci": np.int32, ".ri": np.int32})
    
    return result_df