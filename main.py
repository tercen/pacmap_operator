from tercen.client import context as ctx
import numpy as np
import polars as pl
import math
from operator_funcs import reshape_data, apply_pacmap, format_results

# Create Tercen context
#http://127.0.0.1:5400/thiago.monteiro/w/e46272a348d3feaa384a4a9d8407773f/ds/384afa63-6ebc-4c6b-95a9-bff66bafb046
# tercenCtx = ctx.TercenContext(  workflowId="e46272a348d3feaa384a4a9d8407773f", stepId="384afa63-6ebc-4c6b-95a9-bff66bafb046" )
tercenCtx = ctx.TercenContext(  )

# Get operator properties
# Issue #1 --> Missed the value type
n_components = tercenCtx.operator_property("n_components", typeFn=int, default=2)
n_neighbors = tercenCtx.operator_property("n_neighbors", default=10, typeFn=int)
MN_ratio = tercenCtx.operator_property("MN_ratio", default=0.5, typeFn=float)
FP_ratio = tercenCtx.operator_property("FP_ratio", default=2.0, typeFn=float)

# n_neighbors = int(10 + 15 * (math.log10(70000)-4))
# n_neighbors = int(5)

# Get data from Tercen
# Issue #2 --> .ci and .ri are added as indices
main_data = tercenCtx.select(['.y', '.ci', '.ri'], df_lib="polars")
col_data = tercenCtx.cselect(['logicle..event_id'], df_lib="polars")
col_data = col_data.with_columns(pl.Series(name=".ci", values=range(len(col_data)), dtype=pl.Int32))
row_data = tercenCtx.rselect(['logicle..channel_name', 'logicle..channel_description'], df_lib="polars")
row_data = row_data.with_columns(pl.Series(name=".ri", values=range(len(row_data)), dtype=pl.Int32))

# Process the data
# Issue #3: missed the pyarrow requirement
matrix_data = reshape_data(main_data, col_data, row_data)
embedding = apply_pacmap(matrix_data, n_components, n_neighbors, MN_ratio, FP_ratio)
result_df = format_results(embedding, col_data)

# Add namespace and save results
result_df = tercenCtx.add_namespace(result_df)
result_df = result_df.astype({'pacmap.PaCMAP_1': 'float', 'pacmap.PaCMAP_2': 'float'})
tercenCtx.save(result_df)